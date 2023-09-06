import json
from importlib.resources import files as res_files
import logging

import MDAnalysis as MDA
import numpy as np

from HATreaction.utils.trajectory_utils import extract_subsystems, save_capped_systems

from HATreaction.utils.utils import find_radicals
from kimmdy.recipe import Bind, Break, Place, Relax, Recipe, RecipeCollection
from kimmdy.plugins import ReactionPlugin


class HAT_reaction(ReactionPlugin):
    def __init__(self, *args, **kwargs):
        logging.getLogger("tensorflow").setLevel("CRITICAL")
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        super().__init__(*args, **kwargs)

        # Load model

        if getattr(self.config, "model", None) is None:
            ens_glob = "[!_]*"
        else:
            ens_glob = self.config.model

        ensemble_dir = list(res_files("HATmodels").glob(ens_glob))[0]
        ensemble_size = getattr(self.config, "enseble_size", None)
        self.models = []
        self.means = []
        self.stds = []
        self.hparas = {}
        for model_dir in list(ensemble_dir.glob("*"))[slice(ensemble_size)]:
            tf_model_dir = list(model_dir.glob("*.tf"))[0]
            self.models.append(load_model(tf_model_dir))

            with open(model_dir / "hparas.json") as f:
                hpara = json.load(f)
                self.hparas.update(hpara)

            if hpara.get("scale"):
                with open(model_dir / "scale", "r") as f:
                    mean, std = [float(l.strip()) for l in f.readlines()]
            else:
                mean, std = [0.0, 1.0]
            self.means.append(mean)
            self.stds.append(std)

        self.h_cutoff = self.config.h_cutoff
        self.freqfac = self.config.frequency_factor
        self.polling_rate = self.config.polling_rate

    def get_recipe_collection(self, files) -> RecipeCollection:
        from HATreaction.utils.input_generation import create_meta_dataset_predictions

        logger = files.logger
        logger.debug("Getting recipe for reaction: homolysis")

        tpr = str(files.input["tpr"])
        trr = str(files.input["trr"])
        u = MDA.Universe(str(tpr), str(trr))

        se_dir = files.outputdir / "se"
        # interp_dir = files.outputdir / "interp"

        if getattr(self.config, "radicals", None) is not None:
            rad_idxs = self.config.radicals
        else:
            rad_idxs = getattr(self.runmng, "radical_idxs", [])
        if len(rad_idxs) < 1:
            logger.info("No radicals known, searching in structure..")
            rad_idxs = [str(a[0].index) for a in find_radicals(u)]
        logger.info(f"Found radicals: {rad_idxs}")
        if len(rad_idxs) < 1:
            logger.info(f"--> retuning empty recipe collection")
            return RecipeCollection([])

        sub_atms = u.select_atoms(
            f"((not resname SOL NA CL) and (around 20 index {' '.join([i for i in rad_idxs])}))"
            f" or index {' '.join([i for i in rad_idxs])}",
            updating=True,
        )

        # every 10 rate queries, update environment selection around radical
        for ts in u.trajectory[:: self.polling_rate * 10]:
            u_sub = MDA.Merge(sub_atms)
            u_sub.load_new(str(trr), sub=sub_atms.indices)
            sub_start_t = ts.frame
            sub_end_t = ts.frame + self.polling_rate * 10

            # subuni has different indices, translate: WARNING: id 1-based!!!
            rad_idxs_sub = list(
                map(
                    str,
                    u_sub.select_atoms(f"id {' '.join([i for i in rad_idxs])}").indices,
                )
            )

            # check manually w/ ngl:
            if 0:
                import nglview as ngl

                view = ngl.show_mdanalysis(u_sub, defaultRepresentation=False)
                view.representations = [
                    {"type": "ball+stick", "params": {"sele": ""}},
                    {"type": "spacefill", "params": {"sele": "", "radiusScale": 0.7}},
                ]
                view._set_selection("@" + ",".join(rad_idxs_sub), repr_index=1)
                view.center()
                view

            save_capped_systems(  # maybe just keep in ram? optionally?
                extract_subsystems(
                    u_sub,
                    rad_idxs_sub,
                    h_cutoff=self.h_cutoff,
                    start=sub_start_t,
                    stop=sub_end_t,
                    step=self.polling_rate,
                    unique=False,
                ),
                se_dir,
            )

        in_ds, es, scale_t, meta_ds, metas_masked = create_meta_dataset_predictions(
            meta_files=list(se_dir.glob("*.npz")),
            batch_size=self.hparas["batchsize"],
            mask_energy=False,
            oneway=True,
        )

        # Make predictions
        ys = []
        for model, m, s in zip(self.models, self.means, self.stds):
            y = model.predict(in_ds).squeeze()
            ys.append((y * s) + m)
        ys = np.stack(ys)
        ys = np.mean(np.array(ys), 0)

        # Rate
        rates = list(np.multiply(self.freqfac, np.power(np.e, (-ys / 0.593))))
        recipes = []

        for meta_d, rate in zip(meta_ds, rates):
            sub_idxs = meta_d["indices"][0:2]
            idxs = [u_sub.select_atoms(f"index {sub_idx}").ids for sub_idx in sub_idxs]
            assert all(
                [len(i) == 1 for i in idxs]
            ), f"HAT atom index translation error! \n{meta_d}"
            # idxs = [idx[0] + 1 for idx in idxs] # one-based
            idxs = [int(idx[0]) for idx in idxs]  # zero-based

            f1 = meta_d["frame"] - self.polling_rate
            if f1 < 0:
                f1 = 0
            f2 = meta_d["frame"]
            t1 = u.trajectory[f1].time
            t2 = u.trajectory[f2].time
            old_bound = int(u.select_atoms(f"bonded index {idxs[0]}")[0].index)

            # get end position
            pdb_e = meta_d["meta_path"].with_name(meta_d["meta_path"].stem + "_2.pdb")
            with open(pdb_e) as f:
                finished = False
                while not finished:
                    line = f.readline()
                    if line[:11] == "ATOM      1":
                        finished = True
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())

            # make recipe
            recipes.append(
                Recipe(
                    recipe_steps=[
                        Break(old_bound, idxs[0]),
                        Place(ix_to_place=idxs[0], new_coords=[x, y, z]),
                        Bind(idxs[0], idxs[1]),
                        Relax(),
                    ],
                    # recipe_steps=[Move(ix_to_move=idxs[0], ix_to_bind=idxs[1])],
                    rates=[rate],
                    timespans=[[t1, t2]],
                )
            )

        recipe_collection = RecipeCollection(recipes)
        # recipe_collection.aggregate_reactions() # useless with Place

        return recipe_collection
