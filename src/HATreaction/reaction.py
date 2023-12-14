import json
from importlib.resources import files as res_files
import logging

import MDAnalysis as MDA
import numpy as np

from HATreaction.utils.trajectory_utils import extract_subsystems, save_capped_systems

from HATreaction.utils.utils import find_radicals
from kimmdy.recipe import Bind, Break, Place, Relax, Recipe, RecipeCollection
from kimmdy.plugins import ReactionPlugin

from pprint import pformat
from tempfile import TemporaryDirectory
import shutil
from pathlib import Path
from tqdm.autonotebook import tqdm


class HAT_reaction(ReactionPlugin):
    def __init__(self, *args, **kwargs):
        logging.getLogger("tensorflow").setLevel("CRITICAL")
        import tensorflow as tf

        logging.getLogger("tensorflow").setLevel("CRITICAL")
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
        logger.debug("Getting recipe for reaction: HAT")

        tpr = str(files.input["tpr"])
        trr = str(files.input["trr"])
        u = MDA.Universe(str(tpr), str(trr))

        se_dir = files.outputdir / "se"
        if not self.config.keep_structures:
            se_dir_bck = se_dir
            se_tmpdir = TemporaryDirectory()
            se_dir = Path(se_tmpdir.name)

        if getattr(self.config, "radicals", None) is not None:
            rad_ids = self.config.radicals
        else:
            # One-based strings in top
            rad_ids = [str(int(i) - 1) for i in self.runmng.top.radicals.keys()]
        if len(rad_ids) < 1:
            logger.debug("No radicals known, searching in structure..")
            rad_ids = [str(a[0].id) for a in find_radicals(u)]
        logger.info(f"Found radicals: {len(rad_ids)}")
        if len(rad_ids) < 1:
            logger.info("--> retuning empty recipe collection")
            return RecipeCollection([])
        rad_ids = sorted(rad_ids)
        sub_atms = u.select_atoms(
            f"((not resname SOL NA CL) and (around 20 id {' '.join([i for i in rad_ids])}))"
            f" or id {' '.join([i for i in rad_ids])}",
            updating=True,
        )
        try:
            # environment around radical is updated by ts incrementation
            logger.info(f"Searching trajectory for radical structures.")
            for ts in tqdm(u.trajectory[:: self.polling_rate]):
                u_sub = MDA.Merge(sub_atms)
                u_sub.trajectory[0].dimensions = ts.dimensions

                # check manually w/ ngl:
                if 0:
                    import nglview as ngl

                    view = ngl.show_mdanalysis(u, defaultRepresentation=False)
                    view.representations = [
                        {"type": "ball+stick", "params": {"sele": ""}},
                        {
                            "type": "spacefill",
                            "params": {"sele": "", "radiusScale": 0.7},
                        },
                    ]
                    view._set_selection("@" + ",".join(rad_ids), repr_index=1)
                    view.center()
                    view

                subsystems = extract_subsystems(
                    u_sub,
                    rad_ids,
                    h_cutoff=self.h_cutoff,
                    env_cutoff=10,
                    start=0,
                    stop=1,
                    step=1,
                    cap=False,
                    rad_min_dist=3,
                    unique=False,
                    logger=logger,
                )
                save_capped_systems(subsystems, se_dir)

            in_ds, es, scale_t, meta_ds, metas_masked = create_meta_dataset_predictions(
                meta_files=list(se_dir.glob("*.npz")),
                batch_size=self.hparas["batchsize"],
                mask_energy=False,
                oneway=True,
            )

            # Make predictions
            logger.info("Making predictions.")
            ys = []
            for model, m, s in zip(self.models, self.means, self.stds):
                y = model.predict(in_ds).squeeze()
                ys.append((y * s) + m)
            ys = np.stack(ys)
            ys = np.mean(np.array(ys), 0)

            # Rate; RT=0.593
            logger.info("Creating Recipes.")
            rates = list(np.multiply(self.freqfac, np.float_power(np.e, (-ys / 0.593))))
            recipes = []
            logger.debug(f"Barriers:\n{pformat(ys)}")
            logger.info(f"Max Rate: {max(rates)}, predicted {len(rates)} rates")
            logger.debug(f"Rates:\n{pformat(rates)}")
            for meta_d, rate in zip(meta_ds, rates):
                ids = [int(i) for i in meta_d["indices"][0:2]]   # should be zero-based
                # assert all(           # what is this for? shouldn't meta_d["indices"][0:2] be a list of ints?
                #     [len(i) == 1 for i in ids]
                # ), f"HAT atom index translation error! \n{meta_d}"

                f1 = meta_d["frame"]
                f2 = meta_d["frame"] + self.polling_rate
                if f2 >= len(u.trajectory):
                    f2 = len(u.trajectory) - 1
                t1 = u.trajectory[f1].time
                t2 = u.trajectory[f2].time
                old_bound = int(u_sub.select_atoms(f"bonded id {ids[0]}")[0].id)

                # get end position
                pdb_e = meta_d["meta_path"].with_name(
                    meta_d["meta_path"].stem + "_2.pdb"
                )
                with open(pdb_e) as f:
                    finished = False
                    while not finished:
                        line = f.readline()
                        if line[:11] == "ATOM      1":
                            finished = True
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())

                if self.config.change_coords == "place":
                    # HAT plugin ids are kimmdy ixs (zero-based,int)
                    seq = [
                        Break(old_bound, ids[0]),
                        Place(ix_to_place=ids[0], new_coords=[x, y, z]),
                        Bind(ids[0], ids[1]),
                    ]
                elif self.config.change_coords == "lambda":
                    seq = [Break(old_bound, ids[0]), Bind(ids[0], ids[1]), Relax()]
                else:
                    raise ValueError(
                        f"Unknown change_coords parameter {self.config.change_coords}"
                    )

                # make recipe
                recipes.append(
                    Recipe(recipe_steps=seq, rates=[rate], timespans=[[t1, t2]])
                )

            recipe_collection = RecipeCollection(recipes)
        except Exception as e:
            # backup in case of failure
            if not self.config.keep_structures:
                shutil.copytree(se_dir, se_dir_bck)
            raise e

        if not self.config.keep_structures:
            se_tmpdir.cleanup()
        return recipe_collection
