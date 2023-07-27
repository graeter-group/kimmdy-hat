import json
from importlib.resources import files as res_files
import logging

import MDAnalysis as MDA
import numpy as np

from HATreaction.utils.trajectory_utils import extract_subsystems, save_capped_systems
from HATreaction.utils.input_generation import create_meta_dataset_predictions
from HATreaction.utils.utils import find_radicals
from kimmdy.reaction import (
    Move,
    Recipe,
    RecipeCollection,
    ReactionPlugin,
)

from tensorflow.keras.models import Model, load_model


class HAT_reaction(ReactionPlugin):

    type_scheme = {
        "hat_reaction": {
            "h_cutoff": float,
            "frequency_factor": float,
            "polling_rate": int,
            "model": str,   #TODO: default None
            "enseble_size": int
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load model

        if self.config.model is None:
        model_dirs = list(res_files("HATmodels").glob("[!_]*"))
        model_dir = model_dirs[0]
        tf_model_dir = list(model_dir.glob("*.tf"))[0]
        self.model: Model = load_model(tf_model_dir)

        # Get hparas
        with open(model_dir / "hparas.json") as f:
            self.hparas = json.load(f)

        if self.hparas.get("scale"):
            with open(model_dir / "scale", "r") as f:
                self.mean, self.std = [float(l.strip()) for l in f.readlines()]
        else:
            self.mean, self.std = [0.0, 1.0]

        self.h_cutoff = self.config.h_cutoff
        self.freqfac = self.config.frequency_factor
        self.polling_rate = self.config.polling_rate

    # def get_reaction_result(self, task_files, latest_files, config, radicals) -> ReactionResult:
    def get_recipe_collection(self, files) -> RecipeCollection:
        tpr = str(files.input["tpr"])
        trr = str(files.input["trr"])
        u = MDA.Universe(str(tpr), str(trr))

        se_dir = files.outputdir / "se"
        # interp_dir = files.outputdir / "interp"

        rad_idxs = getattr(self.runmng, "radical_idxs", [])
        if len(rad_idxs) < 1:
            logging.info("No radicals known, searching in structure..")
            rad_idxs = [str(a[0].index) for a in find_radicals(u)]
        logging.info(f"Found radicals: {rad_idxs}")
        # TODO: handle cases w/o radicals --> empty recipe

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
        ys = self.model.predict(in_ds)
        ys = np.concatenate(ys).astype(dtype=float)
        ys = (ys * self.std) + self.mean

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
            idxs = [idx[0] for idx in idxs] # zero-based
            
            f1 = meta_d["frame"] - self.polling_rate
            if f1 < 0:
                f1 = 0
            f2 = meta_d["frame"]
            t1 = u.trajectory[f1].time
            t2 = u.trajectory[f2].time

            recipes.append(
                Recipe(
                    recipe_steps=[Move(ix_to_move=idxs[0], ix_to_bind=idxs[1])],
                    rates=[rate],
                    timespans=[[t1, t2]],
                )
            )

        recipe_collection = RecipeCollection(recipes)
        recipe_collection.aggregate_reactions()

        return recipe_collection
