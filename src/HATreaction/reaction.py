import json
from importlib.resources import files as res_files
from pathlib import Path
import logging

import MDAnalysis as MDA
import numpy as np

from HATreaction.utils.trajectory_utils import extract_subsystems, save_capped_systems
from HATreaction.utils.input_generation import create_meta_dataset_predictions
from HATreaction.utils.utils import find_radicals
from kimmdy.reaction import ConversionRecipe, ConversionType, Reaction, ReactionResult
from kimmdy.runmanager import RunManager
from MDAnalysis.topology.guessers import guess_bonds
from MDAnalysis.topology.tables import vdwradii
from tensorflow.keras.models import Model, load_model


class HAT_reaction(Reaction):

    type_scheme = {"hat_reaction": {"h_cutoff": float, "freqfac": float}}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load model
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

    # def get_reaction_result(self, task_files, latest_files, config, radicals) -> ReactionResult:
    def get_reaction_result(self, files) -> ReactionResult:
        tpr = str(files.input["tpr"])
        trr = str(files.input["trr"])
        u = MDA.Universe(str(tpr), str(trr))


        rad_idxs = self.runmng.radical_idxs
        if len(rad_idxs) < 1:
            logging.info("No radicals known, searching in structure..")
            rad_idxs = [str(a[0].index) for a in find_radicals(u)]
        logging.info(f"Found radicals: {rad_idxs}")
        # TODO: handle cases w/o radicals --> empty recipe

        sub_atms = u.select_atoms(
            f"not resname SOL NA CL and (around 20 index {' '.join([i for i in rad_idxs])})"
        )
        sub_atms += u.select_atoms(f"index {' '.join([i for i in rad_idxs])}")
        u_sub = MDA.Merge(sub_atms)
        u_sub.load_new(trr, sub=sub_atms.indices)

        # subuni has different indices, translate:
        rad_idxs_sub = list(
            map(
                str, u_sub.select_atoms(f"id {' '.join([i for i in rad_idxs])}").indices
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

        se_dir = files.outputdir / "se"
        interp_dir = files.outputdir / "interp"

        save_capped_systems(  # maybe just keep in ram? optionally?
            extract_subsystems(u_sub, rad_idxs_sub, h_cutoff=self.h_cutoff), se_dir
        )

        in_ds, es, scale_t, meta_ds, metas_masked = create_meta_dataset_predictions(
            meta_files=list(se_dir.glob("*.npz")),
            batch_size=self.hparas["batchsize"],
            mask_energy=False,
        )

        # Run Model
        ys = []
        for x in in_ds:
            ys.extend(self.model(x).numpy())
        ys = np.concatenate(ys).astype(dtype=np.float64)
        ys = (ys * self.std) + self.mean

        results = ReactionResult()
        
        # Rate
        results.rates = list(np.multiply(self.freqfac, np.power(np.e, (-ys / 0.593))))

        for sub_idxs in list(map(lambda d: d["indices"][0:2], meta_ds)):
            idxs = [u_sub.select_atoms(f"index {sub_idx}").ids for sub_idx in sub_idxs]
            assert all(
                [len(i) == 1 for i in idxs]
            ), f"HAT atom index translation error! \n{meta_ds}"
            idxs = [str(idx[0] + 1) for idx in idxs]
            print(idxs)

            results.recipes.append(
                ConversionRecipe(type=[ConversionType.MOVE], atom_idx=[idxs])
            )

        return results
