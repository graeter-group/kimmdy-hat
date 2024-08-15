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
        load_model = tf.keras.models.load_model

        super().__init__(*args, **kwargs)

        # Load model
        if getattr(self.config, "model", None) is None:
            match self.runmng.config.changer.topology.parameterization:
                case "basic":
                    ens_glob = "classic_models"
                case "grappa":
                    ens_glob = "grappa_models"
                case _:
                    raise RuntimeError(
                        "Unknown config.changer.topology.parametrization: "
                        "{config.changer.topology.parametrization}"
                    )
        else:
            ens_glob = self.config.model

        ensemble_dirs = list(res_files("HATmodels").glob(ens_glob + "*"))
        assert len(ensemble_dirs) > 0, f"Model {ens_glob} not found. Please check your config yml."
        assert len(ensemble_dirs) == 1, f"Multiple Models found for {ens_glob}. Please check your config yml."
        ensemble_dir = ensemble_dirs[0]
        logging.info(f"Using HAT model: {ensemble_dir.name}")
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
        self.prediction_scheme = self.config.prediction_scheme
        self.polling_rate = self.config.polling_rate
        self.frequency_factor = self.config.arrhenius_equation.frequency_factor
        self.temperature = self.config.arrhenius_equation.temperature
        self.R = 1.9872159e-3  # [kcal K-1 mol-1]
        self.unique = self.config.unique

    def get_recipe_collection(self, files) -> RecipeCollection:
        from HATreaction.utils.input_generation import create_meta_dataset_predictions

        logger = files.logger
        logger.debug("Getting recipe for reaction: HAT")

        tpr = str(files.input["tpr"])
        trr = str(files.input["trr"])
        u = MDA.Universe(str(tpr), str(trr))
        # Make ids unique. ids are persistent in subuniverses, indices not
        u.atoms.ids = u.atoms.indices + 1

        se_dir = files.outputdir / "se"
        if not self.config.keep_structures:
            se_dir_bck = se_dir
            se_tmpdir = TemporaryDirectory()
            se_dir = Path(se_tmpdir.name)

        if getattr(self.config, "radicals", None) is not None:
            rad_ids = self.config.radicals
        else:
            # One-based strings in top
            rad_ids = list(self.runmng.top.radicals.keys())
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
            logger.info("Searching trajectory for radical structures.")
            u_sub = MDA.Merge(sub_atms)
            # u_sub.trajectory[0].dimensions = ts.dimensions
            # for ts in tqdm(u.trajectory[:: self.polling_rate]):

            #     # check manually w/ ngl:
            #     if 0:
            #         import nglview as ngl

            #         view = ngl.show_mdanalysis(u_sub, defaultRepresentation=False)
            #         view.representations = [
            #             {"type": "ball+stick", "params": {"sele": ""}},
            #             {
            #                 "type": "spacefill",
            #                 "params": {"sele": "", "radiusScale": 0.7},
            #             },
            #         ]
            #         view._set_selection("@" + ",".join(rad_ids), repr_index=1)
            #         view.center()
            #         view
            subsystems = extract_subsystems(
                u_sub,
                rad_ids,
                h_cutoff=self.h_cutoff,
                env_cutoff=10,
                start=None,
                stop=None,
                step=self.polling_rate,
                cap=False,
                rad_min_dist=3,
                unique=self.unique,
                logger=logger,
            )
            save_capped_systems(subsystems, se_dir)

            # Build input features
            in_ds, es, scale_t, meta_ds, metas_masked = create_meta_dataset_predictions(
                meta_files=list(se_dir.glob("*.npz")),
                batch_size=self.hparas["batchsize"],
                mask_energy=False,
                oneway=True,
            )

            # Make predictions
            logger.info("Making predictions.")
            if self.prediction_scheme == "all_models":
                ys = []
                for model, m, s in zip(self.models, self.means, self.stds):
                    y = model.predict(in_ds).reshape(-1)
                    ys.append((y * s) + m)
                ys = np.stack(ys)
                ys = np.mean(np.array(ys), 0)
            elif self.prediction_scheme == "efficient":
                # hyperparameters
                # offset to lowest barrier, 11RT offset means, the rates
                # are less than one millionth of the highest rate
                required_offset = 11 / (self.R * self.temperature)
                uncertainty = (
                    3.5  # kcal/mol; expected error to QM of a single model prediction
                )
                # single prediction
                model, m, s = next(zip(self.models, self.means, self.stds))
                ys_single = model.predict(in_ds).reshape(-1)
                # find where to recalculate with full ensemble (low barriers)
                recalculate = ys_single <= (
                    ys_single.min() + required_offset + uncertainty
                )
                # build reduced dataset
                meta_files_recalculate = [
                    s for s, r in zip(list(se_dir.glob("*.npz")), recalculate) if r
                ]
                in_ds_ensemble, _, _, _, _ = create_meta_dataset_predictions(
                    meta_files=meta_files_recalculate,
                    batch_size=self.hparas["batchsize"],
                    mask_energy=False,
                    oneway=True,
                )
                # ensemble prediction
                ys_ensemble = []
                for model, m, s in zip(self.models, self.means, self.stds):
                    y_ensemble = model.predict(in_ds_ensemble).reshape(-1)
                    ys_ensemble.append((y_ensemble * s) + m)
                ys_ensemble = np.stack(ys_ensemble)
                ys_ensemble = np.mean(np.array(ys_ensemble), 0)
                ys_full_iter = iter(ys_ensemble)
                # take ensemble prediction value where there was a recaulcation,
                # else y_single
                ys = np.asarray(
                    [
                        y_single if not r else next(ys_full_iter)
                        for y_single, r in zip(ys_single, recalculate)
                    ]
                )
            else:
                raise ValueError()

            # Rate; RT=0.593 kcal/mol
            logger.info("Creating Recipes.")
            rates = list(
                np.multiply(
                    self.frequency_factor,
                    np.float_power(np.e, (-ys / (self.temperature * self.R))),
                )
            )
            recipes = []
            logger.debug(f"Barriers:\n{pformat(ys)}")
            logger.info(f"Max Rate: {max(rates)}, predicted {len(rates)} rates")
            logger.debug(f"Rates:\n{pformat(rates)}")
            for meta_d, rate in zip(meta_ds, rates):
                ids = [
                    str(i) for i in meta_d["indices"][0:2]
                ]  # one-based as ids are written

                f1 = meta_d["frame"]
                f2 = meta_d["frame"] + self.polling_rate
                if f2 >= len(u.trajectory):
                    f2 = len(u.trajectory) - 1
                t1 = u.trajectory[f1].time
                t2 = u.trajectory[f2].time
                old_bound = str(u_sub.select_atoms(f"bonded id {ids[0]}")[0].id)

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
                        Break(atom_id_1=old_bound, atom_id_2=ids[0]),
                        Place(id_to_place=ids[0], new_coords=[x, y, z]),
                        Bind(atom_id_1=ids[0], atom_id_2=ids[1]),
                    ]
                elif self.config.change_coords == "lambda":
                    seq = [
                        Break(atom_id_1=old_bound, atom_id_2=ids[0]),
                        Bind(atom_id_1=ids[0], atom_id_2=ids[1]),
                        Relax(),
                    ]
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
