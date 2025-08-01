import json
from importlib.resources import files as res_files
import logging

import MDAnalysis as MDA
import numpy as np

from kimmdy_hat.utils.trajectory_utils import (
    extract_subsystems,
)

from kimmdy_hat.utils.utils import find_radicals, free_gpu
from kimmdy.recipe import Bind, Break, Place, Relax, Recipe, RecipeCollection
from kimmdy.plugins import ReactionPlugin

from pprint import pformat
from tempfile import TemporaryDirectory
import shutil
from pathlib import Path
from tqdm.autonotebook import tqdm


class HAT_reaction(ReactionPlugin):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.h_cutoff = self.config.h_cutoff
        self.prediction_scheme = self.config.prediction_scheme
        self.polling_rate = self.config.polling_rate
        self.frequency_factor = self.config.arrhenius_equation.frequency_factor
        self.temperature = self.config.arrhenius_equation.temperature
        self.R = 1.9872159e-3  # [kcal K-1 mol-1]
        self.cap = self.config.cap
        self.change_coords = self.config.change_coords
        self.n_unique = self.config.n_unique
        self.trajectory_format = self.config.trajectory_format

    def get_recipe_collection(self, files) -> RecipeCollection:
        logger = files.logger
        logger.debug("Getting recipe for reaction: HAT")

        ## trajectory parsing   # TODO add gro support
        SOL_RESNAMES = ["SOL", "WAT", "TIP3", "TIP4", "CL", "NA", "K"]
        protein_selection = f"not resname {' '.join(SOL_RESNAMES)}"

        # load topology
        topology_path = files.input[
            "tpr"
        ]  # gro could be used but contains no explicit bonded information and is limited to 100k/1Mio atom indices
        u = MDA.Universe(topology_path.as_posix())

        # load trajectory
        trajectory_path = files.input[self.trajectory_format]
        if trajectory_path is None:
            raise FileNotFoundError(
                f"No trajectory file with format '{self.trajectory_format}' found!"
            )

        if self.trajectory_format == "trr":
            logger.debug("Taking trr trajectory for HAT prediction.")
            system_indices = u.atoms.indices
        elif self.trajectory_format == "xtc":
            logger.debug("Taking xtc trajectory for HAT prediction.")
            for name, mdp in self.runmng.mdps.items():
                if name + ".mdp" in self.runmng.latest_files.keys():
                    if group := mdp.get("compressed-x-grps"):
                        if group.lower() == "protein":
                            system_indices = u.select_atoms(protein_selection).indices
                            u = MDA.Merge(u.select_atoms(protein_selection))
                            logger.debug("Selecting Protein indices")
                        elif group.lower() == "system":
                            system_indices = u.atoms.indices
                            logger.debug("Selecting System indices")
                        else:
                            system_indices = u.select_atoms(protein_selection).indices
                            u = MDA.Merge(u.select_atoms(protein_selection))
                            logger.debug("Unknown group, selecting protein indices")
                        break
                    else:
                        system_indices = u.atoms.indices
                        logger.debug(
                            "compressed-x-grps not defined, selecting system indices"
                        )

        else:
            raise NotImplementedError(
                f"Can't load trajectory with unknown format: {self.trajectory_format}"
            )

        try:
            u.load_new(trajectory_path.as_posix())
        except ValueError:
            if u.trajectory.n_atoms > len(u.atoms):
                raise ValueError(
                    f"More atoms in {self.trajectory_format} file than in "
                    "topology. Check compressed-x-grps is set to the correct "
                    "group in .mdp files."
                )
            elif u.trajectory.n_atoms < len(u.atoms):
                raise ValueError(
                    f"Less atoms in {self.trajectory_format} file than in "
                    "topology. Check compressed-x-grps is set to the correct "
                    "group in .mdp files."
                )

        # add necessary attributes
        if not hasattr(u, "elements"):
            # TODO: Make work for 2 character elements
            elements = [t[0].upper() for t in u.atoms.types]
            u.add_TopologyAttr("elements", elements)
        u.atoms.ids = system_indices + 1
        logger.debug(
            f"Trajectory mda.Universe properties: {u}, {len(u.trajectory)} frames, "
            f"{u.bonds}, elements: {u.atoms.elements[:10]}, "
            f"indices: {u.atoms.indices[:10]}\n{trajectory_path}, {topology_path}"
        )

        se_dir = files.outputdir / "se"
        if not self.config.keep_structures:
            se_dir_bck = se_dir
            se_tmpdir = TemporaryDirectory()
            se_dir = Path(se_tmpdir.name)

        if getattr(self.config, "radicals", None) is not None:
            rad_ids = [int(r) for r in (self.config.radicals).split()]
            logger.debug(f"Radicals read from reaction config: {rad_ids}")
            logger.debug(
                f"Radical atomtypes: {[u.select_atoms(f'id {rad}').names[0] for rad in rad_ids]}"
            )
        else:
            # One-based strings in top
            rad_ids = list(self.runmng.top.radicals.keys())
            logger.debug(f"Radicals obtained from runmanager: {rad_ids}")
        if len(rad_ids) < 1:
            logger.debug("No radicals known, searching in structure..")
            radicals = find_radicals(u)
            for rad in radicals:
                logger.debug(f"{rad}")
            rad_ids = [str(a[0].id) for a in find_radicals(u)]
        logger.info(f"Found {len(rad_ids)} radicals")
        logger.debug(f"Radicals: {rad_ids}")
        if len(rad_ids) < 1:
            logger.info("--> retuning empty recipe collection")
            return RecipeCollection([])

        rad_ids = sorted(rad_ids)

        try:
            extract_subsystems(
                u,
                rad_ids,
                h_cutoff=self.h_cutoff,
                env_cutoff=10,
                start=None,
                stop=None,
                step=self.polling_rate,
                cap=self.cap,
                rad_min_dist=3,
                n_unique=self.n_unique,
                out_dir=se_dir,
                logger=logger,
            )

            kwargs = {
                "se_dir": se_dir,
                "prediction_scheme": self.prediction_scheme,
                "R": self.R,
                "temperature": self.temperature,
                "polling_rate": self.polling_rate,
                "change_coords": self.change_coords,
                "frequency_factor": self.frequency_factor,
                "files": files,
                "config": self.config,
                "runmng": self.runmng,
                "logger": logger,
            }

            recipe_collection = make_predictions(u, **kwargs)

        except Exception as e:
            # backup in case of failure
            if not self.config.keep_structures:
                shutil.copytree(se_dir, se_dir_bck)
            raise e

        if not self.config.keep_structures:
            se_tmpdir.cleanup()

        return recipe_collection


@free_gpu
def make_predictions(
    u: MDA.Universe,
    se_dir,
    prediction_scheme,
    R,
    temperature,
    polling_rate,
    change_coords,
    frequency_factor,
    files,
    config,
    runmng,
    logger: logging.Logger = logging.getLogger(__name__),
):
    from kimmdy_hat.utils.input_generation import create_meta_dataset_predictions

    logging.getLogger("tensorflow").setLevel("CRITICAL")
    import tensorflow as tf

    logging.getLogger("tensorflow").setLevel("CRITICAL")
    load_model = tf.keras.models.load_model

    # Load model
    if getattr(config, "model", None) is None:
        match runmng.config.changer.topology.parameterization:
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
        ens_glob = config.model

    ensemble_dirs = list(res_files("HATmodels").glob(ens_glob + "*"))
    assert (
        len(ensemble_dirs) > 0
    ), f"Model {ens_glob} not found. Please check your config yml."
    assert (
        len(ensemble_dirs) == 1
    ), f"Multiple Models found for {ens_glob}. Please check your config yml."
    ensemble_dir = ensemble_dirs[0]
    logging.info(f"Using HAT model: {ensemble_dir.name}")
    ensemble_size = getattr(config, "enseble_size", None)
    models = []
    means = []
    stds = []
    hparas = {}
    for model_dir in list(ensemble_dir.glob("*"))[slice(ensemble_size)]:
        tf_model_dir = list(model_dir.glob("*.tf"))[0]
        models.append(load_model(tf_model_dir))

        with open(model_dir / "hparas.json") as f:
            hpara = json.load(f)
            hparas.update(hpara)

        if hpara.get("scale"):
            with open(model_dir / "scale", "r") as f:
                mean, std = [float(l.strip()) for l in f.readlines()]
        else:
            mean, std = [0.0, 1.0]
        means.append(mean)
        stds.append(std)

    # Build input features
    se_npzs = list(se_dir.glob("*.npz"))
    in_ds, es, scale_t, meta_ds, metas_masked = create_meta_dataset_predictions(
        meta_files=se_npzs,
        batch_size=hparas["batchsize"],
        mask_energy=False,
        oneway=True,
    )
    assert len(in_ds) > 0, "Empty dataset!"

    # Make predictions
    logger.info("Making predictions.")
    if prediction_scheme == "all_models":
        ys = []
        for model, m, s in zip(models, means, stds):
            y = model.predict(in_ds).reshape(-1)
            ys.append((y * s) + m)
        ys = np.stack(ys)
        ys = np.mean(np.array(ys), 0)
    elif prediction_scheme == "efficient":
        logger.debug("Efficient prediction scheme was chosen.")
        # hyperparameters
        # offset to lowest barrier, 11RT offset means, the rates
        # are less than one millionth of the highest rate
        required_offset = 11 / (R * temperature)
        uncertainty = 3.5  # kcal/mol; expected error to QM of a single model prediction
        # single prediction
        model, m, s = next(zip(models, means, stds))
        ys_single = model.predict(in_ds).reshape(-1)
        # find where to recalculate with full ensemble (low barriers)
        recalculate = ys_single <= (ys_single.min() + required_offset + uncertainty)
        # build reduced dataset
        meta_files_recalculate = [
            s for s, r in zip(list(se_dir.glob("*.npz")), recalculate) if r
        ]
        in_ds_ensemble, _, _, _, _ = create_meta_dataset_predictions(
            meta_files=meta_files_recalculate,
            batch_size=hparas["batchsize"],
            mask_energy=False,
            oneway=True,
        )
        # ensemble prediction
        ys_ensemble = []
        for model, m, s in zip(models, means, stds):
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
        raise ValueError(f"Unknown prediction scheme: {prediction_scheme}")

    # Rate; RT=0.593 kcal/mol
    logger.info("Creating Recipes.")
    rates = list(
        np.multiply(
            frequency_factor,
            np.float_power(np.e, (-ys / (temperature * R))),
        )
    )
    recipes = []
    old_bound_dict = {}
    with open(files.outputdir / "predictions.csv", "x") as f:
        f.write(" ".join(("npz", "barrier", "rate", "\n")))
        for npz, y, r in zip(se_npzs, ys, rates):
            f.write(" ".join((npz.name, str(y), str(r), "\n")))

    trj_time = []
    for ts in tqdm(
        u.trajectory[:], desc="Creating list of simulation time for each frame"
    ):
        trj_time.append(ts.time)  # frame is just index of this list, t in ps

    logger.info(f"Max Rate: {max(rates)}, predicted {len(rates)} rates")
    for meta_d, rate in tqdm(zip(meta_ds, rates), desc="Writing recipes"):
        ids = [str(i) for i in meta_d["indices"][0:2]]  # one-based as ids are written

        f1 = meta_d["frame"]
        f2 = meta_d["frame"] + polling_rate
        if f2 >= len(u.trajectory):
            f2 = len(u.trajectory) - 1
        t1 = trj_time[f1]
        t2 = trj_time[f2]

        # get id of heavy atom bound to HAT hydrogen before reaction
        h_id = int(ids[0]) - 1
        if old_bound := old_bound_dict.get(h_id, None):
            pass
        else:
            old_bound = str(u.atoms[int(ids[0]) - 1].bonded_atoms[0].id)
            old_bound_dict[h_id] = old_bound

        if change_coords == "place":
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
            # HAT plugin ids are kimmdy ixs (zero-based,int)
            seq = [
                Break(atom_id_1=old_bound, atom_id_2=ids[0]),
                Place(id_to_place=ids[0], new_coords=[x, y, z]),
                Bind(atom_id_1=ids[0], atom_id_2=ids[1]),
            ]
        elif change_coords == "lambda":
            seq = [
                Break(atom_id_1=old_bound, atom_id_2=ids[0]),
                Bind(atom_id_1=ids[0], atom_id_2=ids[1]),
                Relax(),
            ]
        else:
            raise ValueError(f"Unknown change_coords parameter {change_coords}")

        # make recipe
        recipes.append(Recipe(recipe_steps=seq, rates=[rate], timespans=[[t1, t2]]))

    return RecipeCollection(recipes)
