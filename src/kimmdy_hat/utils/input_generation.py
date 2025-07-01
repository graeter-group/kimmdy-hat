import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
import numpy as np
from kgcnn.graph.adj import (
    coordinates_to_distancematrix,
    define_adjacency_from_distance,
)
from kgcnn.graph.adj import sort_edge_indices
from ase.io import read
from pathlib import Path
from tqdm.autonotebook import tqdm
import pandas as pd
import logging
from pickle import UnpicklingError

# KGCNN==2.1.0, tensorflow==2.10.0
version = 0.6  # Also used in kimmdy_hat pluginversion


def _preproc_pdb(pdbs):
    mol1_f, mol2_f = pdbs
    mol1 = read(bytes.decode(mol1_f.numpy()))
    mol2 = read(bytes.decode(mol2_f.numpy()))
    # mol1 = read(mol1_f)   # for interactive debugging
    # mol2 = read(mol2_f)

    atm1 = mol1.get_atomic_numbers()
    pos1 = mol1.positions
    pos2 = mol2.positions

    # mark moving H
    atm1[0] = 2
    # add ghost atom at H target pos
    atms = np.concatenate((np.array([0]), atm1), axis=0)
    pos = np.concatenate((np.array([pos2[0]]), pos1), axis=0)

    dist = coordinates_to_distancematrix(pos)
    adj, edge_indices = define_adjacency_from_distance(
        dist, max_distance=5, max_neighbours=25
    )

    # idx 0 --> target position, special species 0
    # idx 1 --> current H position
    # check for edge between H end and target
    ed_mask_1 = np.nonzero(edge_indices[:, 0] == 0)
    ed_mask_2 = np.nonzero(edge_indices[:, 1] == 1)
    rad_edge_idx = np.intersect1d(ed_mask_1, ed_mask_2, True)
    if len(rad_edge_idx) < 1:
        edge_indices = sort_edge_indices(
            np.concatenate([np.array([[0, 1], [1, 0]]), edge_indices], axis=0)
        )

    radical_node_index = tf.convert_to_tensor(((0, 1),), tf.int64)  # shape=(1,2)
    radical_edge_index = tf.convert_to_tensor(((0,),), tf.int64)  # shape=(1,1)
    edge_dist = tf.convert_to_tensor(((dist[0, 1],),), tf.float32)  # shape=(1,1)

    return (
        atms,
        pos,
        edge_indices,
        radical_node_index,
        radical_edge_index,
        edge_dist,
    )


def _tf_preproc_pdb(pdbs):
    (
        atms,
        pos,
        edge_indices,
        radical_node_index,
        radical_edge_index,
        edge_dist,
    ) = tf.py_function(
        _preproc_pdb,
        inp=[pdbs],
        Tout=(
            tf.float32,  # atms
            tf.float32,  # pos
            tf.int64,  # edge_indices
            tf.int64,  # radical_node_index
            tf.int64,  # radical_edge_index
            tf.float32,  # edge_dist
        ),
    )
    atms.set_shape((None,))
    pos.set_shape((None, 3))
    edge_indices.set_shape((None, 2))
    radical_node_index.set_shape((None, 2))
    radical_edge_index.set_shape((None, 1))
    edge_dist.set_shape((None, 1))

    return atms, pos, edge_indices, radical_node_index, radical_edge_index, edge_dist


def mk_mols_ds(pdb_pairs):
    mols_ds = Dataset.from_tensor_slices(pdb_pairs)
    mols_ds = mols_ds.map(
        _tf_preproc_pdb,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    return mols_ds


def metas_to_ds(
    meta_files,
    max_dist,
    min_dist,
    opt,
    scale=False,
    old_scale=None,
    mask_energy=True,
    oneway=False,
    logger: logging.Logger = logging.getLogger(__name__),
):
    # Load energies
    meta_dicts1 = []
    meta_dicts2 = []
    energies1 = []
    energies2 = []
    pdbs1 = []

    pickle_errors = []
    for meta_f in tqdm(meta_files, "Loading data", mininterval=2):
        try:
            meta = np.load(meta_f, allow_pickle=True)
            meta_d = np.expand_dims(meta[meta.files[0]], axis=0)[0]

            m1 = meta_d.copy()
            m2 = meta_d.copy()
            m1["direction"] = 1
            m2["direction"] = 2
            meta_dicts1.append(m1)
            meta_dicts2.append(m2)
            name = meta_f.stem
            if "#" in name:  # duplicated meta files w/o dup pdbs
                name = name.rstrip("0123456789").rstrip("#")
            assert (
                "#" not in name
            ), f"ERROR duplicating pdb from {name}\nold name: {meta_f.stem}"

            pdbs1.append(
                (
                    str(meta_f.parent / (name + "_1.pdb")),
                    str(meta_f.parent / (name + "_2.pdb")),
                )
            )

            # filter translation
            if max_dist is not None:
                if meta_d["translation"] > max_dist:
                    energies1.append(np.NaN)
                    energies2.append(np.NaN)
                    continue
            if min_dist is not None:
                if meta_d["translation"] < min_dist:
                    energies1.append(np.NaN)
                    energies2.append(np.NaN)
                    continue

            # read energies
            if opt:
                if "e_s_opt" in meta_d and "e_ts_opt" in meta_d:
                    energies1.append(meta_d["e_ts_opt"] - meta_d["e_s_opt"])
                else:
                    energies1.append(np.NaN)
                if "e_e_opt" in meta_d and "e_ts_opt" in meta_d:
                    energies2.append(meta_d["e_ts_opt"] - meta_d["e_e_opt"])
                else:
                    energies2.append(np.NaN)

            else:
                if "e_max" in meta_d and "e_00" in meta_d:
                    energies1.append(meta_d["e_max"] - meta_d["e_00"])
                else:
                    energies1.append(np.NaN)
                if "e_10" in meta_d and "e_00" in meta_d:
                    energies2.append(meta_d["e_max"] - meta_d["e_10"])
                else:
                    energies2.append(np.NaN)
        except UnpicklingError:
            pickle_errors.append(meta_f)
            energies1.append(np.NaN)
            energies2.append(np.NaN)

    if len(pickle_errors) > 0:
        logger.warning(
            "Errors encountered during the loading of energies! "
            f"{len(pickle_errors)} meta files could not be loaded."
        )
        if len(pickle_errors) < 50:
            logger.error("\n".join(map(lambda p: str(p), pickle_errors)))

    if oneway:
        pdbs = np.array(pdbs1, dtype=str)
        energies = np.array(energies1, dtype=float)
        meta_dicts = np.array(meta_dicts1)
        metas_masked = np.array(meta_files)
    else:
        pdbs2 = [(j, i) for i, j in pdbs1]
        pdbs = np.array(pdbs1 + pdbs2, dtype=str)
        energies = np.array(energies1 + energies2, dtype=float)
        meta_dicts = np.array(meta_dicts1 + meta_dicts2)
        metas_masked = np.array(meta_files + meta_files)
    mask = np.logical_not(np.isnan(energies))

    if mask_energy:
        energies = energies[mask]
        pdbs = pdbs[mask]
        meta_dicts = meta_dicts[mask]
        metas_masked = metas_masked[mask]

    print(f"Loaded {len(energies)} systems!")

    mean = 0
    std = 1
    if old_scale is not None:
        mean = old_scale[0]
        std = old_scale[1]
        energies = (energies - mean) / std
        print(f"Old scale used: mean {mean}, std {std}")

    elif scale:
        mean = energies.mean()
        std = energies.std()
        energies = (energies - mean) / std
        print(f"Scaled, mean {mean}, std {std}")
    else:
        print("Not scaled")
    scale_t = (mean, std)

    return energies, pdbs, scale_t, meta_dicts, metas_masked


def add_descriptors_ds(descriptors, pdbs, in_ds):
    """Extends input dataset by auxiliary inputs for the NN.

    Parameters
    ----------
    descriptors : list[str]
        Paths to pickled descriptor dataframes. Keys must be pdb file names.
    pdbs : list[tuple[str]]
        List of pdb pairs. First is start structure.
    in_ds : Dataset
        Dataset containing the standard input for the NN.

    Returns
    -------
    in_ds : Dataset
        Dataset with descriptors added to existing inputs.
    """

    def _swapp_idx(s):
        splits = s.split("_")
        if splits[-1][0] == "1":
            splits[-1] = "2" + splits[-1][1:]
        else:
            splits[-1] = "1" + splits[-1][1:]
        return "_".join(splits)

    if descriptors is not None:
        print(f"Using {len(descriptors)} auxiliary descriptors.")
        keys = [pdb[0].split("/")[-1] for pdb in pdbs]  # order and mask
        pkls = [pd.read_pickle(p) for p in descriptors]
        desc_df: pd.DataFrame = pd.concat(pkls).loc[keys]

        desc_rad_df = desc_df.loc[desc_df.index.map(_swapp_idx)]
        desc_df = desc_df.join(desc_rad_df, on=desc_rad_df.index, rsuffix="_rad")

        # normalize
        desc_df.apply(lambda s: (s - s.mean()) / (s.std() + 1e-7))

        desc_ds = tf.data.Dataset.from_tensor_slices(desc_df).batch(
            1, deterministic=True
        )
        in_ds = Dataset.zip((in_ds, desc_ds))
        in_ds = in_ds.map(lambda a, b: (a[0], a[1], a[2], a[3], a[4], a[5], b))
    return in_ds


def create_meta_dataset(
    meta_files: list[Path],
    val_split=0.1,
    batch_size=64,
    cache=None,
    scale=False,
    old_scale=None,
    max_dist=None,
    min_dist=None,
    opt=False,
    eval=False,
    descriptors=None,
):
    """Creates training and validation datasets from pdb files.

    Parameters
    ----------
    meta_files : list[Path]
        Paths to meta files. For each file two related pdbs are expected in the
        same location with suffixes `_1.pdb` and `_2.pdb`
    mode : str
        Name of the model to produce the dataset for.
        painn: node_input, equiv_input, xyz_input, bond_index_input, node_radical_index,
               edge_radical_index
    val_split : float, optional
        Which fraction to set aside for validation, by default 0.1
    batch_size : int
        by default 64
    cache : pathlib.Path, optional
        File path to create a cache. If given, the user must take care of deleting
        it if something changed, by default None
    scale : bool
        Whether to scale the energies by their std and subtract the mean.
        Only applies if old_scale is None.
    old_scale : tuple[float, float]
        If given, applies this mean and std as transformation to the energies, by default None
    max_dist : float
        Only use data points with translations below this distance in angstrom, by default None
    min_dist : float
        Only use data points with translations above this distance in angstrom, by default None
    opt : bool
        If True, use optimized energies, by default False
    eval : bool
        Turns shuffling off and returns a list of dicts containing the meta data, by default False
    descriptors : str
        Path to pickled descriptors to use as auxilary inputs into the dense NN, by default None

    Returns
    -------
    train_ds : Dataset
    val_ds : Dataset
    (mean, std) : tuple[float, float]
    int
        Number of datapoints used for training
    [meta_d] : list[dict], optional
        If eval is True, meta data is returned
    """

    energies, pdbs, scale_t, meta_dicts, metas_masked = metas_to_ds(
        meta_files,
        max_dist,
        min_dist,
        opt,
        scale,
        old_scale,
    )
    inputs = mk_mols_ds(pdbs)

    # Read descriptors, zip to other inputs and flatten
    inputs = add_descriptors_ds(descriptors, pdbs, inputs)

    energies_ds = Dataset.from_tensor_slices(energies)
    # ds_complete = Dataset.zip((mols_ds, energies_ds))
    ds_complete = Dataset.zip((inputs, energies_ds))
    if not eval:
        ds_complete = ds_complete.shuffle(1000, reshuffle_each_iteration=False)
    val_size = tf.cast(tf.math.ceil(len(ds_complete) * val_split), tf.int64)

    print("data points used:", len(ds_complete))
    print("val_split:", val_split)

    val_ds = ds_complete.take(val_size)
    train_ds = ds_complete.skip(val_size)

    if cache:
        if not Path(cache).parent.exists():
            Path(cache).parent.mkdir()
        val_ds = val_ds.cache(str(cache) + "_v")
        train_ds = train_ds.cache(str(cache) + "_t")

    if not eval:
        train_ds = train_ds.shuffle(batch_size * 20, reshuffle_each_iteration=True)

    val_ds = val_ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size, False))
    train_ds = train_ds.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size, False)
    )
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    if cache:
        print("Initializing:", cache)
        for i in tqdm(train_ds, "Initializing train"):
            i
        for i in tqdm(val_ds, "Initializing val"):
            i

    if eval:
        return train_ds, val_ds, scale_t, len(ds_complete), meta_dicts
    return train_ds, val_ds, scale_t, len(ds_complete)


def create_meta_dataset_predictions(
    meta_files: list[Path],
    batch_size=64,
    scale=None,
    max_dist=None,
    min_dist=None,
    opt=False,
    descriptors=None,
    mask_energy=True,
    oneway=False,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """Analogue to create_meta_dataset, but returns inputs and energies separate.
    Returned dataset is not shuffled.


    Parameters
    ----------
    meta_files : list[Path]
        Paths to meta files. For each file two related pdbs are expected in the
        same location with suffixes `_1.pdb` and `_2.pdb`
    batch_size : int, optional
        by default 64
    scale : tuple[float,float], optional
        If given, applies this mean and std as transformation to the energies,
        by default None
    max_dist : float, optional
        Only use data points with translations below this distance in angstrom,
        by default None
    min_dist : float, optional
         Only use data points with translations above this distance in angstrom,
         by default None
    opt : bool, optional
        If True, use optimized energies, by default False
    descriptors : str
        Path to pickled descriptors to use as auxilary inputs into the dense NN.

    Returns
    -------
    mols_ds : Dataset
        Containing only inputs
    energies : np.ndarray
        Energies, aligned with mols
    scale_t : tuple
        (mean, std)
    meta_dicts : list[dict]
        Meta informations, aligned with mols.
        `direction` key added to name relevant reaction direction
    metas_masked : list[path]
        meta_files input aligned with mols
    """

    energies, pdbs, scale_t, meta_dicts, metas_masked = metas_to_ds(
        meta_files,
        max_dist,
        min_dist,
        opt,
        old_scale=scale,
        mask_energy=mask_energy,
        oneway=oneway,
        logger=logger,
    )
    in_ds = mk_mols_ds(pdbs)

    # Read descriptors, zip to other inputs and flatten
    in_ds = add_descriptors_ds(descriptors, pdbs, in_ds)

    in_ds = Dataset.zip((in_ds,))
    in_ds = in_ds.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size, False)
    ).prefetch(tf.data.AUTOTUNE)
    return in_ds, energies, scale_t, meta_dicts, metas_masked
