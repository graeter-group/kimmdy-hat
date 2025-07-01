import logging
import numpy as np
import random
from pathlib import Path
from tqdm.autonotebook import tqdm

import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCReader
from MDAnalysis.analysis.distances import self_distance_array

from kimmdy_hat.utils.capping_utils import (
    save_capped_systems,
    extract_subsystems_capped,
)


version = 0.9

## code currently not used and tested! ##


def make_radicals(
    u: mda.Universe,
    xtc,
    count,
    start=None,
    stop=None,
    step=None,
    unique=True,
    resnames=None,
    res_cutoff=30,
    h_cutoff=3,
    out_dir=None,
    h_index=None,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """Takes non radical trajectory and makes radical
    trajectories from it.

    Parameters
    ----------
    u : mda.Universe
    xtc : Path
    count : int
        How many radicals to generate. Universes will be build one after the other.
    start : Union[int,None], optional
        For slicing the trajectory, by default None
    stop : Union[int,None], optional
        For slicing the trajectory, by default None
    step : Union[int,None], optional
        For slicing the trajectory, by default None
    unique : bool
        If true, only keep one of every set of atoms.
    resnames : list
        List of resnames around which to generate radicals.
        If None, radicals can be at every H position, by default None
    res_cutoff : int
        Distance around given resnames to look for possible radical positions.
        Only relevant if resnames is given.
    h_cutoff : float
        Cutoff radius for hydrogen search around radical, by default 3
    out_dir : Path
        If give, capped systems are saved after each generated radical
    h_index : int
        For debugging, id of H to select, instead of chosing a random one. By default None
    logger:
        logger instance, optional
    """

    all_heavy = u.select_atoms("not element H")
    all_H = u.select_atoms("element H")

    # Make ids unique. ids are persistent in subuniverses, indices not
    u.atoms.ids = u.atoms.indices + 1

    if resnames is None:
        sel_Hs = all_H[random.sample(range(len(all_H)), count)]
    else:
        res_Hs = all_H.select_atoms(f"around {res_cutoff} resname {' '.join(resnames)}")
        sel_Hs = res_Hs[random.sample(range(len(res_Hs)), count)]

    capped_systems = []
    if h_index is not None:
        sel_Hs = [u.select_atoms(f"id {h_index}")[0]]
    for sel_H in sel_Hs:
        logger.debug(f"Selected H to remove: {sel_H}")
        rad = sel_H.bonded_atoms

        # remove one H and reorder atoms
        sub_atoms = rad + (all_H - sel_H) + (all_heavy - rad)

        u_radical = mda.Merge(sub_atoms)
        u_radical.load_new(str(xtc), format=XTCReader, sub=sub_atoms.indices)

        try:
            subs = extract_subsystems_capped(
                u_radical,
                [0],
                start=start,
                stop=stop,
                step=step,
                unique=unique,
                h_cutoff=h_cutoff,
            )
        except Exception as e:
            logger.debug(f"Selected H: {sel_H}")
            raise e

        for sub in subs:
            sub[1]["meta"]["traj_H"] = sel_H.id
        capped_systems.extend(subs)

        if out_dir is not None:
            save_capped_systems(subs, out_dir)
            logger.debug(f"Saved {len(subs)} systems in {out_dir}")

    return capped_systems


def closest(l, K):
    l = np.asarray(l)
    return l[(np.abs(l - K)).argmin()]


def make_radicals_smart(
    u: mda.Universe,
    xtc,
    count,
    start=None,
    stop=None,
    step=None,
    search_step=50,
    window=50,
    unique=True,
    h_cutoff=1.7,
    out_dir=None,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """Takes non radical trajectory and makes radical
    trajectories from it.
    More efficient sampling for small distances, uniform sampling across found distances

    Parameters
    ----------
    u : mda.Universe
    xtc : Path
    count : int
        How many radicals to generate. Universes will be build one after the other.
    start : Union[int,None], optional
        For slicing the trajectory, by default None
    stop : Union[int,None], optional
        For slicing the trajectory, by default None
    step : Union[int,None], optional
        For slicing the trajectory, by default None
    search_step : Union[int,None], optional
        For searching for Hs with small distances
    window : int
        Amount of frames to search before and ahead of found small distance.
    unique : bool
        If true, only keep one of every set of atoms.
    resnames : list
        List of resnames around which to generate radicals.
        If None, radicals can be at every H position, by default None
    res_cutoff : int
        Distance around given resnames to look for possible radical positions.
        Only relevant if resnames is given.
    h_cutoff : float
        Cutoff radius for hydrogen translation, used to preselect Hs, by default 1.7
    out_dir : Path
        If give, capped systems are saved after each generated radical
    logger:
        logger instance, optional
    """
    all_heavy = u.select_atoms("not element H")
    all_Hs = u.select_atoms("element H")

    # Make ids unique. ids are persistent in subuniverses, indices not
    u.atoms.ids = u.atoms.indices + 1

    sub_Hs = []
    for ts in u.trajectory[start:stop:search_step]:
        # define smaller sub-search spaces
        for _ in range(20):
            center_idx = all_Hs[random.sample(range(len(all_Hs)), 1)][0].id
            local_Hs = all_Hs.select_atoms(f"around 15 id {center_idx}")

            d = self_distance_array(local_Hs.positions)
            k = 0
            for i in range(len(local_Hs)):
                for j in range(i + 1, len(local_Hs)):
                    if d[k] < h_cutoff:
                        # if (local_Hs[j], d[k], ts.frame) not in sub_Hs:
                        sub_Hs.append((local_Hs[j], d[k], ts.frame))
                    k += 1
    sub_Hs = np.array(sub_Hs)
    logger.debug("Found small distances:", sub_Hs.shape)
    sub_Hs = sub_Hs[(np.unique(sub_Hs[:, 0], return_index=True))[1]]
    logger.debug("Found unique systems:", sub_Hs.shape)

    # sample uniformly across found distances
    rng = np.random.default_rng()
    targets = rng.uniform(sub_Hs[:, 1].min(), h_cutoff, count)

    # mask for avoiding double sampling and still get to ordered count
    idxs = []
    mask = list(range(sub_Hs[:, 1].shape[0]))
    for t in targets:
        masked_idx = (np.abs(sub_Hs[:, 1] - t))[mask].argmin()
        new_idx = mask[masked_idx]
        idxs.append(new_idx)
        mask.pop(masked_idx)

    sel_Hs = sub_Hs[idxs][:, [0, 2]]
    logger.debug(
        f"Selected {len(idxs)} Hs at distances from {sub_Hs[:, 1].min():.02f} to {sub_Hs[:, 1].max():.02f}"
    )
    # Building of universes
    capped_systems = []
    for sel_H, frame in sel_Hs:
        logger.debug(f"Selected H to remove: {sel_H}")
        rad = sel_H.bonded_atoms

        # remove one H and reorder atoms
        sub_atoms = rad + (all_Hs - sel_H) + (all_heavy - rad)

        u_radical = mda.Merge(sub_atoms)
        u_radical.load_new(str(xtc), format=XTCReader, sub=sub_atoms.indices)

        try:
            subs = extract_subsystems_capped(
                u_radical,
                [0],
                start=frame - window,
                stop=frame + window,
                step=step,
                unique=unique,
                h_cutoff=h_cutoff,
            )
        except Exception as e:
            logger.debug(f"Selected H: {sel_H}")
            raise e
        for sub in subs:
            sub[1]["meta"]["traj_H"] = sel_H.id
        capped_systems.extend(subs)

        if out_dir is not None:
            save_capped_systems(subs, out_dir)
            logger.debug(f"Saved {len(subs)} systems in {out_dir}")

    return capped_systems
