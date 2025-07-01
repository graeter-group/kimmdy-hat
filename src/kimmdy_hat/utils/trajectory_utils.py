from itertools import combinations
import logging
from tqdm.autonotebook import tqdm
from multiprocessing import Process
from collections import defaultdict
import MDAnalysis as mda
import MDAnalysis.coordinates.timestep

import numpy as np
import time
from scipy.spatial.transform import Rotation

from kimmdy_hat.utils.utils import check_cylinderclash
from typing import Optional
import numpy.typing as npt
from pathlib import Path
from tqdm.autonotebook import tqdm

version = 0.9


def find_radical_pos(
    center: mda.core.groups.Atom,
    bonded: mda.core.groups.AtomGroup,
):
    """Calculates possible radical positions of a given radical atom

    Parameters
    ----------
    center : mda.core.groups.Atom
        Radical atom
    bonded : mda.core.groups.AtomGroup
        Atom group of bonded atoms. From its length the geometry is predicted.

    Returns
    -------
    list
        List of radical positions, three dimensional arrays
    """
    scale_C = 1.10
    scale_N = 1.04
    scale_O = 0.97
    scale_S = 1.41

    if len(bonded) in [2, 3]:
        assert center.element in [
            "C",
            "N",
        ], f"Element {center.element} does not match bond number"

        if center.element == "N":
            scale = scale_N
        elif center.element == "C":
            scale = scale_C

        b_normed = []
        for b in bonded:
            b_vec = b.position - center.position
            b_vec_norm = b_vec / np.linalg.norm(b_vec)
            b_normed.append(b_vec_norm)

        midpoint = sum(b_normed)

        if len(bonded) == 3 and np.linalg.norm(midpoint) < 0.6:
            # flat structure -> two end positions:
            # midpoint: 109.5 -> ~1, 120 -> 0
            ab = bonded[1].position - bonded[0].position
            ac = bonded[2].position - bonded[0].position
            normal = np.cross(ab, ac)
            normal = normal / np.linalg.norm(normal)

            rad1 = center.position + (normal * scale)
            rad2 = center.position + (normal * (-1) * scale)
            rads = [rad1, rad2]

        else:
            # two bonds, or three in tetraeder:
            # -> mirror mean bond
            v = midpoint / np.linalg.norm(midpoint)
            rads = [center.position + (-1 * v * scale)]

        return rads

    # Radicals w/ only one bond:
    elif len(bonded) == 1:
        # suggest positions in a 109.5 degree cone
        assert center.element in [
            "N",
            "O",
            "S",
        ], f"Element {center.element} type does not match bond number"
        if center.element == "O":
            scale = scale_O
        elif center.element == "S":
            scale = scale_S
        elif center.element == "N":
            scale = scale_N

        b = bonded[0]
        b_vec = b.position - center.position
        b_vec = b_vec / np.linalg.norm(b_vec)
        rnd_vec = [1, 1, 1]  # to find a vector perpendicular to b_vec

        rnd_rot_ax = np.cross(b_vec, rnd_vec)
        rnd_rot_ax = rnd_rot_ax / np.linalg.norm(rnd_rot_ax)

        r1 = Rotation.from_rotvec(1.911 * rnd_rot_ax)  # 109.5 degree (as in EtOH)
        r2 = Rotation.from_rotvec(0.785 * b_vec)  # 45 degree

        ends = [r1.apply(b_vec)]  # up to 109.5

        for i in range(8):
            ends.append(r2.apply(ends[-1]))  # turn in 45d steps

        # norm and vec --> position
        ends = [(e / np.linalg.norm(e)) * scale + center.position for e in ends]

        return ends

    else:
        raise ValueError(f"Weired count of bonds: {list(bonded)}\n\tCorrect radicals?")


def extract_single_rad(
    u: mda.Universe,
    ts: MDAnalysis.coordinates.timestep,
    rad: mda.AtomGroup,
    bonded_rad: mda.AtomGroup,
    h_cutoff: float = 3,
    env_cutoff: float = 10,
) -> npt.NDArray:
    """Produces one cutout for each possible reaction around one given radical.

    Parameters
    ----------
    u
        Universe around the radical
    ts
        current timestep
    rad
        radical
    bonded_rad
        all atoms bound to the radical
    h_cutoff
        maximum distance a hydrogen can travel, by default 3
    env_cutoff
        size of cutout to make, by default 10

    Returns
    -------
    np.ndarray[dict]
        Array with one dict per reaction. Each dict hold start and end Universe,
        as well as meta data
    """
    env = u.atoms.select_atoms(
        f"point { str(rad.positions).strip('[ ]') } {env_cutoff} and "
        "(not resname SOL NA CL)"
    )
    end_poss = find_radical_pos(rad[0], bonded_rad)
    hs = []
    for end_pos in end_poss:
        hs.append(
            env.select_atoms(
                f"point { str(end_pos).strip('[ ]') } {h_cutoff} and element H"
            )
        )
    hs = sum(hs) - bonded_rad  # exclude alpha-H

    clashes = np.empty((len(hs), len(end_poss)), dtype=bool)
    for h_idx, h in enumerate(hs):
        for end_idx, end_pos in enumerate(end_poss):
            clashes[h_idx, end_idx] = check_cylinderclash(
                end_pos, h.position, env.positions, r_min=0.8
            )

    cut_systems = np.zeros((len(hs),), dtype=object)
    min_translations = np.ones((len(hs),)) * 99

    # iterate over defined HAT reactions
    for h_idx, end_idx in zip(*np.nonzero(clashes)):
        end_pos = end_poss[end_idx]
        h = env.select_atoms(f"id {hs[h_idx].id}")

        translation = np.linalg.norm(end_pos - h.positions)
        # only keep reaction w/ smallest translation
        # there can be multiple end positions for one rad!
        if translation > min_translations[h_idx]:
            continue
        min_translations[h_idx] = translation

        other_atms = env - h - rad

        cut_systems[h_idx] = {
            "start_u": mda.core.universe.Merge(h, rad, other_atms),
            "end_u": mda.core.universe.Merge(h, rad, other_atms),
            "meta": {
                "translation": translation,
                "u1_name": rad[0].resname.lower() + "-sim",
                "u2_name": h[0].resname.lower() + "-sim",
                "trajectory": u._trajectory.filename,
                "frame": ts.frame,
                "indices": (*h.ids, *rad.ids, *other_atms.ids),
                "intramol": rad[0].residue == h[0].residue,
            },
        }

        # change H position in end universe
        cut_systems[h_idx]["end_u"].atoms[0].position = end_pos

        # hashes based on systems rather than subgroups, subgroubs would collide
        cut_systems[h_idx]["meta"]["hash_u1"] = abs(hash(cut_systems[h_idx]["start_u"]))
        cut_systems[h_idx]["meta"]["hash_u2"] = abs(hash(cut_systems[h_idx]["end_u"]))

    return cut_systems[np.nonzero(cut_systems)[0]]


def identify_hat_candidates(
    u: mda.Universe,
    frame: int,
    rad: mda.AtomGroup,
    bonded_rad: mda.AtomGroup,
    h_cutoff: float = 3,
    env_cutoff: float = 5,
) -> list[dict]:
    """Produces one cutout for each possible reaction around one given radical.

    Parameters
    ----------
    u
        Universe around the radical
    ts
        current timestep
    rad
        radical
    bonded_rad
        all atoms bound to the radical
    h_cutoff
        maximum distance a hydrogen can travel, by default 3
    env_cutoff
        size of cutout to make, by default 5

    Returns
    -------
    np.ndarray[dict]
        Array with one dict per reaction. Each dict hold start and end Universe,
        as well as meta data
    """
    env = u.select_atoms(
        f"point { str(rad.positions).strip('[ ]') } {env_cutoff} and "
        "(not resname SOL NA CL)"
    )
    end_poss = find_radical_pos(rad[0], bonded_rad)
    hs = []
    for end_pos in end_poss:
        hs.append(
            env.select_atoms(
                f"point { str(end_pos).strip('[ ]') } {h_cutoff} and element H"
            )
        )
    hs = sum(hs) - bonded_rad  # exclude alpha-H

    HAT_candidates = []
    for h_idx, h in enumerate(hs):
        for end_idx, end_pos in enumerate(end_poss):
            if (
                check_cylinderclash(end_pos, h.position, env.positions, r_min=0.8)
                is False
            ):
                # function returns False if there is a clash
                pass
            else:
                translation = np.linalg.norm(end_pos - h.position)
                HAT_candidates.append(
                    {
                        "reaction_ids": (rad.ids[0], h.id),
                        "translation": translation,
                        "frame": frame,
                        "end_pos": end_pos,
                    }
                )
    return HAT_candidates


def extract_by_reaction_ids(
    u: mda.Universe,
    frame: int,
    rad: mda.AtomGroup,
    h: mda.AtomGroup,
    end_pos=np.ndarray,
    env_cutoff: float = 10,
    filename: str = "",
) -> dict:
    """Produces one cutout for each possible reaction around one given radical.

    Parameters
    ----------
    u
        Universe around the radical
    frame
        frame of the trajectory
    rad
        radical heavy atom participating in proposed reaction
    h
        hydrogen atom participating in proposed reaction
    end_pos
        end position of the reactive hydrogen
    env_cutoff
        size of cutout to make, by default 10
    filename
        trajectory filename

    Returns
    -------
    dict
        dict of reaction. Holds start and end Universe,
        as well as meta data
    """
    env = u.select_atoms(
        f"point { str(rad.positions).strip('[ ]') } {env_cutoff} and "
        "(not resname SOL NA CL)"
    )

    translation = np.linalg.norm(end_pos - h.positions)

    other_atms = env - h - rad

    cut_system = {
        "start_u": mda.core.universe.Merge(h, rad, other_atms),
        "end_u": mda.core.universe.Merge(h, rad, other_atms),
        "meta": {
            "translation": translation,
            "u1_name": rad[0].resname.lower() + "-sim",
            "u2_name": h[0].resname.lower() + "-sim",
            "trajectory": filename,
            "frame": frame,
            "indices": (*h.ids, *rad.ids, *other_atms.ids),
            "intramol": rad[0].residue == h[0].residue,
        },
    }

    # change H position in end universe
    cut_system["end_u"].atoms[0].position = end_pos

    # hashes based on systems rather than subgroups, subgroubs would collide
    cut_system["meta"]["hash_u1"] = abs(hash(cut_system["start_u"]))
    cut_system["meta"]["hash_u2"] = abs(hash(cut_system["end_u"]))

    return cut_system


def extract_subsystems(
    u: mda.Universe,
    rad_ids: list[str],
    h_cutoff: float = 3,
    env_cutoff: float = 7,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
    rad_min_dist: float = 3,
    n_unique: int = 100,
    cap: bool = True,
    out_dir: Optional[Path] = None,
    logger: logging.Logger = logging.getLogger(__name__),
) -> list:
    """Builds subsystems out of a trajectory for evaluation of HAT reaction
    either by DFT or a ML model.
    Aminoacids are optionally capped at peptide
    bonds resulting in amines and amides.
    Subsystems contain the reactive hydrogen at index 0 followed by the
    radical atom.
    Note: This adaptes the residue names in given universe to the break

    Parameters
    ----------
    u
        Main universe
    rad_ids
        Indices of the two radical atoms
    h_cutoff
        Cutoff radius for hydrogen search around radical, by default 3
    env_cutoff
        Cutoff radius for local env, by default 7
    start
        For slicing the trajectory, by default None
    stop
        For slicing the trajectory, by default None
    step
        For slicing the trajectory, by default None
    n_unique
        Number of smallest systems per reaction to save. Set to 0 to consider all.
    cap
        Whether or not the subsystems should be capped. If false, subsystems are
        created by cutting out a sphere with radius env_cutoff. Optional, default: True
    out_dir
        Where to save the output structures. If None, structrues are not saved,
        just returned. Saving is performed iteratively and in parallel.
    logger
        logger instance, optional

    Returns
    -------
    list
        List of capped subsystems
    """

    assert len(rad_ids) > 0, "Error: At least one radical must be given!"
    if out_dir:
        logger.debug("Saving structures is turned on.")
    if n_unique > 0:
        logger.debug(f"Saving the {n_unique} smallest distance(s) of each reaction.")
    else:
        logger.debug("Saving all structures of all reactions.")

    if cap:
        raise NotImplementedError(
            "Use capping_utils.extract_subsystems() for capped systems"
        )

    rad_ids = [int(rad) for rad in rad_ids]
    rads: list[mda.AtomGroup] = [u.select_atoms(f"id {rad}") for rad in rad_ids]

    # Delete bonds between radicals
    if len(rad_ids) > 1:
        combs = combinations(rad_ids, 2)
        for c in combs:
            try:
                u.delete_bonds([c])
            except ValueError:
                continue

    bonded_all = [rad[0].bonded_atoms for rad in rads]
    # remove rads
    bonded_all: list[mda.AtomGroup] = [b - sum(rads) for b in bonded_all]

    # correct residue of radicals to avoid residues w/ only 2 atoms
    # Necessary in case of backbone break other than peptide bond
    for rad, bonded_rad in zip(rads, bonded_all):
        if len(bonded_rad.residues) == 1:
            continue

        res_rad_org = rad[0].residue
        for bonded in bonded_rad:
            if bonded.residue == res_rad_org:
                # bonded to nothing else than the radical:
                if (bonded.bonded_atoms - rad).n_atoms == 0:
                    goal_res = bonded_rad.residues - rad[0].residue
                    if len(goal_res) != 1:
                        logger.warning(
                            f"Unexpected number of different residues at radical {rad[0]}:{len(bonded_rad.residues)}. Could be caused by previous HAT reactions"
                        )
                    # assert len(goal_res) == 1
                    rad[0].residue = goal_res[0]
                    bonded_rad.residues = goal_res[0]

    p = None

    hat_candidates = []
    for ts in tqdm(
        u.trajectory[slice(start, stop, step)], desc="Searching for HAT candidates"
    ):
        frame = ts.frame
        for i, (rad, bonded_rad) in enumerate(zip(rads, bonded_all)):
            # skip radical if small distance to another radical
            skip = False
            for j, other_rad in enumerate(rads):
                if i == j:
                    continue
                if (
                    np.linalg.norm(rad.positions[0] - other_rad.positions[0])
                    < rad_min_dist
                ):
                    logger.debug(
                        f"Radical {rad} distance too small to {other_rad} in frame {ts.frame}, skipping.."
                    )
                    skip = True
            if skip:
                continue

            hat_candidates.extend(
                identify_hat_candidates(
                    u, frame, rad, bonded_rad, h_cutoff, env_cutoff=h_cutoff + 2
                )
            )
    logger.debug(f"Found {len(hat_candidates)} HAT candidates.")

    if n_unique < 1:
        prediction_targets = hat_candidates
    else:
        # find hat candidates with lowest n_unique translations
        candidates_per_reaction_ids = defaultdict(list)
        for entry in hat_candidates:
            candidates_per_reaction_ids[entry["reaction_ids"]].append(entry)
        logger.debug(
            f"{len(candidates_per_reaction_ids.keys())} unique reactions in HAT candidates."
        )

        prediction_targets = []
        for reaction_ids, entries in candidates_per_reaction_ids.items():
            # Sort by 'translation' and select the first 100 (or fewer if there are less than 100)
            prediction_targets.extend(
                sorted(entries, key=lambda x: x["translation"])[:n_unique]
            )
    logger.debug(f"Selected {len(prediction_targets)} prediction targets.")

    # get radical and h ags now once instead of selecting them repeatedly
    rad_ags = {rad: u.select_atoms(f"id {rad}") for rad in rad_ids}
    h_ags = {}
    predictions_per_frame = defaultdict(list)
    for entry in prediction_targets:
        predictions_per_frame[entry["frame"]].append(entry)
        if h_ags.get(entry["reaction_ids"][1], None) is None:
            h_id = entry["reaction_ids"][1]
            h_ags[h_id] = u.select_atoms(f"id {h_id}")

    logger.info(f"Starting to write out {len(prediction_targets)} prediction targets.")
    # create cut systems
    filename = u._trajectory.filename
    cut_systems = []
    for frame, entries in tqdm(
        predictions_per_frame.items(), desc="Writing HAT structures"
    ):
        u.trajectory[frame]  # set frame of trajectory
        for entry in entries:
            cut_sys_dict = extract_by_reaction_ids(
                u,
                frame,
                rad=rad_ags[entry["reaction_ids"][0]],
                h=h_ags[entry["reaction_ids"][1]],
                end_pos=entry["end_pos"],
                env_cutoff=env_cutoff,
                filename=filename,
            )
            cut_systems.append(cut_sys_dict)

        # saving periodically
        if out_dir is not None and len(cut_systems) > 2000:
            if p is not None:
                p.join()
            p = Process(
                target=save_systems,
                kwargs={
                    "systems": cut_systems,
                    "out_dir": out_dir,
                    "logger": logger,
                },
            )
            p.start()
            cut_systems = []
    # final saving
    if out_dir is not None:
        if p is not None:
            p.join()
        p = Process(
            target=save_systems,
            kwargs={
                "systems": cut_systems,
                "out_dir": out_dir,
                "logger": logger,
            },
        )
        p.start()
        p.join()
        cut_systems = []
    return []


def save_systems(
    systems: list[dict],
    out_dir,
    frame: int = None,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """Saves output from `extract_subsystems`

    Parameters
    ----------
    systems
        Systems to save the structures and meta file for.
        First dimension is translation distance,
        second the dict containing the system,
        third bool whether this should be overwritten if it exists already.
    out_dir : Path
        Where to save. Should probably be traj/batch_238/se
    frame
        Overwrite the frame for all given systems
    logger
        logger instance, optional
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    logger.debug("Start saving..")

    for sys_d in tqdm(systems, desc="Writing systems"):
        sys_hash = hash(
            (*sys_d["meta"]["indices"][:2], sys_d["meta"]["translation"])
        )  # assuming reaction ids + translation to be unique

        sys_d["start_u"].atoms.write(out_dir / f"{sys_hash}_1.pdb")
        sys_d["end_u"].atoms.write(out_dir / f"{sys_hash}_2.pdb")

        sys_d["meta"]["meta_path"] = out_dir / f"{sys_hash}.npz"

        if frame is not None:
            sys_d["meta"]["frame"] = frame

        for i in range(10):
            try:
                np.savez(out_dir / f"{sys_hash}.npz", sys_d["meta"])
                break
            except OSError as e:
                logger.exception(e)
                logger.warning(f"{i+1}th retry to save {f'{sys_hash}.npz'}")
                time.sleep(1)
        else:
            raise OSError("Input/output error")
    logger.info(f"Saved {len(systems)} systems.")
