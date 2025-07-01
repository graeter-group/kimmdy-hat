import logging
import time
import numpy as np
from tqdm.autonotebook import tqdm
from pathlib import Path
from multiprocessing import Process
from collections import defaultdict
from itertools import combinations
from typing import Optional

import MDAnalysis as mda
import MDAnalysis.core.universe

from kimmdy_hat.utils.utils import check_cylinderclash


version = 0.9


def get_residue(atm):
    """Builds iteratively an atomgroup containing the residue of the given atom.
    Relies on the resid attribute.

    Parameters
    ----------
    atm : MDAnalysis.core.groups.Atom
        Starting atom

    Returns
    -------
    mda.AtomGroup
        Residue
    """
    resid = atm.resid
    atm_env = atm.universe.select_atoms(f"point { str(atm.position).strip('[ ]') } 15")
    to_check = atm_env.select_atoms(f"bonded id {atm.id}")
    checked = atm_env.select_atoms(f"id {atm.id}")
    resid_group = atm_env.select_atoms(f"id {atm.id}")

    while len(to_check) > 0:
        for c_atm in to_check:
            if c_atm.resid == resid:
                resid_group = resid_group + c_atm
                to_check = to_check + atm_env.select_atoms(f"bonded id {c_atm.id}")
            checked = checked + c_atm
            to_check = to_check - checked
    assert (
        len(np.nonzero(resid_group.names == "CA")) == 1
    ), "ERROR: Multiple CA found in one residue!"
    return resid_group


def get_res_union(atms):
    """Builds atomgroupe containing union of all residues of given atoms.
    Avoids unnecessary calls to get_residue

    Parameters
    ----------
    atms : MDAnalysis.core.groups.Atom
        Atoms which residues will be unionized.
    """
    res = atms[0].universe.atoms[[]]
    for atm in atms:
        if atm not in res:
            res = res | get_residue(atm)
    return res


def _scale_and_mutate(u_cap, h_idxs, h_bondlength=1.01):
    """Scale positions and mutate elements to H"""
    for h in [u_cap.atoms[i] for i in h_idxs]:
        h_alpha = h.bonded_atoms[0]
        bond = h.position - h_alpha.position
        h.position = h_alpha.position + ((bond / np.linalg.norm(bond)) * h_bondlength)

        h.type = "H"
        h.name = "H"
        h.element = "H"


def _get_charge(atm):
    aa_charge_dict = {
        "ala": 0,
        "arg": 1,
        "asn": 0,
        "asp": -1,
        "cys": 0,
        "dop": 0,  # ?
        "gln": 0,
        "glu": -1,
        "gly": 0,
        "his": 0,  # ?
        "hyp": 0,
        "ile": 0,
        "leu": 0,
        "lys": 1,
        "met": 0,
        "phe": 0,
        "pro": 0,
        "ser": 0,
        "thr": 0,
        "trp": 0,
        "tyr": 0,
        "val": 0,
        "l4y": 0,  # HLKNL
        "l5y": 0,  # HLKNL
        "lyx": 1,  # PYD; ly2, ly3 cap
        "ly2": 1,  # PYD
        "ly3": 1,  # PYD
        "ace": 0,
        "nme": 0,
        "eth": 0,  # HLKNL cap
        "mey": 0,  # LYX cap
    }

    charge = aa_charge_dict.get(atm.resname.lower())
    if charge is None:
        logging.warning(f"No charge defined for resname {atm.resname}")
    return charge


def cap_aa(atms):
    """Caps aminoacid(s) with amin or amid group.

    Parameters
    ----------
    atms : mda.AtomGroup
        All atoms of the aminoacid(s) to cap.

    Returns
    -------
    mda.AtomGroup
        Capping atoms. Should be used as union with the
        aminoacid residue as capping occures outside this residue.
    List[int]
        Indices of atoms used as cappes in the old universe.
    """

    possible_ends = atms.select_atoms(f"name C or name N")

    # Handle special cases aka. crosslinks
    special_ends = {
        "L5Y": ["NZ"],  # HLKNL
        "L4Y": ["CE"],  # HLKNL
        "LYX": ["C12", "C13"],  # PYD
        "LY2": ["CB"],  # PYD
        "LY3": ["CG"],  # PYD
    }
    for resname, ends in special_ends.items():
        for res in filter(lambda r: r.resname == resname, atms.residues):
            for end in ends:
                possible_ends += res.atoms.select_atoms(f"name {end}")

    assert len(possible_ends) > 0, "ERROR: No possible ends to cap found!"
    env = atms.universe.select_atoms(
        f"point { str(possible_ends[0].position).strip('[ ]') } 20"
    )
    cap_atms = []
    cap_id = []

    for pe in possible_ends:
        if pe.element == "C":
            # build cap from atoms of next AA
            cap_d = {
                "N": env.select_atoms(
                    f"(bonded id {pe.id}) and not resid {pe.resid} and element N"
                )
            }

            # check cap selection
            assert (
                len(cap_d["N"]) <= 1
            ), f"ERROR: Wrong cap atom selection at id {pe.id}, cap: {list(cap_d['N'])}"
            if pe.name not in [i for l in special_ends.values() for i in l]:
                if len(cap_d["N"]) == 0:
                    continue  # chain end reached (on radical?)
                if cap_d["N"].residues in atms.residues:
                    continue  # next aminoacid included, no need to cap
                if cap_d["N"].resnames[0] == "NME":  # include existing cap
                    cap = cap_d["N"][0].residue.atoms
                    u_cap = MDAnalysis.core.universe.Merge(cap)
                    cap_atms.append(u_cap.atoms)
                    [cap_id.append(i) for i in cap.ids]
                    continue
            if cap_d["N"].residues in atms.residues:
                continue  # next aminoacid included, no need to cap

            # Special cases:
            if pe.name in special_ends["LYX"]:
                cap_d["C"] = env.select_atoms(
                    f"(bonded id {pe.id}) and not resid {pe.resid}"
                )
                cap_d["C_H3"] = cap_d["C"][0].bonded_atoms - pe
                cap = sum([cap_d[k] for k in ["C", "C_H3"]])
                h_idxs = (1, 2, 3)
                # make new universe with fixed order
                u_cap = MDAnalysis.core.universe.Merge(cap)
                u_cap.residues[0].resname = "MEY"

            elif pe.name in special_ends["LY2"] + special_ends["LY3"]:
                cap_d["C"] = env.select_atoms(
                    f"(bonded id {pe.id}) and not resid {pe.resid}"
                )
                cap_d["CC2"] = cap_d["C"][0].bonded_atoms - pe
                cc2_id = " ".join([str(i) for i in cap_d["CC2"].ids])
                cc2_res = " ".join([str(i) for i in cap_d["CC2"].resids])
                cap_d["CCC"] = env.select_atoms(
                    f"(bonded id {cc2_id}) and not resid {cc2_res}"
                )
                cap_d["CCCH3"] = env.select_atoms(
                    f"(bonded id {cap_d['CCC'][0].id}) and resid {cap_d['CCC'][0].resid}"
                )

                exclude_s = " ".join(["N", "C", "CA", "CB", "CG", "CD", "OD"])
                cap_d["ring"] = cap_d["C"][0].residue.atoms.select_atoms(
                    f"not name {exclude_s} and not bonded name {exclude_s}"
                )
                ring_id = " ".join([str(i) for i in cap_d["ring"].ids])
                cap_d["ringC"] = (
                    env.select_atoms(
                        f'resid {cap_d["C"][0].resid} and bonded id {ring_id}'
                    )
                    - cap_d["ring"]
                )
                cap_d["ringCH"] = list(
                    filter(lambda a: a.element == "C", cap_d["ringC"][0].bonded_atoms)
                )[0]

                cap = sum(
                    [cap_d[k] for k in ["CCCH3", "ringCH", "CCC", "ring", "ringC"]]
                )
                h_idxs = (0, 1, 2, 3)

                # make new universe with fixed order
                u_cap = MDAnalysis.core.universe.Merge(cap)
                u_cap.residues[0].resname = "LYX"

            else:  # Standard capping
                # Special treatment if single AA is in between:
                # Build Gly linker
                linker = False
                # one apart:
                if abs(cap_d["N"][0].resindex - pe.resindex) == 1:
                    # pe-linker-rad? can be mirrored, rad must be in atms
                    if (cap_d["N"][0].resindex - pe.resindex) + cap_d["N"][
                        0
                    ].resindex in atms.resindices:
                        # backbone must be intact for linking
                        if (
                            len(cap_d["N"][0].residue.atoms.select_atoms("backbone"))
                            == 4
                        ):
                            linker = True
                        else:
                            logging.debug("Broken backbone in capping AA detected!")

                N_alphas = env.select_atoms(
                    f"((bonded id {cap_d['N'][0].id}) and (resid {cap_d['N'][0].resid})) or ((bonded id {cap_d['N'][0].id}) and name H)"
                )

                # broken bond after N:
                if len(N_alphas) != 2:
                    logging.info(
                        f"Capping group contains radical next to N!\n\t"
                        f"pe: {pe}\n\tN_alphas: {list(N_alphas)}"
                    )
                    cap = cap_d["N"] + N_alphas
                    u_cap = MDAnalysis.core.universe.Merge(cap)
                    cap_atms.append(u_cap.atoms)
                    [cap_id.append(i) for i in cap.ids]
                    continue

                # C_a --> CH3,
                # everything w/ more or less than 1 H attached --> H
                if "H" in N_alphas.elements:
                    cap_d["N_C"] = N_alphas[np.nonzero(N_alphas.elements == "C")[0]][0]
                    cap_d["N_H"] = N_alphas[np.nonzero(N_alphas.elements == "H")[0]][0]
                else:  # two C atoms bond to N, only in PRO, HYP
                    cap_d["N_C"] = N_alphas[np.nonzero(N_alphas.names == "CA")[0]][0]
                    cap_d["N_H"] = N_alphas[np.nonzero(N_alphas.names != "CA")[0]][0]

                assert all(
                    [k in cap_d.keys() for k in ["N_C", "N_H"]]
                ), f"ERROR while building capping group on C-term!\n\tAtom:{cap_d['N'][0]}"

                if not linker:
                    cap_d["NC_3H"] = cap_d["N_C"].bonded_atoms - cap_d["N"]
                    # if broken bond after C_a
                    if len(cap_d["NC_3H"]) != 3:
                        logging.info(
                            f"Capping group contains radical next to Calpha!\n\t"
                            f"pe: {pe}\n\tC_a_alphas: {list(cap_d['NC_3H'])}"
                        )
                    cap = sum([cap_d[k] for k in ["N", "N_H", "N_C", "NC_3H"]])
                    # h_idxs usually: (1,3,4,5) for conversion into H
                    h_idxs = [1]
                    h_idxs += list(range(3, 3 + len(cap_d["NC_3H"])))

                    # make new universe with fixed order
                    u_cap = MDAnalysis.core.universe.Merge(cap)
                    u_cap.residues[0].resname = "NME"

                else:  # LINKER:
                    cap_d["bb"] = cap_d["N"][0].residue.atoms.select_atoms("backbone")
                    assert (
                        len(cap_d["bb"]) == 4
                    ), (
                        breakpoint()
                    )  # f"CAPPING ERROR in linker: backbone {cap_d['N'][0]}"
                    cap_d["NC_2H"] = cap_d["N_C"].bonded_atoms - cap_d["bb"]

                    cap = sum([cap_d[k] for k in ["N_H", "NC_2H", "bb"]])
                    h_idxs = (0, 1, 2)

                    # make new universe with fixed order
                    u_cap = MDAnalysis.core.universe.Merge(cap)
                    u_cap.residues[0].resname = "GLY"

            _scale_and_mutate(u_cap, h_idxs)
            cap_atms.append(u_cap.atoms)
            [cap_id.append(i) for i in cap.ids]

        if pe.element == "N":
            # build cap from atoms of next AA
            cap_d = {
                "C": env.select_atoms(
                    f"((bonded id {pe.id}) and not resid {pe.resid}) and not name H"
                )
            }

            # check cap selection
            assert (
                len(cap_d["C"]) <= 1
            ), f"ERROR: Wrong cap atom selection at id {pe.id}, cap: {list(cap_d['C'])}"
            if len(cap_d["C"]) == 0:
                continue  # chain end reached
            if cap_d["C"].residues in atms.residues:
                continue  # next aminoacid included, no need to cap
            if cap_d["C"].resnames[0] == "ACE":  # include existing cap
                cap = cap_d["C"][0].residue.atoms
                u_cap = MDAnalysis.core.universe.Merge(cap)
                cap_atms.append(u_cap.atoms)
                [cap_id.append(i) for i in cap.ids]
                continue

            # skip if linker AA, always build from C term
            if abs(cap_d["C"][0].resindex - pe.resindex) == 1:
                # pe-linker-rad? can be mirrored, rad must be in atms
                if (cap_d["C"][0].resindex - pe.resindex) + cap_d["C"][
                    0
                ].resindex in atms.resindices:
                    # backbone must be intact for linking
                    if len(cap_d["C"][0].residue.atoms.select_atoms("backbone")) == 4:
                        continue

            C_alphas = env.select_atoms(
                f"(bonded id {cap_d['C'][0].id}) and (resid {cap_d['C'][0].resid})"
            )

            # Cap within HLKNL crosslink
            if pe.name == "NZ":
                cap_d["C_H2"] = C_alphas[np.nonzero(C_alphas.elements == "H")]
                cap_d["CC"] = C_alphas[np.nonzero(C_alphas.elements == "C")]
                C_betas = env.select_atoms(
                    f"(bonded id {cap_d['CC'][0].id}) and not (id {cap_d['C'][0].id})"
                )
                cap_d["CC_H2"] = C_betas[np.nonzero(C_betas.elements != "O")]  # H & CG
                cap_d["CCO"] = C_betas[np.nonzero(C_betas.elements == "O")]
                cap_d["CCOH"] = cap_d["CCO"][0].bonded_atoms - cap_d["CC"]

                cap = sum(
                    [
                        cap_d[k]
                        for k in [
                            "C",
                            "C_H2",
                            "CC",
                            "CC_H2",
                            "CCO",
                            "CCOH",
                        ]
                    ]
                )
                h_idxs = (1, 2, 4, 5, 7)

                # make new universe with fixed order
                u_cap = MDAnalysis.core.universe.Merge(cap)
                u_cap.residues[0].resname = "ETH"

            else:  # Standard capping:
                # if broken bond after C:
                if len(C_alphas) != 2:
                    logging.info(
                        f"Capping group contains radical next to C!\n\t"
                        f"pe: {pe}\n\tC_alphas: {list(C_alphas)}"
                    )
                    cap = cap_d["C"] + C_alphas
                    u_cap = MDAnalysis.core.universe.Merge(cap)
                    cap_atms.append(u_cap.atoms)
                    [cap_id.append(i) for i in cap.ids]
                    continue

                cap_d["O"] = filter(lambda a: a.element == "O", C_alphas).__next__()
                cap_d["CC"] = (C_alphas - cap_d["O"])[0]

                cap_d["CC_H3"] = cap_d["CC"].bonded_atoms - cap_d["C"]
                if len(cap_d["CC_H3"]) != 3:
                    logging.info(
                        f"Capping group contains radical next to Calpha!\n\t"
                        f"pe: {pe}\n\tC_a_alphas: {list(cap_d['CC_H3'])}"
                    )
                cap = sum([cap_d[k] for k in ["C", "O", "CC", "CC_H3"]])
                # h_idxs usually: (3,4,5) for conversion into H
                h_idxs = list(range(3, 3 + len(cap_d["CC_H3"])))

                # make new universe with fixed order
                u_cap = MDAnalysis.core.universe.Merge(cap)
                u_cap.residues[0].resname = "ACE"

            _scale_and_mutate(u_cap, h_idxs)
            cap_atms.append(u_cap.atoms)
            [cap_id.append(i) for i in cap.ids]

    if len(cap_atms) == 0:
        cap = mda.Universe.empty(0).atoms
    else:
        cap = MDAnalysis.core.universe.Merge(*cap_atms).atoms

    return cap, cap_id


def cap_single_rad(u, ts, rad, bonded_rad, h_cutoff=3, env_cutoff=15):
    """Builds capped systems around a single radical in a single frame.
    Aminoacids are capped at peptide bonds resulting in amines and amides.
    Subsystems contain the reactive hydrogen at index 0 followed by the
    radical atom.

    Parameters
    ----------
    u : mda.Universe
        Main universe
    ts : mdanalysis.coordinates.base.Timestep
        On which timestep to operate
    rad : mda.AtomGroup
        Radical atom in its own group
    bonded_rad : mda.AtomGroup
        AtomGroup containing all atoms bonded to the radical
    h_cutoff : float, optional
        Cutoff radius for hydrogen search around radical, by default 3
    env_cutoff : float, optional
        Cutoff radius for local env used for better performance, by default 7

    Returns
    -------
    List
        List of capped subsystems
    """
    # selecting in a smaller env is faster than in whole universe
    env = u.atoms.select_atoms(
        f"(point { str(rad.positions).strip('[ ]') } {env_cutoff}) and "
        "(not resname SOL NA CL)"
    )
    # ts2 = mda.transformations.unwrap(env)(ts)
    # env.unwrap()

    end_poss = find_radical_pos(rad[0], bonded_rad)

    hs = []
    for end_pos in end_poss:
        hs.append(
            env.select_atoms(
                f"point { str(end_pos).strip('[ ]') } {h_cutoff} and element H"
            )
        )
    hs = sum(hs) - bonded_rad  # exclude alpha-H

    # hs = (
    #     env.select_atoms(
    #         f"point { str(rad.positions).strip('[ ]') } {h_cutoff} and type H"
    #     )
    #     - bonded_rad
    # )

    clashes = np.empty((len(hs), len(end_poss)), dtype=bool)
    for h_idx, h in enumerate(hs):
        for end_idx, end_pos in enumerate(end_poss):
            clashes[h_idx, end_idx] = check_cylinderclash(
                end_pos, h.position, env.positions, r_min=0.8
            )

    # get whole residues near radical
    rad_alphas = env.select_atoms(f"bonded id {rad[0].id}")
    # rad_betas = sum([env.select_atoms(f'bonded id {alpha.id}') for alpha in rad_alphas]) - rad

    rad_aa = get_res_union(rad_alphas)

    capped_systems = np.zeros((len(hs),), dtype=object)
    min_translations = np.ones((len(hs),)) * 99

    # iterate over defined HAT reactions
    for h_idx, end_idx in zip(*np.nonzero(clashes)):
        end_pos = end_poss[end_idx]
        h = env.select_atoms(f"id {hs[h_idx].id}")

        translation = np.linalg.norm(end_pos - h.positions)
        # only keep reaction w/ smallest translation
        if translation > min_translations[h_idx]:
            continue
        min_translations[h_idx] = translation

        # get whole residues near reacting H
        h_alpha = env.select_atoms(f"bonded id {h[0].id}")[0]
        h_betas = sum(env.select_atoms(f"bonded id {h_alpha.id}")) - h
        # h_gammas = sum(env.select_atoms(f'bonded index {" ".join([str(i) for i in h_betas.ix])}')) - h_alpha

        h_aa = get_res_union(h_betas)

        core = h_aa | rad_aa
        caps, caps_id = cap_aa(core)

        # core can have more residues than just h-res and rad-res, important for charge!
        core = core - h - rad

        # N terminal end capped w/ RNH3+, fix charge:
        charge_correction = 0
        if "OC1" in core.names:  # OC1 and OC2 form COO- end
            charge_correction += -1
        if "H1" in core.names or "H2" in core.names:  # H1 H2 H3 form NH3+ end
            charge_correction += 1

        # can't merge empty atom group
        to_merge = [h, rad]
        if len(caps) > 0:
            to_merge.append(caps)
        to_merge.append(core)

        capped_systems[h_idx] = {
            "start_u": MDAnalysis.core.universe.Merge(*to_merge),
            "end_u": MDAnalysis.core.universe.Merge(*to_merge),
            "meta": {
                "translation": translation,
                "u1_name": rad[0].resname.lower() + "-sim",
                "u2_name": h[0].resname.lower() + "-sim",
                "charge_u1": _get_charge(rad[0]),
                "charge_u2": _get_charge(h[0]),
                "trajectory": u._trajectory.filename,
                "frame": ts.frame,
                "indices": (*h.ids, *rad.ids, *caps_id, *h_aa.ids, *rad_aa.ids),
                "intramol": rad[0].residue == h[0].residue,
                "charge": sum([_get_charge(res.atoms[0]) for res in core.residues])
                + charge_correction,
            },
        }

        # change H position in end universe
        capped_systems[h_idx]["end_u"].atoms[0].position = end_pos

        # hashes based on systems rather than subgroups, subgroubs would collide
        capped_systems[h_idx]["meta"]["hash_u1"] = abs(
            hash(capped_systems[h_idx]["start_u"])
        )
        capped_systems[h_idx]["meta"]["hash_u2"] = abs(
            hash(capped_systems[h_idx]["end_u"])
        )

    return capped_systems[np.nonzero(capped_systems)[0]]


def save_capped_systems(
    systems: list[tuple],
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

    for new_i_hash, vi in systems.items():
        for max_key, system in vi.items():
            sys_d = system[1]  # 0 is translation
            sys_hash = f"{new_i_hash}_{max_key}"

            if (out_dir / f"{sys_hash}.npz").exists():
                if not system[2]:
                    logger.warning(f"{sys_hash}.npz exists, but overwrite is False!")
                    continue

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


# call this in HAT_reaction.get_recipe_collection() for capped systems
def extract_subsystems_capped(
    u: mda.Universe,
    rad_ids: list[str],
    h_cutoff: float = 3,
    env_cutoff: float = 7,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
    rad_min_dist: float = 3,
    n_unique: int = 50,
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
                    assert len(goal_res) == 1
                    rad[0].residue = goal_res[0]
                    bonded_rad.residues = goal_res[0]

    cut_systems = defaultdict(dict)
    translations_d = defaultdict(dict)
    n_cut_systems = 0
    p = None

    for ts in tqdm(u.trajectory[slice(start, stop, step)]):
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

            # need to rewrite this to work on a idx pair basis
            if cap:
                cut_frame = cap_single_rad(u, ts, rad, bonded_rad, h_cutoff, env_cutoff)
            else:
                raise NotImplementedError(
                    "Use trajectory_utils.extract_subsystems() for uncapped systems"
                )

            for i, cut_sys_dict in enumerate(cut_frame):
                # only h and rad index
                new_i_hash = hash(cut_sys_dict["meta"]["indices"][:2])

                # space left in top list:
                if (n_unique < 1) or (len(translations_d[new_i_hash]) < n_unique):
                    safe_slot = len(translations_d[new_i_hash])
                    translations_d[new_i_hash][safe_slot] = cut_sys_dict["meta"][
                        "translation"
                    ]
                    cut_systems[new_i_hash][safe_slot] = (
                        cut_sys_dict["meta"]["translation"],
                        cut_sys_dict,
                        False,  # overwrite
                    )
                    n_cut_systems += 1

                # not closer than top smallest distances:
                elif cut_sys_dict["meta"]["translation"] >= max(
                    translations_d[new_i_hash].values()
                ):
                    logger.debug("Skipping due to translation")
                    logger.debug(
                        f"{cut_sys_dict['meta']['translation']} < "
                        f"{max(translations_d[new_i_hash].values())}"
                    )
                    continue

                # no space left in top list:
                else:
                    # get longest distance
                    dists = list(translations_d[new_i_hash].values())
                    keys = list(translations_d[new_i_hash].keys())
                    max_key = keys[dists.index(max(dists))]

                    # overwrite longest distance
                    translations_d[new_i_hash][max_key] = cut_sys_dict["meta"][
                        "translation"
                    ]
                    cut_systems[new_i_hash][max_key] = (
                        cut_sys_dict["meta"]["translation"],
                        cut_sys_dict,
                        True,  # Overwrite
                    )

        # saving periodically
        # TODO: this doesn't do anything because it only counts the number of new reactions (radical-hydrogen pairs)
        if out_dir is not None and len(cut_systems) > 5000:
            if p is not None:
                p.join()
            p = Process(
                target=save_capped_systems,
                kwargs={
                    "systems": cut_systems,
                    "out_dir": out_dir,
                    "logger": logger,
                },
            )
            p.start()
            # save_capped_systems(cut_systems.values(), out_dir, logger=logger)
            cut_systems = {}

    if out_dir is not None:
        if p is not None:
            p.join()
        save_capped_systems(cut_systems, out_dir, logger=logger)
        cut_systems = {}

    logger.debug(f"Created {n_cut_systems} isolated systems.")
    return list(cut_systems.values())
