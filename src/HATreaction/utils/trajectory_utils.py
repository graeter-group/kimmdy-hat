from itertools import combinations
import logging
from typing import Union
from MDAnalysis.coordinates.XTC import XTCReader
from MDAnalysis.analysis.distances import self_distance_array

import MDAnalysis as MDA
import numpy as np
import random
from scipy.spatial.transform import Rotation
from tqdm.autonotebook import tqdm

from barrierdata.utils.lmbtr_utils import compress_lmbtr
from barrierdata.utils.structure_creation_utils import mda_to_ase, check_cylinderclash
import logging

version = 0.2


def find_radical_pos(
    center: MDA.core.groups.Atom, bonded: MDA.core.groups.AtomGroup, tetrahedral=False
):
    """Calculates possible radical positions of a given radical atom

    Parameters
    ----------
    center : MDA.core.groups.Atom
        Radical atom
    bonded : MDA.core.groups.AtomGroup
        Atom group of bonded atoms. From its length the geometry is predicted.
    tetrahedral : bool
        Whether to assume a tetrahedral conformation around C and N

    Returns
    -------
    list
        List of radical positions, three dimensional arrays
    """
    scale_C = 1.10
    scale_N = 1.04
    scale_O = 0.97
    scale_S = 1.41

    if center.element == "C" and len(bonded) == 2:
        c_alphas = center.bonded_atoms
        c_o = c_alphas[np.nonzero(c_alphas.elements == "O")]
        assert len(c_o) in [0, 1], "Carboxyl radical?"
        if len(c_o) > 0 and len(c_o[0].bonded_atoms) > 1:
            tetrahedral = True

    if tetrahedral:
        # MAKE SP3 from C with 2 bonds
        # invert bonded positions
        inv_1 = (center.position - bonded[0].position) + center.position
        inv_2 = (center.position - bonded[1].position) + center.position
        # construct rotation axis
        midpoint = (inv_2 - inv_1) * 0.5 + inv_1
        rot_ax = midpoint - center.position
        # 90 degree rotation
        r = Rotation.from_rotvec((np.pi / 2) * (rot_ax / np.linalg.norm(rot_ax)))
        # rotated bonds relative to center
        rad_1 = r.apply(inv_1 - center.position)
        rad_2 = r.apply(inv_2 - center.position)

        # scale to correct bond length, make absolute
        if center.element == "N":
            scale = scale_N
        elif center.element == "C":
            scale = scale_C
        else:
            raise NotImplementedError("H Bondlength to central atom missing")

        rad_1 = (rad_1 / np.linalg.norm(rad_1)) * scale + center.position
        rad_2 = (rad_2 / np.linalg.norm(rad_2)) * scale + center.position
        return [rad_1, rad_2]

    if len(bonded) in [2, 3]:
        assert center.element in [
            "C",
            "N",
        ], f"Element {center.element} does not match bond number"

        # prediction: inverse midpoint between bonded
        # scale to correct bond length, make absolute
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

        v = midpoint / np.linalg.norm(midpoint)
        rad = center.position + (-1 * v * scale)
        return [rad]

    # Radicals w/ only one bond:
    elif len(bonded) == 1:
        assert center.element in ["O", "S"], "Element type does not match bond number"
        if center.element == "O":
            scale = scale_O
        elif center.element == "S":
            scale = scale_S

        b = bonded[0]
        b_vec = b.position - center.position
        b_vec = b_vec / np.linalg.norm(b_vec)
        rnd_vec = [1, 1, 1]  # to find a vector perpendicular to b_vec

        rnd_rot_ax = np.cross(b_vec, rnd_vec)
        rnd_rot_ax = rnd_rot_ax / np.linalg.norm(rnd_rot_ax)

        r1 = Rotation.from_rotvec(1.911 * rnd_rot_ax)  # 109.5 degree
        r2 = Rotation.from_rotvec(0.785 * b_vec)  # 45 degree

        ends = [r1.apply(b_vec)]  # up to 109.5

        for i in range(8):
            ends.append(r2.apply(ends[-1]))  # turn in 45d steps

        # norm and vec --> position
        ends = [(e / np.linalg.norm(e)) * scale + center.position for e in ends]

        return ends

    else:
        raise ValueError(f"Weired count of bonds: {list(bonded)}\nCorrect radicals?")


def idx_to_radicals(u, idx_1, idx_2):
    rad_1 = u.select_atoms(f"index {idx_1}")[0]
    rad_2 = u.select_atoms(f"index {idx_2}")[0]

    bonded_1 = u.select_atoms(f"bonded index {idx_1}")
    bonded_2 = u.select_atoms(f"bonded index {idx_2}")

    rad_positions = []
    rad_positions.extend(find_radical_pos(rad_1, bonded_1 - rad_2))
    rad_positions.extend(find_radical_pos(rad_2, bonded_2 - rad_1))

    return rad_positions


def calc_lmbtr(
    u: MDA.Universe,
    rad_idxs: Union[int, list],
    lmbtr,
    step=1,
    flash=500,
    env_cutoff=7,
    h_cutoff=4,
):

    if isinstance(rad_idxs, int):
        rad_idxs = [rad_idxs]
    rads = []
    for rad_idx in rad_idxs:
        rads.append(u.select_atoms(f"index {rad_idx}")[0])
    bonded = []
    for i, rad_idx in enumerate(rad_idxs):
        bonded.append(u.select_atoms(f"bonded index {rad_idx}"))
        # removes bonds between radicals
        for rad in rads[:i] + rads[i + 1 :]:
            bonded[-1] -= rad

    ase_envs = []
    ase_envs_all = []
    lmbtr_positions = []
    lmbtr_positions_all = []
    lmbtr_results = []
    for _ in tqdm(u.trajectory[::step]):
        for rad, bonded_atoms in zip(rads, bonded):
            rad_pos_list = find_radical_pos(rad, bonded_atoms)

            for rad_pos in rad_pos_list:
                env = u.select_atoms(
                    f"point { str(rad_pos).strip('[ ]') } {env_cutoff}"
                )

                hs = env.select_atoms(
                    f"point { str(rad_pos).strip('[]') } {h_cutoff} and element H"
                )

                for h in hs:
                    lmbtr_positions.append(np.expand_dims(rad_pos, 0))
                    ase_envs.append(mda_to_ase(env - h))
                    ase_h = mda_to_ase(h)
                    ase_h.symbols = "Y"
                    ase_envs[-1] += ase_h

                if len(ase_envs) >= flash:
                    lmbtr_results.extend(
                        lmbtr.create(ase_envs, lmbtr_positions, n_jobs=8)
                    )
                    ase_envs_all.extend(ase_envs)
                    ase_envs = []
                    lmbtr_positions_all.extend(lmbtr_positions)
                    lmbtr_positions = []

    if len(ase_envs) > 0:
        lmbtr_results.extend(lmbtr.create(ase_envs, lmbtr_positions, n_jobs=1))
        ase_envs_all.extend(ase_envs)
        lmbtr_positions_all.extend(lmbtr_positions)

    lmbtr_results = compress_lmbtr(lmbtr, lmbtr_results, flatten=True, k3_center="Y")
    print(f"lmbtr calculated on {len(lmbtr_results)} positions")
    return lmbtr_results, ase_envs_all, lmbtr_positions_all


def get_residue(atm):
    """Builds iteratively an atomgroup containing the residue of the given atom.
    Relies on the resid attribute.

    Parameters
    ----------
    atm : MDA.core.groups.Atom
        Starting atom

    Returns
    -------
    MDA.AtomGroup
        Residue
    """
    resid = atm.resid
    atm_env = atm.universe.select_atoms(f"point { str(atm.position).strip('[ ]') } 15")
    to_check = atm_env.select_atoms(f"bonded index {atm.ix}")
    checked = atm_env.select_atoms(f"index {atm.ix}")
    resid_group = atm_env.select_atoms(f"index {atm.ix}")

    while len(to_check) > 0:
        for c_atm in to_check:
            if c_atm.resid == resid:
                resid_group = resid_group + c_atm
                to_check = to_check + atm_env.select_atoms(f"bonded index {c_atm.ix}")
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
    atms : MDA.core.groups.Atom
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


def cap_aa(atms):
    """Caps aminoacid(s) with amin or amid group.

    Parameters
    ----------
    atms : MDA.AtomGroup
        All atoms of the aminoacid(s) to cap.

    Returns
    -------
    MDA.AtomGroup
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
    cap_ix = []

    for pe in possible_ends:
        if pe.element == "C":
            # build cap from atoms of next AA
            cap_d = {
                "N": env.select_atoms(
                    f"(bonded index {pe.ix}) and not resid {pe.resid} and element N"
                )
            }

            # check cap selection
            assert (
                len(cap_d["N"]) <= 1
            ), f"ERROR: Wrong cap atom selection at index {pe.ix}, cap: {list(cap_d['N'])}"
            if pe.name not in [i for l in special_ends.values() for i in l]:
                if len(cap_d["N"]) == 0:
                    continue  # chain end reached (on radical?)
                if cap_d["N"].residues in atms.residues:
                    continue  # next aminoacid included, no need to cap
                if cap_d["N"].resnames[0] == "NME":  # include existing cap
                    cap = cap_d["N"][0].residue.atoms
                    u_cap = MDA.core.universe.Merge(cap)
                    cap_atms.append(u_cap.atoms)
                    [cap_ix.append(i) for i in cap.ix]
                    continue
            if cap_d["N"].residues in atms.residues:
                continue  # next aminoacid included, no need to cap

            # Special cases:
            if pe.name in special_ends["LYX"]:
                cap_d["C"] = env.select_atoms(
                    f"(bonded index {pe.ix}) and not resid {pe.resid}"
                )
                cap_d["C_H3"] = cap_d["C"][0].bonded_atoms - pe
                cap = sum([cap_d[k] for k in ["C", "C_H3"]])
                h_idxs = (1, 2, 3)
                # make new universe with fixed order
                u_cap = MDA.core.universe.Merge(cap)
                u_cap.residues[0].resname = "MEY"

            elif pe.name in special_ends["LY2"] + special_ends["LY3"]:
                cap_d["C"] = env.select_atoms(
                    f"(bonded index {pe.ix}) and not resid {pe.resid}"
                )
                cap_d["CC2"] = cap_d["C"][0].bonded_atoms - pe
                cc2_ix = " ".join([str(i) for i in cap_d["CC2"].ix])
                cc2_res = " ".join([str(i) for i in cap_d["CC2"].resids])
                cap_d["CCC"] = env.select_atoms(
                    f"(bonded index {cc2_ix}) and not resid {cc2_res}"
                )
                cap_d["CCCH3"] = env.select_atoms(
                    f"(bonded index {cap_d['CCC'][0].ix}) and resid {cap_d['CCC'][0].resid}"
                )

                exclude_s = " ".join(["N", "C", "CA", "CB", "CG", "CD", "OD"])
                cap_d["ring"] = cap_d["C"][0].residue.atoms.select_atoms(
                    f"not name {exclude_s} and not bonded name {exclude_s}"
                )
                ring_ix = " ".join([str(i) for i in cap_d["ring"].ix])
                cap_d["ringC"] = (
                    env.select_atoms(
                        f'resid {cap_d["C"][0].resid} and bonded index {ring_ix}'
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
                u_cap = MDA.core.universe.Merge(cap)
                u_cap.residues[0].resname = "LYX"

            else:  # Standard capping
                # Special treatment if single AA is in between:
                # Build Gly linker
                linker = False
                if (cap_d["N"][0].resindex - pe.resindex) + cap_d["N"][
                    0
                ].resindex in atms.resindices:
                    if abs(cap_d["N"][0].resindex - pe.resindex) == 1:
                        # should be fine to convert crosslinks?!
                        # if cap_d["N"][0].resname not in ["L4Y", "L5Y"]:
                        linker = True

                N_alphas = env.select_atoms(
                    f"(bonded index {cap_d['N'][0].ix}) and (resid {cap_d['N'][0].resid})",f"(bonded index {cap_d['N'][0].ix}) and name H"
                )
                logging.info("Checking NH cap problem:")
                logging.info(N_alphas)
                ref = env.select_atoms(f"(bonded index {cap_d['N'][0].ix})")
                logging.info(ref)

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
                ), f"ERROR while building capping group on C-term!\nAtom:{cap_d['N'][0]}"

                if not linker:
                    cap_d["NC_3H"] = cap_d["N_C"].bonded_atoms - cap_d["N"]
                    assert (
                        len(cap_d["NC_3H"]) == 3
                    ), f"CAPPING ERROR: Atom {cap_d['N'][0]}"

                    cap = sum([cap_d[k] for k in ["N", "N_H", "N_C", "NC_3H"]])
                    h_idxs = (1, 3, 4, 5)

                    # make new universe with fixed order
                    u_cap = MDA.core.universe.Merge(cap)
                    u_cap.residues[0].resname = "NME"

                else:  # LINKER:
                    cap_d["bb"] = cap_d["N"][0].residue.atoms.select_atoms("backbone")
                    assert (
                        len(cap_d["bb"]) == 4
                    ), f"CAPPING ERROR in linker: backbone {cap_d['N'][0]}"
                    cap_d["NC_2H"] = cap_d["N_C"].bonded_atoms - cap_d["bb"]

                    cap = sum([cap_d[k] for k in ["N_H", "NC_2H", "bb"]])
                    h_idxs = (0, 1, 2)

                    # make new universe with fixed order
                    u_cap = MDA.core.universe.Merge(cap)
                    u_cap.residues[0].resname = "GLY"

            _scale_and_mutate(u_cap, h_idxs)
            cap_atms.append(u_cap.atoms)
            [cap_ix.append(i) for i in cap.ix]

        if pe.element == "N":
            # build cap from atoms of next AA
            cap_d = {
                "C": env.select_atoms(
                    f"(bonded index {pe.ix}) and not resid {pe.resid} and not name H"
                )
            }

            # check cap selection
            assert (
                len(cap_d["C"]) <= 1
            ), f"ERROR: Wrong cap atom selection at index {pe.ix}, cap: {list(cap_d['C'])}"
            if len(cap_d["C"]) == 0:
                continue  # chain end reached
            if cap_d["C"].residues in atms.residues:
                continue  # next aminoacid included, no need to cap
            if cap_d["C"].resnames[0] == "ACE":  # include existing cap
                cap = cap_d["C"][0].residue.atoms
                u_cap = MDA.core.universe.Merge(cap)
                cap_atms.append(u_cap.atoms)
                [cap_ix.append(i) for i in cap.ix]
                continue

            # skip if linker AA, always build from C term
            if (cap_d["C"][0].resindex - pe.resindex) + cap_d["C"][
                0
            ].resindex in atms.resindices:
                if abs(cap_d["C"][0].resindex - pe.resindex) == 1:
                    # should be fine to convert crosslinks?!
                    # if cap_d["C"][0].resname not in ["L4Y", "L5Y"]:
                    continue

            C_alphas = env.select_atoms(
                f"(bonded index {cap_d['C'][0].ix}) and (resid {cap_d['C'][0].resid})"
            )

            # Cap within HLKNL crosslink
            if pe.name == "NZ":
                cap_d["C_H2"] = C_alphas[np.nonzero(C_alphas.elements == "H")]
                cap_d["CC"] = C_alphas[np.nonzero(C_alphas.elements == "C")]
                C_betas = env.select_atoms(
                    f"(bonded index {cap_d['CC'][0].ix}) and not (index {cap_d['C'][0].ix})"
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
                u_cap = MDA.core.universe.Merge(cap)
                u_cap.residues[0].resname = "ETH"

            else:  # Standard capping:
                assert (
                    len(C_alphas) == 2
                ), f"ERROR while building capping group on N-term!\nAtom:{cap_d['C'][0]}"

                cap_d["O"] = filter(lambda a: a.element == "O", C_alphas).__next__()
                cap_d["CC"] = (C_alphas - cap_d["O"])[0]

                cap_d["CC_H3"] = cap_d["CC"].bonded_atoms - cap_d["C"]

                cap = sum([cap_d[k] for k in ["C", "O", "CC", "CC_H3"]])
                h_idxs = (3, 4, 5)

                # make new universe with fixed order
                u_cap = MDA.core.universe.Merge(cap)
                u_cap.residues[0].resname = "ACE"

            _scale_and_mutate(u_cap, h_idxs)
            cap_atms.append(u_cap.atoms)
            [cap_ix.append(i) for i in cap.ix]

    if len(cap_atms) == 0:
        cap = MDA.Universe.empty(0).atoms
    else:
        cap = MDA.core.universe.Merge(*cap_atms).atoms

    return cap, cap_ix


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


def cap_single_rad(u, ts, rad, bonded_rad, h_cutoff=3, env_cutoff=7):
    """Builds capped systems around a single radical in a single frame.
    Aminoacids are capped at peptide bonds resulting in amines and amides.
    Subsystems contain the reactive hydrogen at index 0 followed by the
    radical atom.

    Parameters
    ----------
    u : MDA.Universe
        Main universe
    ts : MDAnalysis.coordinates.base.Timestep
        On which timestep to operate
    rad : MDA.AtomGroup
        Radical atom in its own group
    bonded_rad : MDA.AtomGroup
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
        f"point { str(rad.positions).strip('[ ]') } {env_cutoff}"
    )
    ts2 = MDA.transformations.unwrap(env)(ts)

    end_poss = find_radical_pos(rad[0], bonded_rad)

    hs = []
    for end_pos in end_poss:
        hs.append(
            env.select_atoms(
                f"point { str(end_pos).strip('[ ]') } {h_cutoff} and element H"
            )
        )
    hs = sum(hs) - bonded_rad # reacting H be at radical already

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
    rad_alphas = env.select_atoms(f"bonded index {rad[0].ix}")
    # rad_betas = sum([env.select_atoms(f'bonded index {alpha.ix}') for alpha in rad_alphas]) - rad

    rad_aa = get_res_union(rad_alphas)

    capped_systems = np.zeros((len(hs),), dtype=object)
    min_translations = np.ones((len(hs),)) * 99

    # iterate over defined HAT reactions
    for h_idx, end_idx in zip(*np.nonzero(clashes)):
        end_pos = end_poss[end_idx]
        h = env.select_atoms(f"index {hs[h_idx].ix}")

        translation = np.linalg.norm(end_pos - h.positions)
        # only keep reaction w/ smallest translation
        if translation > min_translations[h_idx]:
            continue
        min_translations[h_idx] = translation

        # get whole residues near reacting H
        h_alpha = env.select_atoms(f"bonded index {h[0].ix}")[0]
        h_betas = sum(env.select_atoms(f"bonded index {h_alpha.ix}")) - h
        # h_gammas = sum(env.select_atoms(f'bonded index {" ".join([str(i) for i in h_betas.ix])}')) - h_alpha

        h_aa = get_res_union(h_betas)

        core = h_aa | rad_aa
        caps, caps_ix = cap_aa(core)

        # core can have more residues than just h-res and rad-res, important for charge!
        core = core - h - rad

        # N terminal end capped w/ RNH3+, fix charge:
        charge_correction = 0
        if "OC1" in core.names:  # OC1 and OC2 form COO- end
            charge_correction += -1
        if "H1" in core.names or "H2" in core.names:  # H1 H2 H3 form NH3+ end
            charge_correction += 1

        capped_systems[h_idx] = {
            "start_u": MDA.core.universe.Merge(
                h,
                rad,
                caps,
                core,
            ),
            "end_u": MDA.core.universe.Merge(
                h,
                rad,
                caps,
                core,
            ),
            "meta": {
                "translation": translation,
                "u1_name": rad[0].resname.lower() + "-sim",
                "u2_name": h[0].resname.lower() + "-sim",
                "charge_u1": _get_charge(rad[0]),
                "charge_u2": _get_charge(h[0]),
                "trajectory": u._trajectory.filename,
                "frame": ts.frame,
                "indices": (*h.ix, *rad.ix, *caps_ix, *h_aa.ix, *rad_aa.ix),
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


def extract_subsystems(
    u,
    rad_idxs,
    h_cutoff=3,
    env_cutoff=7,
    start=None,
    stop=None,
    step=None,
    unique=False,
):
    """Builds subsystems out of a trajectory for evaluation of HAT reaction
    either by DFT or a ML model. Aminoacids are capped at peptide bonds
    resulting in amines and amides. Subsystems contain the reactive hydrogen at
    index 0 followed by the radical atom.
    Note: This adaptes the residue names in given universe to the break

    Parameters
    ----------
    u : MDA.Universe
        Main universe
    rad_idxs : List[int]
        Indices of the two radical atoms
    h_cutoff : int, optional
        Cutoff radius for hydrogen search around radical, by default 3
    env_cutoff : int, optional
        Cutoff radius for local env used for better performance, by default 7
    start : Union[int,None], optional
        For slicing the trajectory, by default None
    stop : Union[int,None], optional
        For slicing the trajectory, by default None
    step : Union[int,None], optional
        For slicing the trajectory, by default None
    unique : bool
        If true, only keep one of every set of atoms.

    Returns
    -------
    List
        List of capped subsystems
    """

    assert len(rad_idxs) in [1, 2], "Error: One or two radicals must be given!"

    rads = [u.select_atoms(f"index {rad}") for rad in rad_idxs]

    # Delete bonds between radicals
    if len(rad_idxs) > 1:
        combs = combinations(rad_idxs, 2)
        for c in combs:
            try:
                u.delete_bonds([c])
            except ValueError as e:
                continue

    print("Calculating radical neighbors..")
    bonded_all = [u.select_atoms(f"bonded index {rad}") for rad in rad_idxs]
    bonded_all = [b - sum(rads) for b in bonded_all]  # remove rads

    # correct residue of radicals to avoid residues w/ only 2 atoms
    # Necessary in case of backbone break other than peptide bond
    print("Correcting residue of radical group..")
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

    capped_systems = {}

    for ts in tqdm(u.trajectory[slice(start, stop, step)]):
        if len(rads) > 1:
            if np.linalg.norm(rads[0].positions[0] - rads[1].positions[0]) < 3:
                print(f"Radical distance too small in frame {ts.frame}, skipping..")
                continue

        for rad, bonded_rad in zip(rads, bonded_all):
            capped_frame = cap_single_rad(u, ts, rad, bonded_rad, h_cutoff, env_cutoff)

            for i, capped in enumerate(capped_frame):
                if unique:
                    new_i_hash = hash(capped["meta"]["indices"])
                else:
                    new_i_hash = i

                # skip existing systems w/ bigger translation
                if new_i_hash in capped_systems.keys():
                    if capped["meta"]["translation"] > capped_systems[new_i_hash][0]:
                        continue

                capped_systems[new_i_hash] = (
                    capped["meta"]["translation"],
                    capped,
                )

    print(f"Created {len(capped_systems)} isolated systems.")
    return list(capped_systems.values())


def save_capped_systems(systems, out_dir):
    """Saves output from `extract_subsystems`

    Parameters
    ----------
    systems : list
        Systems to save the structures and meta file for.
    out_dir : Path
        Where to save. Should probably be traj/batch_238/se
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    for system in tqdm(systems):
        system = system[1]  # 0 is translation
        sys_hash = f'{system["meta"]["hash_u1"]}_{system["meta"]["hash_u2"]}'

        if (out_dir / f"{sys_hash}.npz").exists():
            print(f"ERROR: {sys_hash} hash exists!")
            continue

        system["start_u"].atoms.write(out_dir / f"{sys_hash}_1.pdb")
        system["end_u"].atoms.write(out_dir / f"{sys_hash}_2.pdb")

        system["meta"]["meta_path"] = out_dir / f"{sys_hash}.npz"

        np.savez(out_dir / f"{sys_hash}.npz", system["meta"])


def make_radicals(
    u: MDA.Universe,
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
):
    """Takes non radical trajectory and makes radical
    trajectories from it.

    Parameters
    ----------
    u : MDA.Universe
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
        For debugging, index of H to select, instead of chosing a random one. By default None
    """

    all_heavy = u.select_atoms("not element H")
    all_H = u.select_atoms("element H")
    if resnames is None:
        sel_Hs = all_H[random.sample(range(len(all_H)), count)]
    else:
        res_Hs = all_H.select_atoms(f"around {res_cutoff} resname {' '.join(resnames)}")
        sel_Hs = res_Hs[random.sample(range(len(res_Hs)), count)]

    capped_systems = []
    if h_index is not None:
        sel_Hs = [u.select_atoms(f"index {h_index}")[0]]
    for sel_H in sel_Hs:
        print(f"Selected H to remove: {sel_H}")
        rad = sel_H.bonded_atoms

        # remove one H and reorder atoms
        sub_atoms = rad + (all_H - sel_H) + (all_heavy - rad)

        u_radical = MDA.Merge(sub_atoms)
        u_radical.load_new(str(xtc), format=XTCReader, sub=sub_atoms.indices)

        try:
            subs = extract_subsystems(
                u_radical,
                [0],
                start=start,
                stop=stop,
                step=step,
                unique=unique,
                h_cutoff=h_cutoff,
            )
        except Exception as e:
            print(f"Selected H: {sel_H}")
            raise e

        for sub in subs:
            sub[1]["meta"]["traj_H"] = sel_H.index
        capped_systems.extend(subs)

        if out_dir is not None:
            save_capped_systems(subs, out_dir)
            print(f"Saved {len(subs)} systems in {out_dir}")

    return capped_systems


def closest(l, K):
    l = np.asarray(l)
    return l[(np.abs(l - K)).argmin()]


def make_radicals_smart(
    u: MDA.Universe,
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
):
    """Takes non radical trajectory and makes radical
    trajectories from it.
    More efficient sampling for small distances, uniform sampling across found distances

    Parameters
    ----------
    u : MDA.Universe
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
    """
    all_heavy = u.select_atoms("not element H")
    all_Hs = u.select_atoms("element H")

    sub_Hs = []
    for ts in tqdm(u.trajectory[start:stop:search_step], "Searching close H"):
        # define smaller sub-search spaces
        for _ in range(20):
            center_idx = all_Hs[random.sample(range(len(all_Hs)), 1)][0].index
            local_Hs = all_Hs.select_atoms(f"around 15 index {center_idx}")

            d = self_distance_array(local_Hs.positions)
            k = 0
            for i in range(len(local_Hs)):
                for j in range(i + 1, len(local_Hs)):
                    if d[k] < h_cutoff:
                        # if (local_Hs[j], d[k], ts.frame) not in sub_Hs:
                        sub_Hs.append((local_Hs[j], d[k], ts.frame))
                    k += 1
    sub_Hs = np.array(sub_Hs)
    print("Found small distances:", sub_Hs.shape)
    sub_Hs = sub_Hs[(np.unique(sub_Hs[:, 0], return_index=True))[1]]
    print("Found unique systems:", sub_Hs.shape)

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
    print(
        f"Selected {len(idxs)} Hs at distances from {sub_Hs[:, 1].min():.02f} to {sub_Hs[:, 1].max():.02f}"
    )
    # Building of universes
    capped_systems = []
    for sel_H, frame in sel_Hs:
        print(f"Selected H to remove: {sel_H}")
        rad = sel_H.bonded_atoms

        # remove one H and reorder atoms
        sub_atoms = rad + (all_Hs - sel_H) + (all_heavy - rad)

        u_radical = MDA.Merge(sub_atoms)
        u_radical.load_new(str(xtc), format=XTCReader, sub=sub_atoms.indices)

        try:
            subs = extract_subsystems(
                u_radical,
                [0],
                start=frame - window,
                stop=frame + window,
                step=step,
                unique=unique,
                h_cutoff=h_cutoff,
            )
        except Exception as e:
            print(f"Selected H: {sel_H}")
            raise e
        for sub in subs:
            sub[1]["meta"]["traj_H"] = sel_H.index
        capped_systems.extend(subs)

        if out_dir is not None:
            save_capped_systems(subs, out_dir)
            print(f"Saved {len(subs)} systems in {out_dir}")

    return capped_systems
