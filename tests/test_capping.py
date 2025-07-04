# %%
from pathlib import Path
from kimmdy_hat.utils import capping_utils
import pickle
import json
import pytest
import MDAnalysis as MDA


def pickle_universe():
    """Creates pickled MDAnalysis Universe for rapid loading.
    Pickles break between MDA version changes.
    """
    u = MDA.Universe(
        str(Path(__file__).parent / "test_capping_io" / "tri_helix.gro"),
        guess_bonds=True,
        vdwradii={"DUMMY": 0.0},
        in_memory=True,
    )
    u.add_TopologyAttr("elements", u.atoms.types)
    u.atoms.ids = u.atoms.ix
    with open(Path(__file__).parent / "test_capping_io" / "universe.pckl", "wb") as f:
        pickle.dump(u, f)


# create random manual verified testests
@pytest.fixture(scope="module")
def universe():
    try:
        u_p = Path(__file__).parent / "test_capping_io" / "universe.pckl"
        with open(u_p, "rb") as f:
            u = pickle.load(f)
        return u

    except Exception:
        print("Could not load pickled universe, recreating..", end="")
        pickle_universe()

    print("Done\nLoading new pickle")
    with open(u_p, "rb") as f:
        u = pickle.load(f)
    return u


test_files = list((Path(__file__).parent / "test_capping_io").glob("*.json"))


@pytest.fixture(params=test_files, ids=lambda p: p.name)
def cap_ref(request):
    with open(request.param) as f:
        ref = json.load(f)
    return ref


def test_capping(cap_ref, universe):
    inp_atm_idx = cap_ref["inp_atm_idx"]
    true_cap_idxs = cap_ref["cap_idxs"]
    true_cap_positions_flat = cap_ref["cap_positions_flat"]

    cap, cap_idxs = capping_utils.cap_aa(universe.copy().atoms[inp_atm_idx])

    assert true_cap_idxs == [int(p) for p in cap_idxs], "Cap ids wrong"
    assert true_cap_positions_flat == pytest.approx(
        [float(p) for p in cap.positions.reshape(-1)]
    ), "Cap positions wrong"


# %% ----- Create reference capping ------
# import nglview as ngl
# import random
# import MDAnalysis as MDA

# def gen_random_cap(u, rng_count=5):
#         res_prot = u.select_atoms("not resname SOL NA CL").residues
#         res_sel = []

#         res_sel.append(u.select_atoms("resname NME").residues[0])
#         res_sel.append(u.select_atoms("resname ACE").residues[0])
#         res_sel.extend((
#             u.select_atoms("(bonded resname NME) and (not resname NME)").residues[0],
#             u.select_atoms("(bonded resname ACE) and (not resname ACE)").residues[0]
#         ))

#         res_sel.extend(
#             [res_prot[random.randrange(len(res_prot))] for _ in range(rng_count)]
#         )

#         save_dir = Path("/hits/fast/mbm/riedmiki/nn/HAT_reaction_plugin/test/test_capping_io")

#         for res in res_sel:
#             atms = res.atoms

#             cap, cap_idxs = capping_utils.cap_aa(atms)

#             view = ngl.show_mdanalysis(atms, default_representation=False)
#             view.add_licorice(opacity=0.7)
#             view.add_component(cap)
#             yield view
#             inp = input("Accept? [y/n/name]")

#             if inp not in ['n','N']:
#                 to_save = {
#                     'inp_atm_idx' : [int(p) for p in atms.indices],
#                     'cap_idxs' : [int(p) for p in cap_idxs],
#                     'cap_positions_flat' : [float(p) for p in cap.positions.reshape(-1)],
#                 }

#                 out_p = save_dir / (str(random.randint(0,100000))+".json")

#                 if len(inp) > 1:
#                     out_p = save_dir / (str(inp)+".json")

#                 with open(out_p, 'w') as f:
#                     json.dump(to_save, f, indent=1)
#                 print('Saved to', out_p.name)
#             else:
#                 print('Not saving..')


# gen = gen_random_cap(universe(), 5)
# #%% --- save some cappings ---
# gen.__next__()

# #%% --- save pickled universe ---
# def pickle_universe():
#     u = MDA.Universe(
#         str(Path(__file__).parent / "test_capping_io" / "tri_helix.gro"),
#         guess_bonds=True,
#         vdwradii={"DUMMY": 0.0},
#         in_memory=True,
#     )
#     u.add_TopologyAttr("elements", u.atoms.types)
#     with open(Path(__file__).parent / "test_capping_io" / "universe.pckl", 'wb') as f:
#         pickle.dump(u, f)
