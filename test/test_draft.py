#%%
import numpy as np
from pathlib import Path
import MDAnalysis as MDA
import nglview as ngl
from HATreaction.utils import trajectory_utils
import random
import json

# @@@ Test capping @@@

#%% create random manual verified testests
u = MDA.Universe(
    str(Path(__file__).parent / "tri_helix.gro"),
    guess_bonds=True,
    vdwradii={'DUMMY': 0.0}
)
u.add_TopologyAttr("elements", u.atoms.types)
if 0:
    view = ngl.show_mdanalysis(u, gui=True)
    view.close()

#%%
def gen_random_cap(rng_count=5):
        res_prot = u.select_atoms("not resname SOL NA CL").residues
        res_sel = []
        
        res_sel.append(u.select_atoms("resname NME").residues[0])
        res_sel.append(u.select_atoms("resname ACE").residues[0])
        res_sel.extend((
            u.select_atoms("(bonded resname NME) and (not resname NME)").residues[0],
            u.select_atoms("(bonded resname ACE) and (not resname ACE)").residues[0]
        ))

        res_sel.extend(
            [res_prot[random.randrange(len(res_prot))] for _ in range(rng_count)]
        )

        save_dir = Path("/hits/fast/mbm/riedmiki/nn/HAT_reaction_plugin/test/test_capping_io")

        for res in res_sel:
            atms = res.atoms

            cap, cap_idxs = trajectory_utils.cap_aa(atms)

            view = ngl.show_mdanalysis(cap, default_representation=False)
            view.add_licorice()
            view.add_component(atms)
            yield view
            inp = input("Accept? [y/n/name]")
            
            if inp not in ['n','N']:
                to_save = {
                    'inp_atm_idx' : [int(p) for p in atms.indices],
                    'cap_idxs' : [int(p) for p in cap_idxs],
                    'cap_positions_flat' : [float(p) for p in cap.positions.reshape(-1)],
                }

                out_p = save_dir / (str(random.randint(0,100000))+".json")

                if len(inp) > 1:
                    out_p = save_dir / (str(inp)+".json")

                with open(out_p, 'w') as f:
                    json.dump(to_save, f, indent=1)
                print('Saved to', out_p.name)
            else:
                print('Not saving..')


gen = gen_random_cap()
#%%
gen.__next__()

#%%


#%%


#%%
