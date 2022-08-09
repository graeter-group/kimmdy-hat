import MDAnalysis as MDA

def find_radicals(u):
    """
    finds radicals in a MDAnalysis universe
    """
    nbonds_dict = {
        ('MG','NA','CO'):0,
        ('H','HW','HO','HS','HA','HC','H1','H2','H3','HP','H4','H5','HO','H0','HP','O','O2','Cl','Na','I','F','Br'):1,
        ('NB','NC','OW','OH','OS','SH','S'):2,
        ('C','CN','CB','CR','CK','CC','CW','CV','C*','CQ','CM','CA','CD','CZ','N','NA','N*','N2'):3,
        ('CT','N3','P','SO'):4}                                               #compare to atom type perception paper (2006) same as in changemanager.py
    atoms = []
    for atom in u.atoms:
        if atom.resname == 'SOL':
            break  #empty atom group  
        try:
            nbonds = [v for k,v in nbonds_dict.items() if atom.type in k][0]
        except IndexError:
            raise IndexError("{} not in atomtype dictionary nbonds_dict".format(atom.type))
        if len(atom.bonded_atoms) < nbonds:
            atoms.append(MDA.AtomGroup([atom]))
    if len(atoms) == 0:
        return [MDA.Universe.empty(0).atoms]
    return atoms 