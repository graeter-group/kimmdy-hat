import MDAnalysis as mda
import numpy as np
from multiprocessing import Process, Queue


def find_radicals(u):
    """
    finds radicals in a MDAnalysis universe
    """
    nbonds_dict = {
        ("MG", "NA", "CO"): 0,
        (
            "H",
            "HW",
            "HO",
            "HS",
            "HA",
            "HC",
            "H1",
            "H2",
            "H3",
            "HP",
            "H4",
            "H5",
            "HO",
            "H0",
            "HP",
            "O",
            "O2",
            "Cl",
            "Na",
            "I",
            "F",
            "Br",
        ): 1,
        ("NB", "NC", "OW", "OH", "OS", "SH", "S"): 2,
        (
            "C",
            "CN",
            "CB",
            "CR",
            "CK",
            "CC",
            "CW",
            "CV",
            "C*",
            "CQ",
            "CM",
            "CA",
            "CD",
            "CZ",
            "N",
            "NA",
            "N*",
            "N2",
        ): 3,
        ("CT", "N3", "P", "SO"): 4,
    }  # compare to atom type perception paper (2006) same as in changemanager.py
    atoms = []
    for atom in u.atoms:
        if atom.resname == "SOL":
            break  # empty atom group
        try:
            nbonds = [v for k, v in nbonds_dict.items() if atom.type in k][0]
        except IndexError:
            raise IndexError(
                "{} not in atomtype dictionary nbonds_dict".format(atom.type)
            )
        if len(atom.bonded_atoms) < nbonds:
            atoms.append(mda.AtomGroup([atom]))
    return atoms


def check_cylinderclash(a, b, t, r_min=0.8, d_min=None, verbose=False):
    """Checks for atoms in a cylinder around a possible HAT reaction path.
    Cylinder begins/ends 10% after/before the end points.
    Ref.: https://geomalgorithms.com/a02-_lines.html

    Parameters
    ----------
    a : Union[np.ndarray, list]
        Center of cylinder base
    b : Union[np.ndarray, list]
        Center of cylinder top
    t : Union[np.ndarray, list]
        Testpoint, or list of testpoints
    d_min
    r_min : int, optional
        Radius of cylinder, by default 0.8

    Returns
    -------
    bool
        True if no points are inside the cylinder, False otherwise.
    """

    def _check_point(a, b, t, r_min, verbose):
        v = b - a  # apex to base
        w = t - a  # apex to testpoint
        c1 = np.dot(v, w)
        c2 = np.dot(v, v)
        x = c1 / c2  # percentage along axis

        tx = a + (x * v)  # projection of t on x
        r = np.linalg.norm(t - tx)

        if x < 0.1 or x > 0.9 or r > r_min:
            # point is outside cylinder
            return True
        if verbose:
            print(f"a = {a}, b = {b}, t = {t}")
            print(f"x = {x}, r = {r}")
        return False

    if isinstance(t, np.ndarray) and t.shape == (3,):
        return _check_point(a, b, t, r_min, verbose)

    elif (
        (isinstance(t, np.ndarray) and t.shape[1] == 3)
        or isinstance(t, list)
        and len(t[0]) == 3
    ):
        return all(
            [
                _check_point(np.array(a), np.array(b), np.array(p), r_min, verbose)
                for p in t
            ]
        )

    else:
        raise ValueError(f"Type/Shape of t wrong: {t}")


# Decorator and helper function to free the gpu
def _queue_helper(func, q, *args, **kwargs):
    res = func(*args, **kwargs)
    q.put(res)


def free_gpu(func):
    def wrapper(*args, **kwargs):
        q = Queue(1)
        p = Process(target=_queue_helper, args=(func, q, *args), kwargs=kwargs)
        p.start()
        res = q.get()
        p.join()
        p.close()
        q.close()
        return res

    return wrapper
