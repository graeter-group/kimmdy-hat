"""save_predictions.py
Skript for making and saving rate predictions and recipes based on
existing pdb+meta files.
"""

# %%
from pathlib import Path
import json
import shutil
import numpy as np
from tqdm.autonotebook import tqdm
from collections import defaultdict
from importlib.resources import files as res_files

from kimmdy_hat.reaction import make_predictions
import tensorflow as tf
import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError

load_model = tf.keras.models.load_model

# %% ----- Paths & Variables -----
root = Path("/hits/fast/mbm/riedmiki/kimmdy_runs/hat_correlation_lysozyme/kimmdy/")
se_dir = root / "lysozyme_stride_1000" / "1_hat_reaction" / "se_unique"
structure = root / "protein.pdb"
trajectory = root / "prod.xtc"

model_name = "grappa_models"
ensemble_size = 10
prediction_scheme = "efficient"  # `efficient` or `all_models`
R = 1.9872159e-3  # [kcal K-1 mol-1]
temperature = 300
polling_rate = 1000  # should match the rate used during writing
change_coords = "place"
frequency_factor = 0.288

# --------------------
assert se_dir.exists()
assert structure.exists()
assert trajectory.exists()
# %% ----- Load Universe -----
u = mda.Universe(str(structure), str(trajectory))
try:
    print(u.bonds)
except NoDataError:
    u.atoms.guess_bonds()
print(u.bonds)

# %% ----- Load Models -----
ensemble_dirs = list(res_files("HATmodels").glob(model_name))
assert len(ensemble_dirs) > 0, f"Model {model_name} not found."
assert len(ensemble_dirs) == 1, f"Multiple Models found for {model_name}."
ensemble_dir = ensemble_dirs[0]

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

# %% ----- Make Predictions -----
recipe_collection = make_predictions(
    u,
    se_dir,
    hparas,
    prediction_scheme,
    models,
    means,
    stds,
    R,
    temperature,
    polling_rate,
    change_coords,
    frequency_factor,
)

# %% ----- Saving recipes -----
recipe_collection.to_csv(
    Path(
        "/hits/fast/mbm/riedmiki/kimmdy_runs/hat_correlation_lysozyme/kimmdy/lysozyme_stride_1000/1_hat_reaction/recipes_unique.csv"
    )
)

# %% ----- build se dir with unique reactions -----


def get_coords_from_pdb(pdb: Path):
    with open(pdb) as f:
        finished = False
        while not finished:
            line = f.readline()
            if line[:11] == "ATOM      1":
                finished = True
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
    return np.array((x, y, z))


se_dir = Path(
    "/hits/fast/mbm/riedmiki/kimmdy_runs/hat_correlation_lysozyme/kimmdy"
    "/lysozyme_stride_2/1_hat_reaction/se"
)

uniques = defaultdict(lambda: (100, None))

for meta_p in tqdm(se_dir.glob("*.npz")):
    meta = np.load(meta_p, allow_pickle=True)
    meta_d = np.expand_dims(meta[meta.files[0]], axis=0)[0]

    ident = meta_d["indices"][:2]
    trans = meta_d["translation"]

    if trans < uniques[ident][0]:
        uniques[ident] = (trans, meta_p.stem)

se_unique_dir = se_dir.with_name("se_unique")
se_unique_dir.mkdir(exist_ok=True)

for v in tqdm(uniques.values()):
    h = v[1]
    if v[0] > 10:
        continue
    for p in se_dir.glob(h + "*"):
        if not (se_unique_dir / p.name).exists():
            (se_unique_dir / p.name).symlink_to(p)


# %%
