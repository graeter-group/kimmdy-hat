# %%
from pathlib import Path
from kimmdy_hat import HAT_reaction
import pytest
from pprint import pprint
import logging
import subprocess


# %%
class DummyClass:
    logger = logging.getLogger()


class DummyRunmanager:
    def __init__(self):
        self.config = DummyClass()
        self.config.reactions = DummyClass()
        self.config.reactions.Hat_reaction = DummyClass()
        self.config.reactions.Hat_reaction.arrhenius_equation = DummyClass()
        self.config.reactions.Hat_reaction.arrhenius_equation.frequency_factor = 1e8
        self.config.reactions.Hat_reaction.arrhenius_equation.temperature = 300
        self.config.reactions.Hat_reaction.h_cutoff = 3
        self.config.reactions.Hat_reaction.model = None
        self.config.reactions.Hat_reaction.ensemble_size = None
        self.config.reactions.Hat_reaction.prediction_scheme = "efficient"
        self.config.reactions.Hat_reaction.polling_rate = 10
        self.config.reactions.Hat_reaction.radicals = None
        self.config.reactions.Hat_reaction.change_coords = "place"
        self.config.reactions.Hat_reaction.kmc = "extrande"
        self.config.reactions.Hat_reaction.keep_structures = False
        self.config.reactions.Hat_reaction.cap = False
        self.config.reactions.Hat_reaction.n_unique = 0
        self.config.reactions.Hat_reaction.trajectory_format = "trr"
        self.config.changer = DummyClass()
        self.config.changer.topology = DummyClass()
        self.config.changer.topology.parameterization = "grappa"
        self.top = DummyClass()
        self.top.radicals = {}
        self.latest_files = {
            "tpr": Path(__file__).parent / "test_traj_io" / "equilibrium1.tpr",
            "trr": Path(__file__).parent / "test_traj_io" / "equilibrium1.trr",
        }


@pytest.fixture
def recipe_collection(tmpdir):
    plgn = HAT_reaction("Hat_reaction", DummyRunmanager())

    files = DummyClass()
    files.input = {
        "tpr": Path(__file__).parent / "test_traj_io" / "equilibrium1.tpr",
        "trr": Path(__file__).parent / "test_traj_io" / "equilibrium1.trr",
    }
    files.outputdir = Path(tmpdir)

    return plgn.get_recipe_collection(files)


@pytest.fixture
def recipe_collection_unique(tmpdir):
    rmg_unique = DummyRunmanager()
    rmg_unique.config.reactions.Hat_reaction.n_unique = 1
    plgn = HAT_reaction("Hat_reaction", rmg_unique)

    files = DummyClass()
    files.input = {
        "tpr": Path(__file__).parent / "test_traj_io" / "equilibrium1.tpr",
        "trr": Path(__file__).parent / "test_traj_io" / "equilibrium1.trr",
    }
    files.outputdir = Path(tmpdir)

    return plgn.get_recipe_collection(files)


@pytest.fixture
def recipe_collection_n_unique(tmpdir):
    rmg_unique = DummyRunmanager()
    rmg_unique.config.reactions.Hat_reaction.n_unique = 2
    rmg_unique.config.reactions.Hat_reaction.keep_structures = True
    plgn = HAT_reaction("Hat_reaction", rmg_unique)

    files = DummyClass()
    files.input = {
        "tpr": Path(__file__).parent / "test_traj_io" / "equilibrium1.tpr",
        "trr": Path(__file__).parent / "test_traj_io" / "equilibrium1.trr",
    }
    files.outputdir = Path(tmpdir)

    return plgn.get_recipe_collection(files), files


def test_traj_unique(recipe_collection_unique):
    print(recipe_collection_unique.recipes)
    assert len(recipe_collection_unique.recipes) == 5

    for recipe in recipe_collection_unique.recipes:
        assert recipe.recipe_steps[0].atom_ix_1 < recipe.recipe_steps[0].atom_ix_2
        assert recipe.recipe_steps[0].atom_ix_1 < recipe.recipe_steps[0].atom_ix_2
    l = [r.recipe_steps[0] for r in recipe_collection_unique.recipes]
    s = {r.recipe_steps[0] for r in recipe_collection_unique.recipes}
    assert all([r in s for r in l])


# test deprecated because n_unique max_idx is not longer part of hash
# def test_traj_n_unique_files(recipe_collection_n_unique):
#     recipe_collection, files = recipe_collection_n_unique
#     print(recipe_collection.recipes)
#     assert len(recipe_collection.recipes) == 10

#     f_list = [p for p in files.outputdir.glob("se/*_0_*.pdb")]
#     f_list_all = [p for p in files.outputdir.glob("se/*.pdb")]
#     for f in f_list:
#         f_list_all.remove(f)
#         parts = f.name.split("_")
#         other = f.with_name(f"{parts[0]}_1_{parts[2]}")
#         assert other.exists()
#         f_list_all.remove(other)

#     assert len(f_list_all) == 0


def test_traj_to_recipes(recipe_collection):
    print(recipe_collection.recipes)
    assert len(recipe_collection.recipes) == 18
    recipe_collection.aggregate_reactions()
    assert len(recipe_collection.recipes) == 18

    for recipe in recipe_collection.recipes:
        assert len(recipe.rates) == 1
        assert len(recipe.timespans) == 1

    # remove 'place'
    [r.recipe_steps.pop(1) for r in recipe_collection.recipes]
    recipe_collection.aggregate_reactions()

    assert len(recipe_collection.recipes) == 5

    for recipe in recipe_collection.recipes:
        assert len(recipe.rates) in [3, 6]
        assert len(recipe.timespans) in [3, 6]


@pytest.fixture
def gpu_info(recipe_collection):
    gpu_list = subprocess.check_output("nvidia-smi -L", shell=True)
    gpu_names = subprocess.check_output(
        "nvidia-smi --query-compute-apps=name --format=csv,noheader,nounits",
        shell=True,
    )
    return [gpu_list.decode("utf-8").rstrip(), (gpu_names.decode("utf-8"))]


@pytest.mark.gpu
def test_gpu_memory_release(gpu_info):
    assert "GPU" in gpu_info[0]
    assert len(gpu_info[1]) == 0


@pytest.fixture
def recipe_collection_pbc(tmpdir):
    plgn = HAT_reaction("Hat_reaction", DummyRunmanager())

    files = DummyClass()
    files.input = {
        "tpr": Path(__file__).parent / "test_traj_io" / "dopa_pbc.tpr",
        "trr": Path(__file__).parent / "test_traj_io" / "dopa_pbc.trr",
    }
    files.outputdir = Path(tmpdir)

    return plgn.get_recipe_collection(files)


def test_traj_to_recipes_pbc(recipe_collection_pbc):
    print(recipe_collection_pbc.recipes)
    assert len(recipe_collection_pbc.recipes) == 26
    recipe_collection_pbc.aggregate_reactions()
    assert len(recipe_collection_pbc.recipes) == 26

    for recipe in recipe_collection_pbc.recipes:
        assert len(recipe.rates) == 1
        assert len(recipe.timespans) == 1

    # remove 'place'
    [r.recipe_steps.pop(1) for r in recipe_collection_pbc.recipes]
    recipe_collection_pbc.aggregate_reactions()

    assert len(recipe_collection_pbc.recipes) == 2

    for recipe in recipe_collection_pbc.recipes:
        assert len(recipe.rates) in [18, 8]
        assert len(recipe.timespans) in [18, 8]
