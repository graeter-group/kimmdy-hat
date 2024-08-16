# %%
from pathlib import Path
from HATreaction import HAT_reaction
import pytest
from pprint import pprint
import logging


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
        self.config.reactions.Hat_reaction.unique = False
        self.config.changer = DummyClass()
        self.config.changer.topology = DummyClass()
        self.config.changer.topology.parameterization = "grappa"
        self.top = DummyClass()
        self.top.radicals = {}


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
    rmg_unique.config.reactions.Hat_reaction.unique = True
    plgn = HAT_reaction("Hat_reaction", rmg_unique)

    files = DummyClass()
    files.input = {
        "tpr": Path(__file__).parent / "test_traj_io" / "equilibrium1.tpr",
        "trr": Path(__file__).parent / "test_traj_io" / "equilibrium1.trr",
    }
    files.outputdir = Path(tmpdir)

    return plgn.get_recipe_collection(files)


def test_traj_unique(recipe_collection_unique):
    print(recipe_collection_unique.recipes)
    assert len(recipe_collection_unique.recipes) == 5

    for recipe in recipe_collection_unique.recipes:
        assert recipe.recipe_steps[0].atom_ix_1 < recipe.recipe_steps[0].atom_ix_2
        assert recipe.recipe_steps[0].atom_ix_1 < recipe.recipe_steps[0].atom_ix_2
    l = [r.recipe_steps[0] for r in recipe_collection_unique.recipes]
    s = {r.recipe_steps[0] for r in recipe_collection_unique.recipes}
    assert all([r in s for r in l])


def test_traj_to_recipes(recipe_collection):
    print(recipe_collection.recipes)
    assert len(recipe_collection.recipes) == 15
    recipe_collection.aggregate_reactions()
    assert len(recipe_collection.recipes) == 15

    for recipe in recipe_collection.recipes:
        assert len(recipe.rates) == 1
        assert len(recipe.timespans) == 1

    # remove 'place'
    [r.recipe_steps.pop(1) for r in recipe_collection.recipes]
    recipe_collection.aggregate_reactions()

    assert len(recipe_collection.recipes) == 5

    for recipe in recipe_collection.recipes:
        assert len(recipe.rates) == 3
        assert len(recipe.timespans) == 3


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
    assert len(recipe_collection_pbc.recipes) == 4
    recipe_collection_pbc.aggregate_reactions()
    assert len(recipe_collection_pbc.recipes) == 4

    for recipe in recipe_collection_pbc.recipes:
        assert len(recipe.rates) == 1
        assert len(recipe.timespans) == 1

    # remove 'place'
    [r.recipe_steps.pop(1) for r in recipe_collection_pbc.recipes]
    recipe_collection_pbc.aggregate_reactions()

    assert len(recipe_collection_pbc.recipes) == 2

    for recipe in recipe_collection_pbc.recipes:
        assert len(recipe.rates) == 2
        assert len(recipe.timespans) == 2
