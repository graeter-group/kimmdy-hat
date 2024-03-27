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
    config = DummyClass()
    config.reactions = DummyClass()
    config.reactions.Hat_reaction = DummyClass()
    config.reactions.Hat_reaction.arrhenius_equation = DummyClass()
    config.reactions.Hat_reaction.arrhenius_equation.frequency_factor = 1e8
    config.reactions.Hat_reaction.arrhenius_equation.temperature = 300
    config.reactions.Hat_reaction.h_cutoff = 3
    config.reactions.Hat_reaction.model = None
    config.reactions.Hat_reaction.ensemble_size = None
    config.reactions.Hat_reaction.polling_rate = 10
    config.reactions.Hat_reaction.radicals = None
    config.reactions.Hat_reaction.change_coords = "place"
    config.reactions.Hat_reaction.kmc = "extrande"
    config.reactions.Hat_reaction.keep_structures = False

    def __init__(self):
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
