"""File for pytest configuration in python and fixture definition.

The name 'conftest.py' is recognized by pytest to execute it before tests.
"""

import pytest
import shutil
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from kimmdy.plugins import discover_plugins

from kimmdy.tasks import TaskFiles
from kimmdy.utils import get_gmx_dir

## fixtures for setup and teardown ##
@pytest.fixture
def arranged_tmp_path(tmp_path: Path, request: pytest.FixtureRequest):
    """Arrange temporary directory for tests.

    With files for the test and a symlink to forcefield.
    """

    # if fixture was parameterized, use this for directory with input files
    if hasattr(request, "param"):
        file_dir = Path(__file__).parent / request.param
    # else use stem of requesting file to find directory with input files
    else:
        file_dir = Path(__file__).parent / request.path.stem
    # arrange tmp_path
    
    shutil.copytree(file_dir, tmp_path, dirs_exist_ok=True)
    assetsdir = Path(__file__).parent.parent.parent.parent / "tests" / "test_files" / "assets"
        
    if not (tmp_path / "amber99sb-star-ildnp.ff").exists():
        Path(tmp_path / "amber99sb-star-ildnp.ff").symlink_to(
            assetsdir / "amber99sb-star-ildnp.ff",
            target_is_directory=True,
        )
    # change cwd to tmp_path
    os.chdir(tmp_path.resolve())
    return tmp_path

# Functions for parsing --gpu to enable gpu teest
def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        dest="gpu",
        default=False,
        help="enable gpu memory release tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test to be ran on GPU")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)