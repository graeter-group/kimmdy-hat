import pytest


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
