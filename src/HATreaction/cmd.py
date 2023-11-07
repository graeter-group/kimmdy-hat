import argparse
import logging
from pathlib import Path
import sys

if sys.version_info > (3, 10):
    from importlib_metadata import version
else:
    from importlib.metadata import version


def get_cmdline_args():
    """Parse command line arguments and configure logger.

    Returns
    -------
    Namespace
        parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "Hydrogen atom tranfer rate predictions\n"
            "Predicts HAT rates using an ensemble graph neural network"
        )
    )
    parser.add_argument(
        "--version", action="version", version=f'HAT reactions {version("HATreaction")}'
    )
    parser.add_argument(
        "--top",
        type=Path,
        help="topology file, preferably tpr",
    )
    parser.add_argument(
        "--traj",
        type=Path,
        help="trajectory file, preferably trr",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Create analysis plots for reaction results.",
    )
