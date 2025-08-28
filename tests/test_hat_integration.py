import os
import subprocess as sp
from pathlib import Path

import pytest

from kimmdy.cmd import kimmdy_run
from kimmdy.constants import MARK_DONE, MARK_FINISHED
from kimmdy.parsing import read_top, write_top
from kimmdy.plugins import parameterization_plugins
from kimmdy.topology.topology import Topology
from kimmdy.utils import get_task_directories


def read_last_line(file):
    with open(file, "rb") as f:
        try:  # catch OSError in case of one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        return f.readline().decode()


@pytest.mark.parametrize("arranged_tmp_path", (["test_hat_integration"]), indirect=True)
def test_integration_hat_reaction(arranged_tmp_path):

    kimmdy_run()
    assert "Finished running last task" in read_last_line(
        Path("alanine_hat_000.kimmdy.log")
    )
    assert (
        len(list(Path.cwd().glob("alanine_hat_000/*"))) == 22
    )  # don't forget, .kimmdy_finished counts
