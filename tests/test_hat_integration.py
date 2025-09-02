import os
from pathlib import Path

from kimmdy.cmd import kimmdy_run
from kimmdy.constants import MARK_DONE, MARK_FINISHED
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


def test_integration_hat_reaction(arranged_tmp_path):
    kimmdy_run()
    assert "Finished running last task" in read_last_line(
        Path("alanine_hat_000.kimmdy.log")
    )
    assert (
        len(list(Path.cwd().glob("alanine_hat_000/*"))) == 12
    )  # don't forget, .kimmdy_finished counts


def test_integration_hat_restart(arranged_tmp_path):
    run_dir = Path("alanine_hat_000")
    kimmdy_run(input=Path("kimmdy_restart.yml"))
    n_files_original = len(list(run_dir.glob("*")))

    # restart already finished run
    kimmdy_run(input=Path("kimmdy_restart.yml"))
    assert "already finished" in read_last_line(Path("alanine_hat_000.kimmdy.log"))

    # try restart from stopped md
    task_dirs = get_task_directories(run_dir)
    (task_dirs[-1] / MARK_DONE).unlink()
    (arranged_tmp_path / run_dir / MARK_FINISHED).unlink()
    kimmdy_run(input=Path("kimmdy_restart.yml"))
    n_files_continue_md = len(list(run_dir.glob("*")))

    assert "Finished running last task" in read_last_line(
        Path("alanine_hat_000.kimmdy.log")
    )
    assert n_files_original == n_files_continue_md == 15

    # try restart from finished md
    task_dirs = get_task_directories(run_dir)
    (task_dirs[-4] / MARK_DONE).unlink()
    (arranged_tmp_path / run_dir / MARK_FINISHED).unlink()
    kimmdy_run(input=Path("kimmdy_restart.yml"))
    n_files_restart = len(list(run_dir.glob("*")))

    assert "Finished running last task" in read_last_line(
        Path("alanine_hat_000.kimmdy.log")
    )
    assert n_files_original == n_files_restart == 15
