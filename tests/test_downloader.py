from pathlib import Path
from kimmdy_hat.utils.model_downloader import (
    download_models,
    ensure_models_exist,
    MODELS,
)
import zipfile
import pytest


def zip_dummy(tmp_path: Path, directories: list[Path]):
    for c in directories:
        (tmp_path / c).mkdir()
        to_write = tmp_path / c / "dummy.txt"
        to_write.touch()
        with zipfile.ZipFile(tmp_path / "arch.zip", "a") as zip_f:
            zip_f.write(to_write)
    # return tmp_path / "arch.zip"
    return tmp_path / "arch.zip"


def test_download_models(mocker, tmp_path):
    contents = [Path(m) for m in MODELS]
    archive = zip_dummy(tmp_path, contents)

    urlopen_mock = mocker.patch("kimmdy_hat.utils.model_downloader.request.urlopen")
    with open(archive, "rb") as url:
        urlopen_mock.return_value = url
        download_models(tmp_path / "target")

    for m in MODELS:
        assert len(list((tmp_path / "target").glob(m))) == 1


def test_ensure_models_exist_all(mocker):
    downloader = mocker.patch("kimmdy_hat.utils.model_downloader.download_models")
    mock_all = mocker.patch("kimmdy_hat.utils.model_downloader.all")
    mock_all.return_value = True
    ensure_models_exist()
    downloader.assert_not_called()


def test_ensure_models_exist_any(mocker):
    downloader = mocker.patch("kimmdy_hat.utils.model_downloader.download_models")
    mock_all = mocker.patch("kimmdy_hat.utils.model_downloader.all")
    mock_all.return_value = False
    mock_any = mocker.patch("kimmdy_hat.utils.model_downloader.any")
    mock_any.return_value = True
    ensure_models_exist()
    downloader.assert_called_once()


def test_ensure_models_exist_none(mocker):
    downloader = mocker.patch("kimmdy_hat.utils.model_downloader.download_models")
    mock_all = mocker.patch("kimmdy_hat.utils.model_downloader.all")
    mock_all.return_value = False
    mock_any = mocker.patch("kimmdy_hat.utils.model_downloader.any")
    mock_any.return_value = False
    ensure_models_exist()
    downloader.assert_called_once()
