import logging
from urllib.request import urlretrieve
from pathlib import Path
from shutil import copytree, rmtree
import zipfile


URL = "https://github.com/graeter-group/kimmdy-hat/archive/refs/tags/v0.1.1.zip"
MODELS = [
    "classic_models",
    "grappa_models",
]
logger = logging.getLogger(__name__)


def download_models(target_dir):
    logger.info("Starting to download HAT prediction models..")
    tmp_path, header = urlretrieve(URL)
    tmp_path = Path(tmp_path)
    logger.info("Download finished!")
    logger.debug(f"Zip archive: {tmp_path}")

    unzipped = tmp_path.parent / "HATmodels"
    with zipfile.ZipFile(tmp_path, "r") as zip_f:
        zip_f.extractall(unzipped)

    model_paths = []
    for model in MODELS:
        model_paths.extend(list(unzipped.glob(f"**/{model}")))

    assert len(MODELS) == len(
        model_paths
    ), f"Not all models could be downloaded, found: {model_paths}"

    for model in model_paths:
        logger.debug(f"Copy {model.name} to {target_dir}")
        copytree(model, target_dir / model.name)

    tmp_path.unlink()
    rmtree(unzipped)
    logger.debug("Removed downloads")


def ensure_models_exist():
    models_path = Path(__file__).parent.parent.parent / "HATmodels"
    existing = [(models_path / m).exists() for m in MODELS]
    download = False
    if all(existing):
        logger.info("Found all expected HAT models.")
    elif any(existing):
        logger.info("HAT Models download not complete.")
        download = True
    else:
        logger.info("HAT models not downloaded yet.")
        download = True

    if download:
        download_models(models_path)
