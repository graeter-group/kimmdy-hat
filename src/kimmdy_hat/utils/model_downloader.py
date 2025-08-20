import logging
import certifi
import ssl
from urllib import request
from pathlib import Path
from shutil import copytree
import zipfile
import tempfile


URL = "https://github.com/graeter-group/kimmdy-hat/archive/refs/tags/v0.1.1.zip"
MODELS = [
    "classic_models",
    "grappa_models",
]
logger = logging.getLogger(__name__)


def download_models(target_dir):
    print("Starting to download HAT prediction models..", end="", flush=True)
    context = ssl.create_default_context(cafile=certifi.where())

    url = request.urlopen(URL, context=context)
    with tempfile.TemporaryDirectory() as tmp_dir:
        zipped = Path(tmp_dir) / "kimmdy_hat.zip"
        unzipped = Path(tmp_dir) / "kimmdy_hat"

        with open(zipped, "b+w") as fp:
            fp.write(url.read())

        with zipfile.ZipFile(zipped, "r") as zip_f:
            zip_f.extractall(unzipped)

        model_paths = []
        for model in MODELS:
            model_paths.extend(list(unzipped.glob(f"**/{model}")))

        assert len(MODELS) == len(
            model_paths
        ), f"Not all models could be downloaded, found: {model_paths}"

        for model in model_paths:
            if (target_dir / model.name).exists():
                logger.debug(f"{model.name} exists already.")
                continue
            logger.debug(f"Copy {model.name} to {target_dir}")
            copytree(model, target_dir / model.name)

        print(" Done!")

    logger.debug("Removed downloads")


def ensure_models_exist():
    models_path = Path(__file__).parent.parent.parent / "HATmodels"
    existing = [(models_path / m).exists() for m in MODELS]
    download = False
    if all(existing):
        logging.debug("Found all expected HAT models.")
    elif any(existing):
        print("HAT Models download not complete.")
        download = True
    else:
        print("HAT models not downloaded yet.")
        download = True

    if download:
        download_models(models_path)
