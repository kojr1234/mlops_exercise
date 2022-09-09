import json
import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from classifier_model.config.core import DATASET_DIR, config

ROOT = Path.home()
KAGGLE_CRED_DIR = ROOT / ".kaggle"
KAGGLE_CRED_FILE = KAGGLE_CRED_DIR / "kaggle.json"


def download_data() -> None:

    if "train.csv" not in os.listdir(DATASET_DIR) and "test.csv" not in os.listdir(
        DATASET_DIR
    ):
        try:
            credentials = {}
            credentials["username"] = os.environ["KAGGLE_USERNAME"]
            credentials["key"] = os.environ["KAGGLE_KEY"]

            if not KAGGLE_CRED_DIR.is_dir():
                os.mkdir(ROOT / ".kaggle")

            if not KAGGLE_CRED_FILE.is_file():
                with open(KAGGLE_CRED_FILE, "w") as f:
                    json.dump(credentials, f)
        except KeyError as e:
            print(e)

        api = KaggleApi()
        api.authenticate()
        api.competition_download_file(
            competition="spaceship-titanic", file_name="train.csv", path=DATASET_DIR
        )

        old_file = DATASET_DIR / "train.csv"
        new_file = DATASET_DIR / config.app_config.full_data

        os.rename(old_file, new_file)
