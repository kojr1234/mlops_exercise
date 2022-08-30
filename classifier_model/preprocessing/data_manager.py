import logging
import re

from pathlib import Path
from typing import Any, List, Union

import pandas as pd
import joblib

from classifier_model import __version__ as _version
from classifier_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

logger = logging.getLogger(__name__)

def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Remove old model pipelines
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported.
    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file not in do_not_delete:
            model_file.unlink()

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    Simply load the dataset
    """

    df = pd.read_csv(f'{DATASET_DIR} / {file_name}')
    return df

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """
    Save the pipeline
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when package is
    published.
    """

    save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)