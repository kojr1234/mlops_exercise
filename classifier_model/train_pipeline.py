import logging

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from classifier_model.split_data import split_train_test_data
from classifier_model.config.core import DATASET_DIR, LOGS_DIR, config
from classifier_model.pipeline import spaceship_titanic_pipeline
from classifier_model.preprocessing.data_manager import load_dataset, save_pipeline

logging.basicConfig(
    filename=LOGS_DIR / "training_pipeline.log", filemode="w", level=logging.DEBUG
)

logger = logging.getLogger("training_pipeline")


def log_performance(
    *, pipeline: Pipeline, test_X: pd.DataFrame, test_y: pd.Series
) -> None:

    pred = pipeline.predict(test_X)
    acc = round(accuracy_score(test_y, pred), 3)

    print(f"Model's accuracy: {acc}")


def run_training() -> None:
    """
    Train the model
    """

    logger.info("Running training function")

    split_train_test_data()

    logger.info("Dataset loaded")
    train = pd.read_csv(DATASET_DIR/config.app_config.train_data)
    test = pd.read_csv(DATASET_DIR/config.app_config.test_data)

    # improve the code to support parameter tunning
    spaceship_titanic_pipeline.fit(
        train[config.model_config.features], 
        train[config.model_config.target]
    )
    logger.info("Model Trained")

    log_performance(
        pipeline=spaceship_titanic_pipeline,
        test_X=test[config.model_config.features],
        test_y=test[config.model_config.target])

    save_pipeline(pipeline_to_persist=spaceship_titanic_pipeline)
    logger.info("Pipeline Saved")
    logger.info("Training completed!")


if __name__ == "__main__":
    run_training()
