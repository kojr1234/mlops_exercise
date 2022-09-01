import pandas as pd
import logging

from sklearn.model_selection import train_test_split

from classifier_model.config.core import (config, LOGS_DIR)
from classifier_model.pipeline import spaceship_titanic_pipeline
from classifier_model.preprocessing.data_manager import load_dataset, save_pipeline

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

logging.basicConfig(filename=LOGS_DIR/'training_pipeline.log', filemode='w', level=logging.DEBUG)

logger = logging.getLogger('training_pipeline')

def log_performance(*, pipeline: Pipeline, test_X: pd.DataFrame, test_y: pd.Series) -> None:

    pred = pipeline.predict(test_X)
    acc = accuracy_score(test_y, pred)

    print(f"Model's accuracy: {round(acc,3)}")


def run_training() -> None:
    """
    Train the model
    """

    logger.info(f"Running training function")
    data = load_dataset(file_name=config.app_config.train_data)

    logger.info(f"Dataset loaded")
    X_, X, y_, y = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # improve the code to support parameter tunning
    spaceship_titanic_pipeline.fit(X_, y_)
    logger.info(f"Model Trained")

    log_performance(
        pipeline=spaceship_titanic_pipeline,
        test_X=X,
        test_y=y
    )

    save_pipeline(pipeline_to_persist=spaceship_titanic_pipeline)
    logger.info(f"Pipeline Saved")
    logger.info(f"Training completed!")


if __name__ == "__main__":
    run_training()
