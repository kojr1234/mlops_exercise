import logging

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from classifier_model.config.core import LOGS_DIR, config
from classifier_model.pipeline import spaceship_titanic_pipeline
from classifier_model.preprocessing.data_manager import load_dataset, save_pipeline
from classifier_model.split_data import split_train_test_data

logging.basicConfig(
    filename=LOGS_DIR / "training_pipeline.log", filemode="w", level=logging.DEBUG
)

logger = logging.getLogger("training_pipeline")


def log_performance(*, pipeline: Pipeline) -> None:

    test = load_dataset(file_name=config.app_config.test_data)

    pred = pipeline.predict(test[config.model_config.features])
    acc = round(accuracy_score(test[config.model_config.target], pred), 3)

    print(f"Model's accuracy: {acc}")


def run_training() -> None:
    """
    Train the model
    """

    logger.info("Running training function")

    split_train_test_data()

    logger.info("Dataset loaded")
    train = load_dataset(file_name=config.app_config.train_data)

    # improve the code to support parameter tunning
    spaceship_titanic_pipeline.fit(
        train[config.model_config.features], train[config.model_config.target]
    )
    logger.info("Model Trained")

    log_performance(pipeline=spaceship_titanic_pipeline)

    save_pipeline(pipeline_to_persist=spaceship_titanic_pipeline)
    logger.info("Pipeline Saved")
    logger.info("Training completed!")


if __name__ == "__main__":
    run_training()
