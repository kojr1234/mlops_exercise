from sklearn.model_selection import train_test_split

from classifier_model.config.core import config
from classifier_model.pipeline import spaceship_titanic_pipeline
from classifier_model.preprocessing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    """
    Train the model
    """

    data = load_dataset(filename = config.app_config.train_data)

    X_, X, y_, y = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state
    )

    # improve the code to support parameter tunning
    spaceship_titanic_pipeline.fit(X_, y_)

    save_pipeline(pipeline_to_persist=spaceship_titanic_pipeline)

if __name__ == '__main__':
    run_training()