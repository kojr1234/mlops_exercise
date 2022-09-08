import pytest
from sklearn.model_selection import train_test_split

from classifier_model.config.core import config
from classifier_model.preprocessing.data_manager import load_dataset


@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name=config.app_config.test_data)
    X = data[config.model_config.features]
    y = data[config.model_config.target]

    return X, y
