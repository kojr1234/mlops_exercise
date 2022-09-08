from classifier_model.config.core import config, DATASET_DIR
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test_data() -> None:
    full_dataset_path = DATASET_DIR/config.app_config.full_data

    if full_dataset_path.is_file():
        train_full_dataset = pd.read_csv(full_dataset_path)
        X, X_, y, y_ = train_test_split(
            train_full_dataset[config.model_config.features],
            train_full_dataset[config.model_config.target],
            random_state=config.model_config.random_state,
            test_size=config.model_config.test_size
        )

        train = pd.concat([X, y], axis=1)
        test = pd.concat([X_, y_], axis=1)

        train.to_csv(DATASET_DIR/config.app_config.train_data, index=False)
        test.to_csv(DATASET_DIR/config.app_config.test_data, index=False)

        full_dataset_path.unlink()
