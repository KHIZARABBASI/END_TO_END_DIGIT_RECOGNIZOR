from sklearn.model_selection import train_test_split
from cnnDigitReco.config.configuration import DataTransformationConfig
from cnnDigitReco.utils.common import create_directories
import pandas as pd
import numpy as np



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def split_data_file(self):
        data = pd.read_csv(self.config.input_data_file)
        labels = data["label"]
        pictures = data.drop("label", axis=1)

        output_dir = self.config.output_data_file

        # pictures = pictures.values.reshape(-1, 28, 28, 1)

        # ##
        # pd.DataFrame(pictures)
        ##

        labels.to_csv(output_dir / "labels.csv", index=False)
        pictures.to_csv(output_dir / "pictures.csv", index=False)

    def train_test_split(self):
        picture_file = pd.read_csv(self.config.output_data_file / "pictures.csv")
        labels_file = pd.read_csv(self.config.output_data_file / "labels.csv")


        x_train, x_val, y_train, y_val = train_test_split(
            picture_file, labels_file, test_size=0.2, random_state=42
        )

        # Create training and validation directories
        create_directories([self.config.training_dir, self.config.valid_dir])

        x_train.to_csv(self.config.training_dir / "x_train.csv", index=False)
        y_train.to_csv(self.config.training_dir / "y_train.csv", index=False)
        x_val.to_csv(self.config.valid_dir / "x_val.csv", index=False)
        y_val.to_csv(self.config.valid_dir / "y_val.csv", index=False)