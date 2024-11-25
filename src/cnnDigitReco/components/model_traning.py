from cnnDigitReco.config.configuration import DataTransformationConfig
from cnnDigitReco.config.configuration import TrainingConfig
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config



    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    

    def train(self,data_config: DataTransformationConfig):
        self.data_config = data_config
    
        # Load the dataset
        x_train = pd.read_csv(f"{self.data_config.training_dir}/x_train.csv")
        y_train = pd.read_csv(f"{self.data_config.training_dir}/y_train.csv")
        x_test = pd.read_csv(f"{self.data_config.valid_dir}/x_val.csv")
        y_test = pd.read_csv(f"{self.data_config.valid_dir}/y_val.csv")

        # Reshape the image data (flattened) back to 28x28x1
        x_train = x_train.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # Normalize pixel values
        x_test = x_test.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        # Labels do not need reshaping
        y_train = y_train.values.flatten()  # Ensure it's a 1D array
        y_test = y_test.values.flatten()


        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),  # Drop 25% of neurons
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        

        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)


        self.save_model(path=self.config.trained_model_path, model=model)