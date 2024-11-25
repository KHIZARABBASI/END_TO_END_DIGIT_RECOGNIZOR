from cnnDigitReco.config.configuration import ConfigurationManager
from cnnDigitReco.components.model_traning import Training
from cnnDigitReco import logger
from cnnDigitReco.config.configuration import DataTransformationConfig


STAGE_NAME = "Data Transformation stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # config = ConfigurationManager()
        # training_config = config.get_training_config()
        # training = Training(config=training_config)
        # training.train(DataTransformationConfig)
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        training_config = config.get_training_config()

        training = Training(training_config)
        training.train(data_transformation_config)