from cnnDigitReco.config.configuration import ConfigurationManager
from cnnDigitReco.components.data_transformation import DataTransformation
from cnnDigitReco import logger


STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.split_data_file()
        data_transformation.train_test_split()