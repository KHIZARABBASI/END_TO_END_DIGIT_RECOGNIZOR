from cnnDigitReco.constants import *
from cnnDigitReco.utils.common import read_yaml, create_directories
from cnnDigitReco.entity.config_entity import (DataIngestionConfig,
                                               DataTransformationConfig,
                                               TrainingConfig)
from cnnDigitReco import logger

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        try:
            data_ingestion_config = DataIngestionConfig(
                root_dir = config.root_dir,
                local_data_file = config.local_data_file,
                unzip_dir = config.unzip_dir,
                source_dir = config.source_dir
            )

            return data_ingestion_config
        except Exception as e:
            logger.info(e)
            raise e
        

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])



        return DataTransformationConfig(
            root_dir = Path(config.root_dir),
            input_data_file = Path(config.input_data_file),
            output_data_file = Path(config.output_data_file),
            training_dir = Path(config.training_dir),
            valid_dir = Path(config.valid_dir)
        )



    def get_training_config(self) -> TrainingConfig:
        config = self.config.model_traning


        create_directories([config.root_dir])
        return TrainingConfig(
            root_dir = Path(config.root_dir),
            trained_model_path = Path(config.trained_model_path)
        )