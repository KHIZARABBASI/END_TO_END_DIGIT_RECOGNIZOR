from cnnDigitReco.config.configuration import ConfigurationManager
from cnnDigitReco.components.data_ingestion import DataIngestion
from cnnDigitReco import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.save_zip()
        data_ingestion.extract_zip_file()