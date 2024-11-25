from cnnDigitReco.entity.config_entity import DataIngestionConfig
from cnnDigitReco import logger
import shutil
import os
import zipfile


class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config = config
        
    
    
    def save_zip(self):
        # Implement the logic to save the zip file locally
        if not os.path.exists(self.config.local_data_file):
            source_file = self.config.source_dir
            destination_file = self.config.local_data_file
            shutil.copyfile(source_file, destination_file)
            logger.info(f"Zip file saved to {destination_file}")

        else:
            logger.info(f"Zip file already exists at {self.config.local_data_file}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

