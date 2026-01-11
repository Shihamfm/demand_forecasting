import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import data_preprocessing, train_test_split
import pandas as pd

from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","raw.csv")
    preprocessed_data_path: str=os.path.join("artifacts","preprocessed_data.csv")

class DataIngestioin:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:

            # Get the directory of the current file (data_ingestion.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Navigate up to the project root and then into the data folder
            # Adjust the "../.." based on how many levels deep your script is
            file_path = os.path.join(current_dir, '..', '..', 'notebook', 'data', 'walmart_daily_sales_2025_realistic.csv')

            df = pd.read_csv(file_path, parse_dates=['date'])
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            preprocesed_df = data_preprocessing(df)
            logging.info("Pre processed the raw dataset")

            preprocesed_df.to_csv(self.ingestion_config.preprocessed_data_path, index=False, header=True)

            train_df, test_df = train_test_split(preprocesed_df, 0.8)
            logging.info("Train and test spit completed")

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the raw data, preprocessed data, train data and test data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
         
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestioin()
    obj.initiate_data_ingestion()