import os
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomExeption
from logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from data_transformation import DataTransformation, DataTransformationConfig
from model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('data','cleaned',"train.csv")
    test_data_path:str = os.path.join('data','cleaned',"test.csv")
    raw_data_path:str = os.path.join('data','raw',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            _ = pd.read_csv(r"data\raw\customer_churn_dataset-training-master.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df = _.dropna()
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split initiated")

            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42) 
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("data ingestion completed succesfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("data ingestion failed")
            raise CustomExeption(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path , test_data_path = obj.initiate_data_ingestion()

    data_transformer_obj = DataTransformation()
    _ = data_transformer_obj.initiate_data_transformation(train_data_path=train_data_path,test_data_path=test_data_path)
    train_arr , test_arr , preprocessor_path = _

    model_trainer_obj = ModelTrainer()
    r2_score = model_trainer_obj.initiate_model_training(train_arr=train_arr , test_arr=test_arr)
    print(f"final score of the best model: {r2_score}")
    
    
    