import pandas as pd
import os
import sys
import numpy as np
import logging
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifect","train.csv")
    test_data_path: str=os.path.join("artifect","test.csv")
    raw_data_path: str=os.path.join("artifect","raw.csv")

class DataIngestion:
    def __init__(self):
        self.Ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'C:\Users\rgarlay\Desktop\DS\ML OPS\Project\project\src\archive\StudentsPerformance.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.Ingestion_config.raw_data_path,index=False,header=True)
                
            logging.info("train test split initaiated")
            train_set,test_set = train_test_split(df,test_size=42,random_state=32)

            train_set.to_csv(self.Ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.Ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
            
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

data_transformation=DataTransformation()
data_transformation.initiate_data_transformation(train_data, test_data)