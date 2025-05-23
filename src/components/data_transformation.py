import os
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomExeption
from logger import logging
from utils import save_object

import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function is responsible for creating data transformer obj
        
        '''
        try:
            numerical_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
            categorical_columns = ['Gender', 'Subscription Type', 'Contract Length']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )

            logging.info("numerical columns transformation completed")
            logging.info("categorical columns transformation completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_columns),
                    ("categorical_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging("Failed to create the preprocessor")
            raise CustomExeption(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        '''
            This function is responsible for applying the transformer
        '''

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("reading train and test data completed")

            logging.info("obtaining preprocessing object")
            
            preprocessor_obj = self.get_data_transformer_object()

            target_column = "Churn"

            # numerical_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
            # categorical_columns = ['Gender', 'Subscription Type', 'Contract Length']

            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            
            target_feature_train_df = train_df[target_column]
            target_feature_test_df = test_df[target_column]

            logging.info("applying preprocessing on the training and the testing dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)



            train_arr = np.c_[
                input_feature_train_arr , np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test_df)
            ]

            logging.info("saving preprocessing object...")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomExeption(e,sys)