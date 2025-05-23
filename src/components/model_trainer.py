import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomExeption
from logger import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import r2_score , confusion_matrix , classification_report

import numpy as np

from utils import save_object , evaluate_models

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("models" , "model.pkl")    

class ModelTrainer:
    def __init__(self):
        self.model_training_config = ModelTrainerConfig()
    
    def get_models(self):
        try:
            logging.info('getting the models')
            RFC = RandomForestClassifier(n_estimators=100 , random_state=42)
            LR = LogisticRegression()
            KNN = KNeighborsClassifier(n_neighbors=5)
            XGB = XGBClassifier()

            models_list = {
                'random_forest_classifier' : RFC ,
                'logistic_regression' : LR,
                'k_nearest_neighbors' : KNN,
                'xgb_classifier' : XGB
                }
            
            logging.info('models returned')
            return models_list
        except Exception as e:
            logging.info('failed to get the models')
            raise CustomExeption(e,sys)

    
    def initiate_model_training(self , train_arr , test_arr):
        try:

            logging.info("splitting the training and testing data")

            X_train , y_train , X_test , y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = self.get_models()

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_info = max(model_report.items(), key=lambda item:item[1])

            best_model_name , best_model_score = best_model_info 

            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomExeption("No best model found" , sys)
            
            logging.info(f"best model on the provided dataset: {best_model_name} with r2_score: {best_model_score}")
            save_object(file_path=self.model_training_config.trained_model_path , obj=best_model)

            predicted = best_model.predict(X_test)
            r2_s = r2_score(y_pred=predicted , y_true=y_test)
            return r2_s
        
        except Exception as e:
            logging.info('failed to train the models')
            raise CustomExeption(e,sys)
        
    

        


        


             




