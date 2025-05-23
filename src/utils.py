import os
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomExeption
from logger import logging

import dill
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import r2_score , confusion_matrix , classification_report

def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path , "wb") as file_obj:
            dill.dump(obj,file_obj)

        logging.info(f"object saved succesfully to {file_path}")  
    except Exception as e:
        logging.info("Failed to save the object")
        raise CustomExeption(e,sys)
    

def evaluate_models(X_train , y_train , X_test, y_test , models:dict) -> dict:
    try:
        logging.info("started sequentially training the models")
        scores = dict()
        for model_name , model in models.items():
            model.fit(X_train , y_train)

            y_pred = model.predict(X_test)

            r2s = r2_score(y_pred=y_pred , y_true=y_test)
            scores[model_name] = r2s

        logging.info("models trained and evaluated")
        return scores

    except Exception as e:
        logging.info('failed to train and evaluate the models')
        raise CustomExeption(e,sys)