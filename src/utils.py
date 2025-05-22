import os
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomExeption
from logger import logging

import dill
import numpy as np
import pandas as pd

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