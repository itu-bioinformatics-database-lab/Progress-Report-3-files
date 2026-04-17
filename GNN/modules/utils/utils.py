import os
import json
import dill
import pickle
import sys

from modules.exception import CustomException
from modules.logger import logging
        
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Saved Model object at '{file_path}'")
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

        logging.info(f"Loaded object from '{file_path}'")

    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e, sys)
    

def load_json(json_path):
    """
    Load json data as a dictionary
    """

    try:
        with open(json_path, encoding="utf-8") as f:
            file = f.read()
        json_data = json.loads(file)
        logging.info(f"Loaded JSON data from '{json_path}'")
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e, sys)

    return json_data


def save_json(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        json_object = json.dumps(obj, indent=2)

        with open(file_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(json_object)

        logging.info(f"Saved JSON data at '{file_path}'")

    except Exception as e:
        logging.info(CustomException(e, sys))
        raise CustomException(e, sys)