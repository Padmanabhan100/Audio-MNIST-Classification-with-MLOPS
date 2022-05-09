from src.utils.all_utils import read_yaml, create_directory
import argparse
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import logging
from sklearn.model_selection import train_test_split

def save_train_test_split(config_path):
    # Load the config file
    config = read_yaml(config_path)

    # Get the path of raw data dir to extract features
    clean_data_path = os.path.join("artifacts",config['local_data_dir'][0])

    # Load the raw data
    X = np.load(f"{clean_data_path}/X.npy")
    Y = np.load(f"{clean_data_path}/Y.npy")


    print("\n\nLoaded\n\n")
    # Perform Train-Test-Split 
    print(X.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8,random_state=12)
    logging.info("Train Test Split Successful")

    # Save the Data
    np.save(f"artifacts/{config['local_data_dir'][1]}/X_train.npy",X_train)
    np.save(f"artifacts/{config['local_data_dir'][1]}/Y_train.npy",Y_train)
    np.save(f"artifacts/{config['local_data_dir'][2]}/X_test.npy",X_test)
    np.save(f"artifacts/{config['local_data_dir'][2]}/Y_test.npy",Y_test)
    

    logging.info("Saved Train Test Split Data Successfully")
    

if __name__ == "__main__":
    # Create a argument parser object
    args = argparse.ArgumentParser()

    # Add a argument
    args.add_argument("--config",'-c',default='config/config.yaml')

    # Call the parse_args method
    parsed_args = args.parse_args()

    try:
        logging.info("============== Stage 02: Save Train Test Split (Initiated) ==========================")
        # Get the data 
        save_train_test_split(config_path=parsed_args.config)
        # Log the status of the first Stage
        logging.info("============== Stage 02: Save Train Test Split (SUCCESSFUL) =========================\n\n")
    except Exception as e:
        logging.exception(e)
        raise e

