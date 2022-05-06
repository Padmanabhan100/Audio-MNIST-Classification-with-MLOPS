from src.utils.all_utils import read_yaml, create_directory
import argparse
import pandas as pd
import os
from tqdm import tqdm
import logging
import numpy as np
import librosa

# *********************** Logging Setup ********************************
# Crete a logging pattern
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
# name of the logging directory
log_dir = "logs"
# Create logging file
os.makedirs(log_dir, exist_ok=True)
# Configure the logging file
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"), 
                    level=logging.INFO, format=logging_str, filemode='a')

# ************************ Loading the data *****************************

def get_data(config_path):
    # Read the Yaml file
    config = read_yaml(config_path)
    # Load the data from the path in config file
    data_list = []
    # Loop through each folder(A speaker)
    source_data_dir = config['source_data_dir'][0]
    print(source_data_dir[0])
    for folder in tqdm(os.listdir(source_data_dir)[:1],colour='green'):
        # Loop through each file(Speaker's audio) and load the data
        for file in tqdm(os.listdir(os.path.join(source_data_dir,folder)),colour='blue'):
            # Load the data
            data_path = source_data_dir + folder + '/' + file
            data = librosa.load(data_path)
            # Fetch the class of the audio from the name of the file
            label = file.split("_")[0]
            # Append the data and its class in the data_list
            data_list.append([data,label])

    # Log the current activity
    logging.info("All files with labels loaded successfully")

    # Convert to a dataframe
    df = pd.DataFrame(data_list,columns=['audio','class'])
    # Save it as a csv file
    dir_to_save, file_name = os.path.join('artifacts',config['local_data_dir'][0]),config['local_data_dir'][1]
    create_directory([dir_to_save])
    # Write the data in the folder
    np.save(dir_to_save+"X.npy",df['audio'].values)
    np.save(dir_to_save+"Y.npy",df['class'].values)

    # Log the current activity
    logging.info("Raw Data Saved Successfully")
    

if __name__ == "__main__":
    # Create a argument parser object
    args = argparse.ArgumentParser()

    # Add a argument
    args.add_argument("--config",'-c',default='config/config.yaml')

    # Call the parse_args method
    parsed_args = args.parse_args()

    try:
        logging.info("============== Stage 01: Load and Save (Initiated) ==========================")
        # Get the data 
        get_data(config_path=parsed_args.config)
        # Log the status of the first Stage
        logging.info("============== Stage 01: Load and Save (SUCCESSFUL) ==========================\n\n")
    except Exception as e:
        logging.exception(e)
        raise e

