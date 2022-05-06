import yaml
import logging
import os
import time

def read_yaml(path_to_yaml):
    with open(path_to_yaml) as yaml_file:
        # Load the contents of the yaml file 
        content = yaml.safe_load(yaml_file)
    # Log the current activity
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    # Return the contents of the yaml_file
    return content

def create_directory(dirs:list):
    # iterate through the list of directory_path
    for dir_path in dirs:
        # Create the directory
        os.makedirs(dir_path, exist_ok=True)
        # Log ther current activity
        logging.info(f"directory is created at {dir_path}")

def get_timestamp(name):
    # Get the timestamp
    timestamp = time.asctime().replace(" ","_").replace(":","_")
    # Get unique name wrt timestamp
    unique_name = f"{name}_at_{timestamp}"
    # Return the unique name
    return unique_name