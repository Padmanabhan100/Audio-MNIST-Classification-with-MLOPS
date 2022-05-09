from src.utils.all_utils import read_yaml, create_directory
from tensorflow.keras.models import load_model
import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Create a logging pattern
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
# name of the logging directory
log_dir = "logs"
# Create logging file
os.makedirs(log_dir, exist_ok=True)
# Configure the logging file
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"), 
                    level=logging.INFO, format=logging_str, filemode='a')


def metrics(config_path):
    # Load the config file & params file
    config = read_yaml(config_path)

    # Load the trained model
    #path = "D:/DATA SCIENCE/Kaggle Datasets/MNIST AUDIO/archive/MNIST Audio Classifier.hdf5"
    model = load_model("artifacts\checkpoints")
    #model = load_model(path)

    # Reading Train & Test Path From configuration file
    train_path,test_path = os.path.join("artifacts",config['local_data_dir'][1]), os.path.join("artifacts",config['local_data_dir'][2])

    # Load the training and testing data
    X_test,Y_test = np.load(f"{test_path}/X_test.npy"),np.load(f"{test_path}/Y_test.npy")
    
    # Making Predictions on Test Set
    Y_pred = [np.argmax(x) for x in model.predict(X_test)]

    # Confusion Matrix
    cm = confusion_matrix(Y_test,Y_pred)
    logging.info(cm)

    # Plot the confusion matrix
    sns.set_style("dark")
    plt.figure(figsize=(12,8))
    plt.title("CONFUSION MATRIX AUDIO MNIST")
    sns.heatmap(cm,cbar=False,annot=True,cmap='cool',fmt='g')
    plt.xlabel("ACTUAL")
    plt.ylabel("PREDICTED")
    plt.savefig("confusion_matrix.png")

    # Log the activity
    logging.info("Confusion Matrix Saved Successfully")


if __name__ == "__main__":
    # Create a argument parser object
    args = argparse.ArgumentParser()

    # Add a argument
    args.add_argument("--config",'-c',default='config/config.yaml')


    # Call the parse_args method
    parsed_args = args.parse_args()

    try:
        logging.info("============== Stage 04: Model Metrics Generation (Initiated) ==========================")
        # Get the data 
        metrics(config_path=parsed_args.config)
        # Log the status of the first Stage
        logging.info("============== Stage 04: Model Metrics Generation (SUCCESSFUL) =========================\n\n")
    except Exception as e:
        logging.exception(e)
        raise e

