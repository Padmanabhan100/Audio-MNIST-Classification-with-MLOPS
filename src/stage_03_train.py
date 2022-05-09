from src.utils.all_utils import read_yaml, create_directory
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import os
import numpy as np
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


def train(config_path,params_path):
    # Load the config file & params file
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    # Reading Train & Test Path From configuration file
    train_path,test_path = os.path.join("artifacts",config['local_data_dir'][1]), os.path.join("artifacts",config['local_data_dir'][2])
    checkpoint_dir_path = os.path.join('artifacts',config['artifacts']['CHECKPOINT_DIR'])
    trained_model_dir, trained_model_name = os.path.join("artifacts",config['artifacts']['TRAINED_MODEL_DIR']), config['artifacts']['TRAINED_MODEL_NAME']

    # Reading Model Parameters
    n_layer1,n_layer2,n_layer3 = int(params['model']['neurons_layer1']),int(params['model']['neurons_layer2']),int(params['model']['neurons_layer3'])
    optimizer,n_epochs,batch_size = params['training']['optimizer'], params['training']['n_epochs'],params['training']['batch_size']
    
    # Creating Model
    model = Sequential()
    # 1st Layer
    model.add(Dense(n_layer1,input_shape=(40,), activation='relu'))
    # 2nd Layer
    model.add(Dense(n_layer2, activation='relu'))
    # 3rd Layer
    model.add(Dense(n_layer3, activation='relu'))
    # Output Layers
    model.add(Dense(10, activation='softmax'))

    # Log the activity
    logging.info(f"Model Creation Successful\n{model.summary()}")
    
    # Compiling Model
    model.compile(loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'],
                   optimizer=optimizer)

    # Log the activity
    logging.info("Model Successfully Compiled")

    # Read the Training & Testing Data
    X_train,X_test,Y_train,Y_test = np.load(f"{train_path}/X_train.npy"),np.load(f"{test_path}/X_test.npy"),np.load(f"{train_path}/Y_train.npy"),np.load(f"{test_path}/Y_test.npy")

    # Create Model checkpoints
    checkpoint = ModelCheckpoint(checkpoint_dir_path,verbose=1, save_best_only=True)

    # Fit the model
    model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=batch_size,epochs=n_epochs,callbacks=[checkpoint])

    # Log the activity
    logging.info("Model Trained Successful")

    # Create dir to save the model
    create_directory([trained_model_dir])

    # Save Model
    model.save(trained_model_dir)

    # Log the activity
    logging.info("Trained Model Saved Successfully")

if __name__ == "__main__":
    # Create a argument parser object
    args = argparse.ArgumentParser()

    # Add a argument
    args.add_argument("--config",'-c',default='config/config.yaml')
    args.add_argument("--params",'-p',default='params.yaml')


    # Call the parse_args method
    parsed_args = args.parse_args()

    try:
        logging.info("============== Stage 03: Model Creation & Training (Initiated) ==========================")
        # Get the data 
        train(config_path=parsed_args.config,params_path=parsed_args.params)
        # Log the status of the first Stage
        logging.info("============== Stage 03: Model Creation & Training (SUCCESSFUL) =========================\n\n")
    except Exception as e:
        logging.exception(e)
        raise e

