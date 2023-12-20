# Wind Forecasting Training Script

# This script trains a Convolutional Neural Network (CNN) for wind forecasting using GRIB data.

# Script Execution:
#     Execute this script to train the wind forecasting model.

# Usage:
#     $ python train.py

# Requirements:
#     - PyGrib (pygrib)
#     - TensorFlow (tensorflow)
#     - Keras (tensorflow.keras)
#     - NumPy (numpy)
#     - Xarray (xarray)

# Note: Make sure to configure the necessary parameters in the train_config.py file before executing the script.

import os
import pygrib
import warnings
import logging
import tensorflow as tf
import tensorflow.keras as keras

from train_config import (
    MAX_EPOCHS, 
    GRIB_FILE, 
    VARIABLE_LEVELS_DICT, 
    LEAD_TIME, 
    BATCH_SIZE, 
    MODEL_DIR
)
from data_transform import process_grib_data, split_dataset, create_data_generators
from model import build_cnn

logging.basicConfig(level=logging.INFO)

def limit_mem():
    """
    Limit GPU Memory Growth

    This function configures TensorFlow to allow GPU memory growth.

    Returns:
        None
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

def train_model():
    """
    Train Wind Forecasting Model

    This function performs the training of the wind forecasting model.

    Returns:
        None
    """
    limit_mem()
    logging.info(tf.test.gpu_device_name())

    if not os.path.exists(MODEL_DIR):
        logging.info(f"Creating directory: {MODEL_DIR}")
        os.makedirs(MODEL_DIR)
    else:
        logging.warning(f"Model directory already exists: {MODEL_DIR}")

    grbs = pygrib.open(GRIB_FILE)
    ds = process_grib_data(grbs)
    ds_train, ds_val, ds_test = split_dataset(ds)
    dg_train, dg_valid, dg_test = create_data_generators(ds_train, ds_val, ds_test, VARIABLE_LEVELS_DICT, LEAD_TIME, BATCH_SIZE)

    cnn = build_cnn([64, 64, 64, 64, 2], [5, 5, 5, 5, 5], (44, 42, 2))

    if "arm64" in tf.config.list_physical_devices():
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    warnings.filterwarnings("ignore", category=Warning, module="tensorflow")

    cnn.compile(optimizer, 'mse')

    cnn.fit(dg_train,
            epochs=MAX_EPOCHS,
            validation_data=dg_valid,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=0,
                                                        patience=2,
                                                        verbose=1,
                                                        mode='auto')])

    model_save_file = os.path.join(MODEL_DIR, 'cnn.h5')

    if os.path.exists(model_save_file):
        logging.warning(f"Overwriting existing model: {model_save_file}")

    keras.models.save_model(cnn, model_save_file)
    logging.info(f"Model saved: {model_save_file}")

if __name__ == '__main__':
    train_model()
