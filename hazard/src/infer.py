# Wind Forecasting Inference Script

# Perform wind forecasting using a trained Convolutional Neural Network (CNN) model.

# Execute this script with: $ python infer.py

# Requirements:
# - NumPy (numpy)
# - Xarray (xarray)
# - TensorFlow (tensorflow)
# - Keras (tensorflow.keras)
# - PyGrib (pygrib)

# Note: Configure parameters in train_config.py before executing the script.

import os
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
import pygrib
import logging

from train_config import (
    GRIB_FILE,
    LEAD_TIME,
    MAX_LEAD_TIME,
    MODEL_DIR,
    OUT_DIR,
    VARIABLE_LEVELS_DICT,
    BATCH_SIZE
)
from data_transform import process_grib_data, split_dataset, create_data_generators
from infer_visualisation import plot_wind_intensity, visualize_lead_time, visualize_and_save_results
from model import PeriodicConv2D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_file):
    """
    Load a pre-trained CNN model from training export file.

    Args:
        model_file (str): Path to the saved model file.

    Returns:
        tf.keras.models.Model: Loaded CNN model.
    """
    logger.info(f"Loading the trained model from {model_file}")
    with keras.utils.custom_object_scope({'PeriodicConv2D': PeriodicConv2D}):
        model = keras.models.load_model(model_file)
    logger.info("Model loaded successfully.")
    return model

def generate_iterative_predictions(model, data_generator, max_lead_time):
    """
    Generate iterative wind predictions using the trained CNN model.

    Args:
        model (tf.keras.models.Model): Trained CNN model.
        dg (DataGenerator): Data generator for wind forecasting.
        max_lead_time (int): Maximum lead time for predictions.

    Returns:
        xarray.Dataset: Iterative wind predictions dataset.
    """
    logger.info("Generating iterative wind predictions...")
    
    state = data_generator.data[:data_generator.n_samples]
    preds = []

    for _ in range(max_lead_time // data_generator.lead_time):
        state = model.predict(state)
        p = state * data_generator.std.values + data_generator.mean.values
        preds.append(p)

    preds = np.array(preds)

    lead_time = np.arange(data_generator.lead_time, max_lead_time + data_generator.lead_time, data_generator.lead_time)
    das = []
    lev_idx = 0

    for var, levels in data_generator.variable_levels_dict.items():
        if levels is None:
            das.append(xr.DataArray(
                preds[:, 0, :, :, lev_idx],
                dims=['lead_time', 'latitude', 'longitude'],
                coords={'lead_time': lead_time, 'latitude': data_generator.ds.latitude, 'longitude': data_generator.ds.longitude},
                name=var
            ))
            lev_idx += 1
        else:
            nlevs = len(levels)
            das.append(xr.DataArray(
                preds[:, 0, :, :, lev_idx:lev_idx + nlevs],
                dims=['lead_time', 'latitude', 'longitude', 'level'],
                coords={'lead_time': lead_time, 'latitude': data_generator.ds.latitude, 'longitude': data_generator.ds.longitude, 'level': levels},
                name=var
            ))
            lev_idx += nlevs

    logger.info("Prediction generation completed.")
    return xr.merge(das)

def infer():
    """
    Inference for Wind Forecasting Model

    This function performs the inference stages of the wind forecasting model.

    Returns:
        None
    """
    logger.info("Starting wind forecasting inference...")

    model_save_file = os.path.join(MODEL_DIR, 'cnn.h5')
    model = load_trained_model(model_save_file)

    grbs = pygrib.open(GRIB_FILE)
    ds = process_grib_data(grbs)
    ds_train, ds_val, ds_test = split_dataset(ds)
    dg_train, dg_valid, dg_test = create_data_generators(ds_train, ds_val, ds_test, VARIABLE_LEVELS_DICT, LEAD_TIME, BATCH_SIZE)

    predictions = generate_iterative_predictions(model, dg_test, MAX_LEAD_TIME)

    lead_time_to_visualize = LEAD_TIME
    visualize_and_save_results(predictions, lead_time_to_visualize, OUT_DIR)

    logger.info("Wind forecasting inference completed.")

if __name__ == '__main__':
    infer()
