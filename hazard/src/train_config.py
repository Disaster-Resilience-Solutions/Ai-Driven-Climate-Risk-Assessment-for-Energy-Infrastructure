# Configuration File

# This file contains configuration parameters for wind speed component forecasting, including file paths,
# model training settings, and variable information.

# Attributes:
#     FILE_NAME (str): The name of the GRIB file.
#     BASE_DIR (str): The base directory for storing data and model files.
#     DATA_DIR (str): The directory for storing data files.
#     MODEL_DIR (str): The directory for storing trained model files.
#     GRIB_FILE (str): The full path to the GRIB data file.
#     MAX_EPOCHS (int): The maximum number of epochs for model training.
#     BATCH_SIZE (int): The batch size used during training and data generation.
#     LEAD_TIME (int): The lead time for forecasting.
#     MAX_LEAD_TIME (int): The maximum lead time allowed.

#     VARIABLE_LEVELS_DICT (dict): Dictionary specifying variable names and associated levels.

# Usage:
#     - Update FILE_NAME to the desired GRIB file name.
#     - Specify the desired BASE_DIR for storing data and model files.
#     - Adjust other parameters such as MAX_EPOCHS, BATCH_SIZE, LEAD_TIME, etc., as needed.
#     - Ensure that the GRIB_FILE path is correctly constructed based on DATA_DIR and FILE_NAME.
#     - Modify VARIABLE_LEVELS_DICT to include the relevant variable names and associated levels.

import os

FILE_NAME = "sample.grib"

BASE_DIR = os.path.join(os.path.expanduser("~"), "Desktop/wind_forecast_update")

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUT_DIR = os.path.join(BASE_DIR, "output")

GRIB_FILE = os.path.join(DATA_DIR, FILE_NAME)

MAX_EPOCHS = 100
BATCH_SIZE = 32
LEAD_TIME = 6
MAX_LEAD_TIME = 5 * 24

VARIABLE_LEVELS_DICT = {'U-component-wind': None,
                        'V-component-wind': None}
