# Wind Forecasting Data Processing and Generator Utilities

# This module provides utility functions for processing meteorological data from GRIB files and creating data generators
# for training and evaluating wind forecasting models.

# Classes:
#     - DataGenerator: A Keras Sequence class for creating data generators that yield batches of input-output pairs.

# Functions:
#     - process_grib_data: Converts GRIB data containing U and V wind components into an xarray Dataset.
#     - split_dataset: Splits a dataset into training, validation, and test sets based on specified percentages.
#     - create_data_generators: Creates data generators for training, validation, and testing.

# Usage:
#     1. Import the necessary modules: TensorFlow, Keras, numpy, and xarray.
#     2. Use the DataGenerator class to create data generators for training and evaluation.
#     3. Utilize the process_grib_data function to convert GRIB data into an xarray Dataset.
#     4. Split your dataset using the split_dataset function.
#     5. Call the create_data_generators function to obtain data generators for training, validation, and testing.

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import pygrib
import numpy as np
import xarray as xr


class DataGenerator(keras.utils.Sequence):
    """
    DataGenerator Class for Wind Forecasting Models

    This class extends the Keras Sequence class to create data generators that yield batches of input-output pairs
    for training and evaluating wind forecasting models.

    Args:
        ds (xarray.Dataset): Input dataset containing meteorological data.
        variable_levels_dict (dict): Dictionary specifying variable names and associated levels.
        lead_time (int): Lead time for forecasting.
        batch_size (int): Size of each batch.
        shuffle (bool): Whether to shuffle the data between epochs.
        load (bool): Whether to load the entire dataset into memory.
        mean (xarray.Dataset, optional): Mean values for normalization. If None, computed from the input dataset.
        std (xarray.Dataset, optional): Standard deviation values for normalization. If None, computed from the input dataset.

    Methods:
        - __len__: Returns the number of batches in the generator.
        - __getitem__: Generates a batch of input-output pairs.
        - on_epoch_end: Shuffles the data indices at the end of each epoch.

    Usage:
        # Create a data generator
        dg = DataGenerator(ds, variable_levels_dict, lead_time, batch_size, shuffle=True, load=True)
    """

    def __init__(self, ds, variable_levels_dict, lead_time, batch_size, shuffle=True, load=True, mean=None, std=None):
        self.ds = ds
        self.variable_levels_dict = variable_levels_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time

        data = []
        for var in variable_levels_dict:
            data.append(ds[var])

        self.data = xr.concat(data, 'variable').transpose('time', 'latitude', 'longitude', 'variable')
        self.mean = self.data.mean(('time', 'latitude', 'longitude')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('latitude', 'longitude')).compute() if std is None else std
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self.on_epoch_end()

        if load:
            self.data.load()

    def __len__(self):
        """
        Returns the number of batches in the generator.

        Returns:
            int: Number of batches.
        """
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        """
        Generates a batch of input-output pairs.

        Args:
            i (int): Index of the batch.

        Returns:
            tuple: Input-output pair for the batch.
        """
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=idxs).values
        y = self.data.isel(time=idxs + self.lead_time).values
        return X, y

    def on_epoch_end(self):
        """
        Shuffles the data indices at the end of each epoch.
        """
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)

def process_grib_data(grbs):
    """
    Process GRIB Data into xarray Dataset

    This function takes GRIB data containing U and V wind components and converts it into an xarray Dataset.

    Args:
        grbs (list): List of pygrib messages.

    Returns:
        xarray.Dataset: Processed dataset containing U and V wind components.
    """
    u_data = []
    v_data = []

    for grb in grbs:
        if "U wind component" in grb.parameterName:
            u_data.append(grb.values)
        elif "V wind component" in grb.parameterName:
            v_data.append(grb.values)

    u_data = np.array(u_data)
    v_data = np.array(v_data)
    time_values = range(len(u_data))

    ds = xr.Dataset({'U-component-wind': (['time', 'latitude', 'longitude'], u_data),
                     'V-component-wind': (['time', 'latitude', 'longitude'], v_data)},
                    coords={'time': time_values})
    return ds

def split_dataset(ds, train_percent=0.6, val_percent=0.2, test_percent=0.2):
    """
    Split Dataset into Training, Validation, and Test Sets

    This function splits a dataset into training, validation, and test sets based on specified percentages.

    Args:
        ds (xarray.Dataset): Input dataset.
        train_percent (float): Percentage of data for training.
        val_percent (float): Percentage of data for validation.
        test_percent (float): Percentage of data for testing.

    Returns:
        tuple: Training, validation, and test datasets.
    """
    total_samples = len(ds.time)

    train_size = int(train_percent * total_samples)
    val_size = int(val_percent * total_samples)
    test_size = total_samples - train_size - val_size

    ds_train = ds.isel(time=slice(0, train_size))
    ds_val = ds.isel(time=slice(train_size, train_size + val_size))
    ds_test = ds.isel(time=slice(train_size + val_size, None))

    return ds_train, ds_val, ds_test

def create_data_generators(ds_train, ds_val, ds_test, variable_levels_dict, lead_time, batch_size):
    """
    Create Data Generators for Training, Validation, and Testing

    This function creates data generators for training, validation, and testing.

    Args:
        ds_train (xarray.Dataset): Training dataset.
        ds_val (xarray.Dataset): Validation dataset.
        ds_test (xarray.Dataset): Test dataset.
        variable_levels_dict (dict): Dictionary specifying variable names and
            associated levels for normalization.
        lead_time (int): Lead time for forecasting.
        batch_size (int): Size of each batch.

    Returns:
        tuple: Data generators for training, validation, and testing.

    Usage:
        # Create data generators
        dg_train, dg_valid, dg_test = create_data_generators(ds_train, ds_val, ds_test, variable_levels_dict, lead_time, batch_size)
    """
    dg_train = DataGenerator(ds_train, variable_levels_dict, lead_time, batch_size, load=True)
    dg_valid = DataGenerator(ds_val, variable_levels_dict, lead_time, batch_size, mean=dg_train.mean, std=dg_train.std, shuffle=False)
    dg_test = DataGenerator(ds_test, variable_levels_dict, lead_time, batch_size, mean=dg_train.mean, std=dg_train.std, shuffle=False)

    return dg_train, dg_valid, dg_test
