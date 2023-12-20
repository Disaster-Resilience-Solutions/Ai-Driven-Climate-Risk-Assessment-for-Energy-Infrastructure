# Convolutional Neural Network (CNN) for Wind Speed Forecasting

This repository contains scripts and files for wind speed forecasting using a Convolutional Neural Network (CNN). The CNN model is trained on GRIB data and can generate iterative wind speed predictions.

## CNN Model

The CNN model architecture is defined in the [`model.py`](model.py) file. The model uses a series of convolutional layers with periodic convolutions to capture spatial dependencies in the wind data.

## Training

The training script ([`train.py`](train.py)) processes GRIB data, splits it into training, validation, and test sets, and trains the CNN model. The trained model is then saved to the specified output directory (`MODEL_DIR` in [`train_config.py`](train_config.py)).

### Configuration

Before training, configure the necessary parameters in [`train_config.py`](train_config.py), such as the path to the GRIB file (`GRIB_FILE`), lead time (`LEAD_TIME`), and batch size (`BATCH_SIZE`).

## Inference

The inference script ([`infer.py`](infer.py)) loads the trained model and performs wind forecasting on specified designated test data. The predictions are visualized and saved as intensity plots in the specified output directory (`OUT_DIR` in [`train_config.py`](train_config.py)).

## Directory Structure

- [`data_transform.py`](data_transform.py): Contains functions for processing and transforming GRIB data.
- [`infer.py`](infer.py): Script for wind forecasting inference using the trained CNN model.
- [`infer_visualisation.py`](infer_visualisation.py): Visualization functions for wind speed intensity.
- [`model.py`](model.py): Defines the CNN model architecture.
- [`train.py`](train.py): Script for training the CNN model on GRIB data.
- [`train_config.py`](train_config.py): Configuration file for training parameters.

## Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Execute Containerised:**
   ```bash
   CMD ["sh", "-c", "python train.py && python infer.py"]
 
