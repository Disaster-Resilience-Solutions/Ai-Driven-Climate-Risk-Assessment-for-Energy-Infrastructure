# API Request for extracting time domain values of wind speed U and V components

# Considered with UK specific coordinates only - alternatively use pre-uploaded GRIB dataset

import cdsapi
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def request_wind_data(url, key, output_path):
    """
    Retrieve reanalysis wind data from the Copernicus Climate Data Store (CDS).

    Args:
        url (str): The URL of the CDS API.
        key (str): The user's API key for authentication.
        output_path (str): The path where the downloaded data will be saved.

    Returns:
        None
    """
    logger.info("Requesting wind data from the Copernicus Climate Data Store...")

    c = cdsapi.Client(url=url, key=key)

    request_params = {
        'product_type': 'reanalysis',
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
        'year': ['2022', '2023'],
        'month': [f'{month:02d}' for month in range(1, 13)],
        'day': [f'{day:02d}' for day in range(1, 32)],
        'time': [f'{hour:02d}:00' for hour in range(24)],
        'area': [60.86, -8.61, 49.96, 1.77],
        'output': output_path
    }

    c.retrieve('reanalysis-era5-single-levels', request_params)

    logger.info(f"Wind data downloaded and saved at: {output_path}")

if __name__ == '__main__':

    cds_url = "INSERT_HERE"  # Replace with the actual CDS API URL
    cds_key = "INSERT_HERE"  # Replace with the actual user's API key
    output_file_path = "SPECIFY_OUTPUT_PATH"  # Replace with the desired output file path

    request_wind_data(cds_url, cds_key, output_file_path)
