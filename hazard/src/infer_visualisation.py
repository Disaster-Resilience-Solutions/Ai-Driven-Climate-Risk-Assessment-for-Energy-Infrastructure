# Wind Forecast Visualization Utilities

# This module provides functions to visualize wind forecast data at specific lead times.
# It includes a function to create a DataFrame for a given lead time and two-dimensional scatter plots
# to visualize the U-component and V-component wind intensities.

# Functions:
#     - visualize_lead_time: Extract and structure wind forecast data for a specific lead time.
#     - plot_wind_intensity: Plot wind intensity for a specific lead time and wind component.
#     - visualize_and_save_results: Saves intensity plots of wind intensity value components.

# Usage:
#     1. Import the necessary modules: matplotlib.pyplot and pandas.
#     2. Use visualize_lead_time to obtain a DataFrame for the desired lead time.
#     3. Call plot_wind_intensity to visualize wind intensities on scatter plots.
#     4. Call visualize_and_save_results to export sufficiently.

# Example:
#     # Visualize wind forecast data
#     fc_iter_df = visualize_lead_time(fc_iter, lead_time_to_visualize)

#     # Plot U-component and V-component wind intensities
#     plot_wind_intensity(fc_iter_df, 'U-component-wind')
#     plot_wind_intensity(fc_iter_df, 'V-component-wind')

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

def visualize_lead_time(fc_iter, lead_time_to_visualize):
    """
    Visualize Wind Forecast Data at a Specific Lead Time

    Args:
        fc_iter (xarray.Dataset): Iterative wind forecast dataset.
        lead_time_to_visualize (int): Lead time to visualize.

    Returns:
        pd.DataFrame: DataFrame containing the data for the specified lead time.
    """
    fc_iter_at_lead_time = fc_iter.sel(lead_time=lead_time_to_visualize)

    fc_iter_df = fc_iter_at_lead_time.stack(points=('latitude', 'longitude')).to_dataframe()
    fc_iter_df['coordinates'] = list(zip(fc_iter_df['latitude'], fc_iter_df['longitude']))
    fc_iter_df.drop(['latitude', 'longitude'], axis=1, inplace=True)

    return fc_iter_df

def plot_wind_intensity(fc_iter_at_lead_time, variable, lead_time_to_visualize, save_path=None, overwrite=True):
    """
    Plot Wind Intensity

    Args:
        fc_iter_at_lead_time (DataFrame): Wind intensity data.
        variable (str): Variable name ('U-component-wind' or 'V-component-wind').
        lead_time_to_visualize (int): Lead time for visualization.
        save_path (str, optional): Path to save the plot.
        overwrite (bool, optional): Whether to overwrite the file if it already exists.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))

    # Scatter plot of wind intensity
    sc = plt.scatter(fc_iter_at_lead_time['latitude'], fc_iter_at_lead_time['longitude'],
                     c=fc_iter_at_lead_time[variable], cmap='viridis', marker='o', s=50)

    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title(f'{variable} Intensity (Lead Time = {lead_time_to_visualize})')
    plt.colorbar(sc, label=f'{variable} intensity')

    # Save the plot
    if save_path:
        if overwrite or not os.path.exists(save_path):
            plt.savefig(save_path)
        else:
            print(f"File '{save_path}' already exists. Skipping save.")
    else:
        plt.show()

def visualize_and_save_results(predictions, lead_time, out_dir):
    """
    Visualize and save wind intensity results.

    Args:
        predictions (xarray.Dataset): Iterative wind predictions dataset.
        lead_time (int): Lead time for visualization.
        out_dir (str): Output directory for saving plots.

    Returns:
        None
    """
    predictions_df = visualize_lead_time(predictions, lead_time)
    predictions_df['latitude'] = predictions_df['coordinates'].apply(lambda x: x[0])
    predictions_df['longitude'] = predictions_df['coordinates'].apply(lambda x: x[1])

    os.makedirs(out_dir, exist_ok=True)

    plot_wind_intensity(predictions_df, 'U-component-wind', lead_time, save_path=os.path.join(out_dir, 'wind_intensity_U.png'))
    plot_wind_intensity(predictions_df, 'V-component-wind', lead_time, save_path=os.path.join(out_dir, 'wind_intensity_V.png'))
