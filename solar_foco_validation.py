"""
File: solar_foco_validation.py

Description:
------------
This script performs analysis on solar panel model outputs and real-world solar data.
It loads and processes modeled data from previous simulations, compares it with real 
solar power data, and visualizes the results in a plot for easy comparison. The script 
also applies smoothing to real data using the Savitzky-Golay filter to reduce noise.

Functions:
---------
- load_solar_data_FoCO: Loads solar panel simulation results and converts them to a pandas DataFrame.
- load_real_data: Loads real-world solar data from a CSV file, processes it, and smooths the data.
- plot_solar: Plots the modeled and real solar power input data over time.

Dependencies:
-------------
- Python 3.x
- NumPy
- Pandas
- Matplotlib (for plotting)
- SciPy (for smoothing real data)
- run_solar_panel_foco

Author: Kyan Shlipak
Date: 09/28/24
"""
from run_solar_panel_foco import run_solar_panel_foco

# load model output data from saving directory
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
def load_solar_data_FoCO(results_dir, base_time  = datetime(2024, 6, 19, 9, 0, 0)):
    """
    Loads and processes solar panel simulation data from the specified directory.

    Parameters:
    -----------
    results_dir : str
        Directory containing the saved simulation results (NumPy compressed files).
    base_time : datetime, optional
        The start time for the simulation data, default is 2024-06-19 09:00:00.

    Returns:
    --------
    model_df : pandas.DataFrame
        A DataFrame containing the time-indexed simulation data, including solar 
        charge, power input, and power output resampled at one-minute intervals.
    """
    
    # Names of ndarrays to load (time, charge, power input, and power output)
    array_names = ["time_steps", "charges", "power_ins", "power_outs"]

    # Load each of the numpy arrays and store them in global variables
    for var in array_names:
        globals()[var] = np.load(results_dir + "/" + var + ".npz")['data']

    # Create a pandas DataFrame using the time_steps array
    model_df = pd.DataFrame({'seconds_past': time_steps})

    # Convert 'seconds_past' to a timedelta and add it to the base_time for datetime conversion
    model_df['datetime'] = model_df['seconds_past'].apply(lambda x: base_time + timedelta(seconds=x))
    
    # Assign charge, power input, and power output to DataFrame columns
    model_df['charge'] = charges
    model_df['power_in'] = power_ins
    model_df['power_out'] = power_outs

    # Set the datetime column as the index and resample the data to 1-minute intervals
    model_df.set_index('datetime', inplace=True)
    model_df = model_df.resample('min').mean()

    return model_df


def load_real_data(path = "../real_data/solar_data_2024_07_15.csv", volts = 20.5):
    """
    Loads and processes real solar data from a CSV file.

    Parameters:
    -----------
    path : str, optional
        File path of the CSV containing real solar data, default is '../real_data/solar_data_2024_07_15.csv'.
    volts : float, optional
        Voltage value to convert current (in mA) to power (in W), default is 20.5V.

    Returns:
    --------
    solar_df : pandas.DataFrame
        A DataFrame with real solar power input data, including smoothed data using the Savitzky-Golay filter.
    """
    
    import pandas as pd

    # Load the solar data CSV file into a DataFrame
    solar_df = pd.read_csv(path)

    # Set the 'Timestamp' column as the index and convert it to datetime
    solar_df = solar_df.set_index('Timestamp')
    solar_df.index = pd.to_datetime(solar_df.index)

    # Calculate the solar power in Watts by multiplying current (mA) by voltage (V)
    solar_df['solar_power'] = solar_df['Current (mA)'] * volts / 1000

    # Resample the data to one-minute intervals
    solar_df = solar_df.resample('min').mean()

    # Smooth the solar power data using Savitzky-Golay filter
    from scipy.signal import savgol_filter

    # Define the window length and polynomial order for the smoothing filter
    window_length = 10  # Filter window length (must be odd)
    polyorder = 2       # Polynomial order

    # Apply the Savitzky-Golay filter to smooth the 'solar_power' column
    solar_df['smoothed_solar'] = savgol_filter(solar_df['solar_power'], window_length, polyorder)

    return solar_df


def plot_solar(model_data, solar_data, start_date_time, end_date_time):
    """
    Plots the modeled solar data and real solar data on the same graph for comparison.

    Parameters:
    -----------
    model_data : pandas.Series or DataFrame
        Modeled solar power input data to be plotted.
    solar_data : pandas.Series or DataFrame
        Real solar power input data to be plotted.
    start_date_time : datetime
        The start date and time for plotting the data.
    end_date_time : datetime
        The end date and time for plotting the data.

    Returns:
    --------
    None
    """
    
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # Create a new figure and axis for the plot
    _, ax = plt.subplots(figsize=(12, 8))

    # Plot the modeled solar power data
    plt.plot(model_data, label="Modeled Power Input [W]")

    # Plot the real solar power data
    plt.plot(solar_data, label="Real Power Input [W]")

    # Limit the x-axis to 12 date labels for better readability
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))

    # Format the x-axis labels as month/day and hour/minute
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set the y-axis label to indicate the unit of measurement
    plt.ylabel('Charge in W*h')

    # Set the title of the plot
    plt.title("Battery Solar Charge Over Time")

    # Add a legend to differentiate between modeled and real data
    plt.legend()

    # Display the plot
    plt.show()

