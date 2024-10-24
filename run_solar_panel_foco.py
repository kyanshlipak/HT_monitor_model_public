"""
File: run_solar_panel_foco.py

Description:
------------
This script simulates the energy output and battery charging performance of a solar panel system 
based on geographic location, time period, and various panel and battery parameters. It allows 
customization of panel area, tilt, efficiency, battery specifications, and other parameters to model 
solar performance for any location on Earth. The simulation calculates power inputs and outputs, 
logs the process, and saves the results as compressed NumPy files.

Features:
---------
- Simulates solar panel output for specified locations and times.
- Models battery charging behavior based on solar power inputs.
- Supports logging of key events and data points.
- Saves results (time steps, charges, power inputs/outputs) in a directory for future analysis.
- Allows specifying custom shading periods to account for obstructed sunlight.

Usage:
------
This script can be run as part of a larger simulation or standalone to generate data for any 
given set of parameters. Modify parameters such as solar panel area, tilt, efficiency, and 
battery capacity to suit different scenarios. Additional configurations can be passed via 
keyword arguments.

Dependencies:
-------------
- Python 3.x
- NumPy
- Pandas (for data handling)
- Logging module (for event tracking)
- Custom utility functions for retrieving solar and weather data (get_data, get_timezone, etc.)

Author: Kyan Shlipak
Date: 09/28
"""

from datetime import datetime
from numerical_modelling import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def run_solar_panel_foco(solar_panel_area = 0.2, 
                        solar_panel_tilt = 20,
                        solar_panel_azimuth = 270,
                        solar_panel_efficiency = 0.18,
                        battery_rated_capacity = 120,
                        battery_efficiency = 0.8,
                        dt = 1,
                        latitude=40.5853,
                        longitude=-105.0844,                  
                        start_date_time='2024-06-28 12:00:00',
                        end_date_time='2024-07-01 13:00:00',
                        power_consumption = 5,
                        shading_ranges = [],
                        **kwargs):
    """
    run_solar_panel_foco: Simulates the solar panel system performance for a specified location and time range.

    This function models the energy production of a solar panel system in Fort Collins, Colorado, using
    local weather data for the specified date range. The model accounts for various parameters including
    solar panel area, tilt, azimuth, efficiency, battery capacity, and efficiency. The function also 
    considers shading effects and power consumption of the connected load. Simulation results are saved
    in a compressed format.

    Parameters:
        solar_panel_area (float): Surface area of the solar panel in square meters (default = 0.2).
        solar_panel_tilt (float): Tilt angle of the solar panel in degrees (default = 20).
        solar_panel_azimuth (float): Azimuth angle of the solar panel in degrees (default = 270).
        solar_panel_efficiency (float): Efficiency of the solar panel (default = 0.18).
        battery_rated_capacity (float): Rated capacity of the battery in Ah (default = 120).
        battery_efficiency (float): Efficiency of the battery (default = 0.8).
        dt (float): Time step for the simulation in hours (default = 1).
        latitude (float): Latitude of the location for the simulation (default = 40.5853).
        longitude (float): Longitude of the location for the simulation (default = -105.0844).
        start_date_time (str): Start date and time for the simulation in 'YYYY-MM-DD HH:MM:SS' format (default = '2024-06-28 12:00:00').
        end_date_time (str): End date and time for the simulation in 'YYYY-MM-DD HH:MM:SS' format (default = '2024-07-01 13:00:00').
        power_consumption (float): Constant power consumption of the connected load in watts (default = 5).
        shading_ranges (list): A list of shading periods for the simulation (default = []).
        **kwargs: Additional keyword arguments that override the default parameters.

    Returns:
        str: Path to the folder where simulation results are stored.
    """
    
	# any kwargs override default keyword arguemnt
    # Update local variables with any additional keyword arguments provided via **kwargs
    locals().update(kwargs)
    
    # Notify the user that the solar panel modeling process is starting
    print("*** Modelling Solar in Fort Collins ***")
    
    # Convert string start and end times into datetime objects
    start_date_time = datetime.strptime(start_date_time, '%Y-%m-%d %H:%M:%S')
    end_date_time = datetime.strptime(end_date_time, '%Y-%m-%d %H:%M:%S')
    
    # Retrieve local weather data for the simulation period
    df = get_local_data(start_date_time.strftime("%Y-%m-%d"), end_date_time.strftime("%Y-%m-%d"))
    df = pd.read_csv("combined.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Set up logging for the simulation process
    import logging
    filename = 'validation.log'  # Log file name
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Reduce logging verbosity for the werkzeug logger
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # Function to return the constant power consumption
    def power_out_func(*args, **kwargs):
        return power_consumption
    
    # Print the shading ranges provided for the simulation
    print(shading_ranges)

    # Call the solar panel model function to run the simulation and collect outputs
    time_steps, charges, power_ins, power_outs = solar_panel_model(
        latitude, longitude, logging, start_date_time, end_date_time, df,
        solar_panel_area, solar_panel_tilt, solar_panel_azimuth, solar_panel_efficiency, 
        battery_rated_capacity, battery_efficiency, dt,
        power_out_func, shading_ranges=shading_ranges
    )
    
    # Define the directory path for saving the results
    import os
    dir = "../../model_results_numpy/"  # Parent directory for saving results
    folder_name = "Solar_FoCO" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Unique folder name based on current timestamp
    
    # Create the directory if it doesn't already exist
    os.makedirs(os.path.join(dir, folder_name), exist_ok=True)
    
    # Save the time steps of the simulation in a compressed NumPy file
    np.savez_compressed(os.path.join(dir, folder_name, "time_steps.npz"), data=time_steps.astype(np.float32))
    
    # Save the battery charge states in a compressed NumPy file
    np.savez_compressed(os.path.join(dir, folder_name, "charges.npz"), data=charges.astype(np.float32))
    
    # Save the power input from the solar panel in a compressed NumPy file
    np.savez_compressed(os.path.join(dir, folder_name, "power_ins.npz"), data=power_ins.astype(np.float32))
    
    # Save the power consumption (outputs) in a compressed NumPy file
    np.savez_compressed(os.path.join(dir, folder_name, "power_outs.npz"), data=power_outs.astype(np.float32))

    # Return the path to the folder where the simulation results are stored
    return os.path.join(dir, folder_name)

def run_solar_panel_anywhere(solar_panel_area=0.2, 
                        solar_panel_tilt=20,
                        solar_panel_azimuth=270,
                        solar_panel_efficiency=0.18,
                        battery_rated_capacity=120,
                        battery_efficiency=0.8,
                        dt=1,
                        latitude=40.5853,
                        longitude=-105.0844,                  
                        start_date_time='2024-06-28 12:00:00',
                        end_date_time='2024-07-01 13:00:00',
                        power_consumption=5,
                        logging=False,
                        stop_event=False,
                        shading_ranges=[],
                        **kwargs):
    """
    Simulates the energy output of a solar panel system at any location on Earth.
    
    Parameters:
    -----------
    solar_panel_area : float
        Area of the solar panel in square meters (default 0.2).
    solar_panel_tilt : float
        Tilt angle of the solar panel in degrees (default 20).
    solar_panel_azimuth : float
        Azimuth angle of the solar panel in degrees (default 270).
    solar_panel_efficiency : float
        Efficiency of the solar panel as a fraction (default 0.18).
    battery_rated_capacity : float
        Capacity of the battery in Wh (default 120).
    battery_efficiency : float
        Efficiency of the battery as a fraction (default 0.8).
    dt : int
        Time step in hours for the simulation (default 1).
    latitude : float
        Latitude of the location (default 40.5853).
    longitude : float
        Longitude of the location (default -105.0844).
    start_date_time : str
        Start date and time for the simulation in the format 'YYYY-MM-DD HH:MM:SS' (default '2024-06-28 12:00:00').
    end_date_time : str
        End date and time for the simulation in the format 'YYYY-MM-DD HH:MM:SS' (default '2024-07-01 13:00:00').
    power_consumption : float
        Power consumption in watts (default 5).
    logging : bool
        Flag to enable logging (default False).
    stop_event : bool
        Flag to stop the simulation (default False).
    shading_ranges : list
        List of tuples specifying shading times and shading percentages (default empty list).
    kwargs : dict
        Additional arguments to override any default keyword arguments.

    Returns:
    --------
    str
        The directory path where simulation results are saved.
    """
    
    # Any additional keyword arguments in **kwargs override the default values
    locals().update(kwargs)
    
    # Notify the user that the solar panel simulation is being executed for the given latitude and longitude
    print(f"*** Modelling Solar ({latitude}, {longitude}) ***")
    
    # Convert the start and end date-time strings into datetime objects
    start_date_time = datetime.strptime(start_date_time, '%Y-%m-%d %H:%M:%S')
    end_date_time = datetime.strptime(end_date_time, '%Y-%m-%d %H:%M:%S')

    # Get the timezone of the location based on latitude and longitude
    timezone = get_timezone(latitude, longitude)

    # Retrieve the necessary weather and solar data for the given location and time period
    df = get_data(start_date_time=start_date_time, end_date_time=end_date_time, latitude=latitude, timezone=timezone, longitude=longitude, logging=logging, stop_event=stop_event)

    # Set up logging for the simulation if enabled
    import logging
    filename = 'validation.log'  # Log file name
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Reduce logging verbosity for Werkzeug to avoid excess log messages
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # Function to return constant power consumption at each time step
    def power_out_func(*args, **kwargs):
        return power_consumption

    # Call the solar panel model with provided parameters to simulate energy generation and battery charging
    time_steps, charges, power_ins, power_outs = solar_panel_model(
        latitude, longitude, logging, start_date_time, end_date_time, df,
        solar_panel_area, solar_panel_tilt, solar_panel_azimuth, solar_panel_efficiency, 
        battery_rated_capacity, battery_efficiency, dt,
        power_out_func, stop_event=stop_event, shading_ranges=shading_ranges)
    
    ## Save the simulation results to files
    import os
    dir = "../../model_results_numpy/"  # Base directory for saving results
    folder_name = "Solar" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Unique folder name based on current timestamp
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.join(dir, folder_name), exist_ok=True)
    
    # Save the time steps of the simulation as a compressed NumPy array
    np.savez_compressed(os.path.join(dir, folder_name, "time_steps.npz"), data=time_steps.astype(np.float32))
    
    # Save the battery charge levels as a compressed NumPy array
    np.savez_compressed(os.path.join(dir, folder_name, "charges.npz"), data=charges.astype(np.float32))
    
    # Save the power input from the solar panels as a compressed NumPy array
    np.savez_compressed(os.path.join(dir, folder_name, "power_ins.npz"), data=power_ins.astype(np.float32))
    
    # Save the power consumption (output) as a compressed NumPy array
    np.savez_compressed(os.path.join(dir, folder_name, "power_outs.npz"), data=power_outs.astype(np.float32))

    # Return the directory path where the results have been saved
    return os.path.join(dir, folder_name)
