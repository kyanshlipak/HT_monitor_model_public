"""
File: validation.py

Description:
------------
This script handles the loading, processing, and visualization of temperature data from model outputs 
and real-world sensor readings for thermal analysis of battery temperatures. It loads the model output 
data from saved numpy arrays, smooths real-world sensor data using the Savitzky-Golay filter, and plots 
comparisons between the model and real data for visualization purposes.

Functions:
----------
- load_data_FoCO: Loads thermal model simulation results and converts them to a pandas DataFrame.
- load_data: Loads data from saved numpy arrays, processes it, and formats it into a pandas DataFrame.
- plot_real_data: Plots smoothed real-world sensor data using matplotlib.
- import_data: Imports sensor data from CSV, drops unnecessary columns, resamples, and applies smoothing.
- import_and_plot: Imports sensor data and plots smoothed data.
- plot_comparison: Compares smoothed real-world sensor data with model data on the same plot.
- plot_data: Plots model temperature data over time.
  
Dependencies:
-------------
- Python 3.x
- NumPy
- Pandas
- Matplotlib (for plotting)
- SciPy (for smoothing)

Author: Kyan Shlipak
Date: 09/28/24
"""

import pandas as pd
import numpy as np
from datetime import * 
import matplotlib.pyplot as plt

# load model output data from saving directory
def load_data_FoCO(results_dir, base_time  = datetime(2024, 6, 19, 9, 0, 0)):
	"""
	Load thermal model simulation results and convert time steps to datetime format.

	Parameters:
	results_dir (str): Directory path containing the model output data files.
	base_time (datetime): Base time for converting time steps to datetime. Default is June 19, 2024, 9:00 AM.

	Returns:
	model_df - pd.DataFrame containing internal, battery, and outside temperature data indexed by datetime.

	Notes:
	The function removes the loaded numpy files and directory after processing the data.
	"""

	# names of ndarrays to get
	import os
	array_names = ["time_steps", "avg_temp", "battery_temp",  "outside_temp", "dt"],

	# load all numpy ndarrays locally
	for var in array_names:
		file_path = results_dir + "/" + var + ".npz"
		globals()[var] = np.load(file_path)['data']
		os.remove(file_path)
	os.rmdir(results_dir)
	# Create a pandas DataFrame
	model_df = pd.DataFrame({'seconds_past': time_steps})

	# Convert seconds_past to Timedelta and add to base_time
	model_df['datetime'] = model_df['seconds_past'].apply(lambda x: base_time + timedelta(seconds=x))
	model_df['internal'] = avg_temp
	model_df['battery_temp'] = battery_temp
	model_df['outside_temp'] = outside_temp
	model_df.set_index('datetime', inplace=True)

	return model_df


# load model output data from saving directory
def load_data(results_dir, base_time  = datetime(2024, 6, 19, 9, 0, 0)):
	"""
	Load time step and temperature data from a directory and convert time steps to datetime.

	Parameters:
	results_dir (str): Directory path containing the numpy arrays.
	base_time (datetime): Base time for converting time steps to datetime. Default is June 19, 2024, 9:00 AM.

	Returns:
	model_df - pd.DataFrame containing time steps and temperature data indexed by datetime.

	Notes:
	The function deletes the numpy arrays and directory after data processing.
	"""

	import os
	# names of ndarrays to get
	array_names = ["time_steps", "avg_temp", "battery_temp",  "outside_temp", "dt"]

	# load all numpy ndarrays locally
	for var in array_names:
		file_path = results_dir + "/" + var + ".npz"
		globals()[var] = np.load(file_path)['data']
		os.remove(file_path)

	# Create a pandas DataFrame
	os.rmdir(results_dir)
	model_df = pd.DataFrame({'seconds_past': time_steps})

	# Convert seconds_past to Timedelta and add to base_time
	model_df['datetime'] = model_df['seconds_past'].apply(lambda x: base_time + timedelta(seconds=x))
	model_df['internal'] = avg_temp
	model_df['battery_temp'] = battery_temp
	model_df['outside_temp'] = outside_temp
	model_df.set_index('datetime', inplace=True)

	return model_df

# plot thermocouple data from raspi
def plot_real_data(sensor_readings_df, smoothed_df, labels_dict, start_date_time, end_date_time, logging):
	"""
	Plot real-world smoothed temperature sensor readings over time.

	Parameters:
	sensor_readings_df (pd.DataFrame): DataFrame with original sensor readings.
	smoothed_df (pd.DataFrame): DataFrame with smoothed sensor readings.
	labels_dict (dict): Dictionary mapping sensor column indices to human-readable labels.
	start_date_time (datetime): Start time for the plot range.
	end_date_time (datetime): End time for the plot range.
	logging (logging.Logger): Logger instance for logging the process.

	Returns:
	None

	Notes:
	The plot is saved as a PNG file showing temperature variations over time.
	"""	
	
	import matplotlib.dates as mdates
	_, ax = plt.subplots(figsize = (15,6))
	logging.info("*** Plotting ***")

	# for col in sensor_readings_df.columns:
	# 	ax.plot(sensor_readings_df[col]) #, label = labels_dict[int(col[0])])
	for col in smoothed_df.columns:
		ax.plot(smoothed_df[col], label = labels_dict[int(col[0])])

	ax.xaxis.set_major_locator(plt.MaxNLocator(15))  # Limit to 10 date labels
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
	plt.ylim([0,60])
	plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
	plt.ylabel('Temperature (C)')
	plt.title("Temperature Over Time")
	plt.legend()
	plt.savefig(f"../model_results_figures/FoCO_real_data_{start_date_time}_{end_date_time}")  

# import, load, and smooth data from raspi
def import_data(path, logging, window_length, polyorder, columns_to_drop = ['sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'fan']):
	"""
	Import temperature sensor data, resample, and apply Savitzky-Golay smoothing.

	Parameters:
	path (str): Path to the CSV file containing sensor readings.
	logging (logging.Logger): Logger instance for logging the process.
	window_length (int): Window length for the Savitzky-Golay smoothing filter.
	polyorder (int): Polynomial order for the Savitzky-Golay filter.
	columns_to_drop (list): List of columns to remove from the DataFrame.

	Returns:
	sensor_readings_df - pd.DataFrame containing resampled sensor readings.
	smoothed_df - pd.DataFrame containing smoothed sensor readings.

	Notes:
	The Savitzky-Golay filter is used to smooth noisy sensor data.
	"""

	from scipy.signal import savgol_filter
	sensor_readings_df = pd.read_csv(path)
	#print(sensor_readings_df.head(5))
	sensor_readings_df = sensor_readings_df.set_index('timestamp')
	sensor_readings_df.index = pd.to_datetime(sensor_readings_df.index)
	for col in columns_to_drop:
		try:
			sensor_readings_df.drop(columns=[col], inplace=True)
		except Exception as e:
			logging.info("error excepted", e)
	sensor_readings_df.index = sensor_readings_df.index.round('s')
	sensor_readings_df = sensor_readings_df.resample('min').mean()
	
	# Define the window length and polynomial order
	smoothed_df = pd.DataFrame()
	smoothed_df.index = sensor_readings_df.index

	logging.info("*** Data Loaded ***")

	# Apply the Savitzky-Golay filter to each column
	for col in sensor_readings_df.columns:
		smoothed_df[col[-1] + '_smooth'] = savgol_filter(sensor_readings_df[col], window_length, polyorder)
	
	return sensor_readings_df, smoothed_df

# import and plot raspi data
def import_and_plot(path, start_date_time, end_date_time, logging, window_length = 29, polyorder = 2, columns_to_drop = ['sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'fan']):
	"""
    Import and process sensor data, apply smoothing, and plot the data over a specified time range.

    Parameters:
    path (str): File path to the CSV containing sensor readings.
    start_date_time (datetime): Start time for the plot range.
    end_date_time (datetime): End time for the plot range.
    logging (logging.Logger): Logger instance for logging the process.
    window_length (int): Window length for the Savitzky-Golay smoothing filter.
    polyorder (int): Polynomial order for the Savitzky-Golay filter.
    columns_to_drop (list): List of sensor columns to remove before processing.

    Returns:
    sensor_readings_df - pd.DataFrame containing raw sensor data.
    smoothed_df - pd.DataFrame containing smoothed sensor data.

    Notes:
    The real-world data is plotted as a PNG file, including smoothed values.
    """


	sensor_readings_df, smoothed_df = import_data(path, logging, window_length, polyorder, columns_to_drop)
	labels_dict = {0: 'top face inner (same side as 6)',
               1: 'side face inner (same side as 7)',
               2: 'battery',
               3: 'battery',
               4: 'inner air',
               5: 'side face outer',
               6: 'Top face outer',
               7: 'opposite side face outer'}
	
	plot_real_data(sensor_readings_df, smoothed_df, labels_dict, start_date_time, end_date_time, logging)
	return sensor_readings_df, smoothed_df

# plot comparison of raspi and model data
def plot_comparison(smoothed_df, model_dfs, start_date_time, end_date_time, col1 = '2_smooth', col2 = '3_smooth', title = "Temperature Over Time"):
	"""
	Plot a comparison of real-world smoothed sensor data and modeled temperature data.

	Parameters:
	smoothed_df (pd.DataFrame): DataFrame with smoothed sensor readings.
	model_dfs (list): List of dictionaries with model data and their labels.
	start_date_time (datetime): Start time for the comparison plot.
	end_date_time (datetime): End time for the comparison plot.
	col1 (str): Column name for the first sensor reading for battery temperature averaging.
	col2 (str): Column name for the second sensor reading for battery temperature averaging.
	title (str): Title of the plot.

	Returns:
	None

	Notes:
	The Celsius data is converted to Fahrenheit, and the plot is saved as a PNG file.
	"""

	def to_F(num):
		return num * 9/5 + 32
	
	import matplotlib.dates as mdates
	smoothed_df = smoothed_df.loc[(smoothed_df.index >= start_date_time) & (smoothed_df.index <= end_date_time)]

	smoothed_df['batt_avg'] = 0.5 * (smoothed_df[col1] + smoothed_df[col2])

	fig, ax = plt.subplots(figsize = (12,8))

	#plt.plot(smoothed_df['2_smooth'], label = "battery2 temperature with smoothing", color = "red")
	#plt.plot(smoothed_df['3_smooth'], label = "battery3 temperature with smoothing", color = "red")

	for model_dict in model_dfs:
		plt.plot(to_F((model_dict['data'])['internal']-273.15), label = model_dict['label'])
		#plt.plot((model_dict['data'])['battery_temp']-273.15, label = "modelled battery temp")
	plt.plot(to_F(smoothed_df['batt_avg']), label = "Internal Battery Temperature")

	ax.xaxis.set_major_locator(plt.MaxNLocator(12))  # Limit to 12 date labels
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M:%S'))

	plt.xticks(rotation=45, fontsize = 14)  # Rotate x-axis labels for better readability
	plt.yticks(fontsize = 14)
	plt.ylabel('Temperature (F)', fontsize = 16)
	plt.xlabel("Timestamp", fontsize = 14)
	plt.title(title, fontsize = 20)
	plt.legend(fontsize = 14)
	plt.show()


def plot_data(df, latitude, longitude):
	"""
	Plot temperature data from a model output for a specified location.

	Parameters:
	df (pd.DataFrame): DataFrame containing temperature data.
	latitude (float): Latitude of the location for which data is plotted.
	longitude (float): Longitude of the location for which data is plotted.

	Returns:
	None

	Notes:
	The plot is saved as a PNG file, including the latitude, longitude, and timestamp in the filename.
	"""
	import matplotlib.dates as mdates
	model_df = model_df.reset_index()
	#model_df['datetime'] = model_df['datetime'].dt.tz_localize(timezone).dt.tz_convert(timezone)

	_, ax = plt.subplots(figsize = (12,8))
	plt.plot(model_df['internal'] - 273.15, label = "Modelled Internal Temperature")

	ax.xaxis.set_major_locator(plt.MaxNLocator(12))  # Limit to 12 date labels
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M:%S'))

	plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
	plt.ylabel('Temperature (C)')
	plt.title("Temperature Over Time")
	plt.legend()
	plt.savefig(f"../model_results_figures/Global_Validation_{latitude}_{longitude}_{start_date_time.strftime('%Y%m%d')}_{end_date_time.strftime('%Y%m%d')}.png")  
	plt.show()