"""
app.py

Uses run_anywhere.py file
Runs plotly-dash app allowing users to use heat transfer and solar radiation models
Can be run from a compute server
Automatically allows for multiple users and instances 

Author: Kyan Shlipak
Date: 09/28
"""

# import functions
import dash
from dash import dcc, html, Input, Output, State
from threading import Thread, Event
import logging
import dash_bootstrap_components as dbc
from datetime import datetime as dt
from validation import *
import queue
import plotly.graph_objects as go
import plotly.express as px
import os
import tempfile
import zipfile
from flask import Flask
import uuid

# import scripts to run models
from run_anywhere import run_anywhere, run_combined_anywhere

# Initialize Flask server
server = Flask(__name__)

# Dictionary to store user id for each user session
processes = {}

def setup_unique_logger(session_id):
    """
    Set up a logger with a unique log file for each session.

    Parameters:
    session_id (str): Unique identifier for the session.

    Returns:
    logger - logging.Logger configured logger instance.
	log_file_path - string containing the file path of the temporary logfile
    """

	# create the unique logger using the session ID
    logger = logging.getLogger(f'app_logger_{session_id}')
    logger.setLevel(logging.INFO)

    # Create a unique temporary file for logging, which logger will associate with
    log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log", prefix=f"{session_id}_")
    log_file_path = log_file.name
    log_file.close()  # Close the file to allow the logger to write to it

    # Create a file handler for logging
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

	# Return the log file path and logger
    return logger, log_file_path

def verbose_results_solar(logger, solar_df):
	"""
	Log verbose results about solar failure rates to the app console

	Parameters:
	logger - the logfile instance
	solar_df - the model output dataframe of solar predictions
	"""

	logger.info("------ Solar Failure Results -----")

	# find how often solar system left battery uncharged
	df_over_mean = solar_df.loc[solar_df['charges'] <= 0]
	ratio_over_threshold_mean = len(df_over_mean) / len(solar_df)

	# find how many days contained at least an instance of battery empty
	days_over_threshold_mean = len(df_over_mean.index.strftime("%Y-%m-%d").unique())

	# log results
	logger.info(f"Battery charge was at zero for {round(ratio_over_threshold_mean * 100, 2)}% of the time for the last ten years")
	logger.info(f"Battery charge was at zero on {days_over_threshold_mean} days ({round(100 * days_over_threshold_mean/(365*5),2)} % )of the last ten years\n")

def verbose_results(logger, model_df, batt_threshold, batt_min_threshold, with_upper = False, upper_df = False, lower_df = False):
	"""
	Log verbose results about heat transfer model predicted battery temperature failure rates to the app console

	Parameters:
	logger - the logfile instance
	model_df - the model output dataframe of heat transfer predictions
	batt_threshold - the maximum battery temperature allowed (prior user input)
	batt_min_threshold - the minimum battery temperature allowed (prior user input)
	with_upper - if bounds are to be outputted verbose
	upper_df - 90% case dataframe of HT model predictions
	lower_df - 10% case dataframe of HT model predictions
	"""

	# log maximium and minimum temperatures
	logger.info("------ Results -----")
	logger.info(f"Maximum internal temperature reached for 2023 case: { round(max(model_df['internal']) - 273.15, 3) } C")
	logger.info(f"Minimum internal temperature reached for 2023 case: { round(min(model_df['internal']) - 273.15, 3) } C")
	if with_upper: 
		logger.info(f"Maximum internal temperature reached for 90% case: { round(max(upper_df['internal']) - 273.15, 3) } C")
		logger.info(f"Minimum internal temperature reached for 10% case: { round(min(lower_df['internal']) - 273.15, 3) } C")

	logger.info(f"Maximum outdoor temperature reached: { round(max(model_df['outside_temp']) - 273.15, 3)} C")
	logger.info(f"Minimum outdoor temperature reached: { round(min(model_df['outside_temp']) - 273.15, 3)} C\n")

	# calculate 2023 dataframe failure rates over battery temperature threshold
	df_over_2023 = model_df.loc[model_df['battery_temp'] >= batt_threshold + 273.15]
	ratio_over_threshold_mean = len(df_over_2023) / len(model_df)
	days_over_threshold_mean = len(df_over_2023.index.strftime("%Y-%m-%d").unique())
	logger.info(f"Battery temperature was over threshold for {round(ratio_over_threshold_mean * 100, 2)}% of the time for the 2023 case")
	logger.info(f"Battery temperature was over threshold on {days_over_threshold_mean} days for the 2023 case\n")

	# calculate 2023 dataframe failure rates under battery temperature minimum threshold
	df_under_mean = model_df.loc[model_df['battery_temp'] <= batt_min_threshold + 273.15]
	ratio_under_threshold_mean = len(df_under_mean) / len(model_df)
	days_under_threshold_mean = len(df_under_mean.index.strftime("%Y-%m-%d").unique())
	logger.info(f"Battery temperature was under threshold for {round(ratio_under_threshold_mean * 100, 2)}% of the time for the 2023 case")
	logger.info(f"Battery temperature was under threshold on {days_under_threshold_mean} days for the 2023 case\n")

	if with_upper:
		# calculate 2023 dataframe failure rates over battery temperature max threshold for 90% case
		df_over_upper = upper_df.loc[upper_df['battery_temp'] >= batt_threshold + 273.15]
		ratio_over_threshold_upper = len(df_over_upper) / len(upper_df)
		days_over_threshold_upper = len(df_over_upper.index.strftime("%Y-%m-%d").unique())
		logger.info(f"Battery temperature was over threshold for {round(ratio_over_threshold_upper * 100, 2)}% of the time for the 90% case")
		logger.info(f"Battery temperature was over threshold on {days_over_threshold_upper} days for the 90% case\n")

		# calculate 2023 dataframe failure rates under battery temperature min threshold for 10% case
		df_under_lower = lower_df.loc[lower_df['battery_temp'] <= batt_min_threshold + 273.15]
		ratio_under_threshold_lower = len(df_under_lower) / len(lower_df)
		days_under_threshold_lower = len(df_under_lower.index.strftime("%Y-%m-%d").unique())
		logger.info(f"Battery temperature was under threshold for {round(ratio_under_threshold_lower * 100, 2)}% of the time for the 10% case")
		logger.info(f"Battery temperature was under threshold on {days_under_threshold_lower} days for the 10% case\n")

def chart_solar_failure(solar_data):
	"""
	Create a heatmap of solar battery failure for each hour of each month of the year (10 years of historical data 2014-2023 used)

	Parameters:
	solar_data - the solar model predictions of battery power and charge over time for 10 years

	Returns:
	fig - plotly figure of monthly hourly solar energy failure
	"""
	import pandas as pd
	import plotly.graph_objects as go

	# Convert Period columns to strings or numeric values
	solar_data['month'] = solar_data.index.month
	solar_data['hour'] = solar_data.index.hour

	# Mark failures based on charges
	solar_data['failure'] = (solar_data['charges']/3600 <= 0.001) * 100

	# Aggregate the data
	agg_df = solar_data.groupby(['month', 'hour']).agg({'failure': 'mean'}).reset_index()

	# Fix the issue with month and hour ranges
	agg_df['month'] = agg_df['month'] - 1  # Adjust month to 0-11 range
	agg_df['hour'] = agg_df['hour'] % 24  # Ensure hour is within 0-23 range

	# Pivot the DataFrame to create a matrix for the heatmap
	heatmap_data = agg_df.pivot(index='hour', columns='month', values='failure')

	# Define month abbreviations
	month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

	# Define x-tick positions for months
	n_months = len(heatmap_data.columns)
	month_tick_positions = [i for i in range(n_months)]
	month_tick_labels = month_abbr  # Use month abbreviations

	# Create the heatmap
	fig = go.Figure(data=go.Heatmap(
		z=heatmap_data.values,  # 2D array for heatmap
		x=heatmap_data.columns, # Months
		y=heatmap_data.index,   # Hours
		colorscale='Magma',
		colorbar=dict(title='Failure Rate'),
	))

	# Update figure layout
	fig.update_layout(
		title='Failure Rates Heatmap (Hours x Months)',
		xaxis_title='Month',
		yaxis_title='Hour of the Day',
		xaxis=dict(
			tickmode='array',
			tickvals=month_tick_positions,  # Center ticks on bins
			ticktext=month_tick_labels,  # Use month abbreviations
			tickson='labels',  # Ensure ticks are centered on labels
			title=dict(text='Month'),
		),
		yaxis=dict(
			tickmode='linear',
		),
		margin=dict(l=40, r=40, t=40, b=60),
		height=400,
		autosize=True,
		coloraxis_colorbar=dict(
			title='Failure Rate',
		)
	)

	return fig

def chart_failure(model_df, batt_threshold, batt_min_threshold, bounds = False, upper_df = False, lower_df = False, solar = False, solar_df = False):
	"""
	Char hourly failure rates for solar and heat transfer model predictions

	Parameters:
	model_df - the model output dataframe of heat transfer predictions
	batt_threshold - the maximum battery temperature allowed (prior user input)
	batt_min_threshold - the minimum battery temperature allowed (prior user input)
	bounds - boolean value storing if bounds are to be outputted verbose
	upper_df - 90% case dataframe of HT model predictions
	lower_df - 10% case dataframe of HT model predictions
	solar - boolean value storing if solarmodel has been run and is to be included in results plots
	solar_df - solar model output dataframe of battery power & charge predictitons

	Returns:
	fig - plotly figure of hourly battery temperature and solar energy failure
	"""
	
	# Create the figure
	fig = go.Figure()

	# Calculate datapoints outside battery temperature thresholds
	lower_threshold = batt_min_threshold + 273.15
	upper_threshold = batt_threshold + 273.15
	model_df['Outside_Threshold'] = (model_df['internal'] < lower_threshold) | (model_df['internal'] > upper_threshold)

	# Calculate the percentage of time outside the threshold for each hour
	hourly_outside_threshold = model_df['Outside_Threshold'].groupby(model_df.index.hour).mean() * 100

	# Add the bar plot for temperature failure rates per hour
	fig.add_trace(go.Bar(
		x=hourly_outside_threshold.index,
		y=hourly_outside_threshold.values,
		name='Outside Threshold',
		marker_color='blue'
	))

	# If solar df included, bar plot hourly mean failure rates (battery empty)
	if solar:
		solar_df['Failure'] = (solar_df['charges'] <= 0.001)
		hourly_solar_failure = solar_df['Failure'].groupby(solar_df.index.hour).mean() * 100
		fig.add_trace(go.Bar(
			x=hourly_solar_failure.index,
			y=hourly_solar_failure.values,
			name='Battery Energy Failure',
			marker_color='orange'
		))

	# figure settings
	fig.update_layout(
		title='Failure Rate for Each Hour',
		xaxis_title='Hour of the Day',
		yaxis_title='Percentage of Time Outside Threshold',
		xaxis=dict(
		#    tickmode='auto',  # Adjust as needed
			nticks=24  # Limit to 12 date labels
		),
		margin=dict(l=40, r=40, t=40, b=60),
		height=400,
		autosize=True,
		legend=dict(
			x=0.99, y=0.01,  # Position the legend in the bottom right corner
			xanchor='right',
			yanchor='bottom',
			bgcolor='rgba(255, 255, 255, 0.5)'  # Set background color to semi-transparent white
		),
	)

	# Return the figure
	return fig

def plot_solar_data(solar_df):
	"""
	Create a plot of solar battery charge for each year and an average charge (10 years of historical data 2014-2023 used)

	Parameters:
	solar_df - the solar model predictions of battery power and charge over time for 10 years

	Returns:
	fig - plotly figure of solar model battery predictions over the course of each year
	"""
		
	fig = go.Figure() #create plotly figure

	def safe_datetime(year, month, day, hour):
		"""
		Given datetime parameters, create and return a datetime object while avoiding any leap year issues

		Parameters:
		year - integer year
		month - integer month
		day - integer day
		hour - integer hour

		Returns:
		pd.Timestamp object generated given the inputs (leap day of non-leap years will return NaT)
		"""
		try:
			return pd.Timestamp(year=year, month=month, day=day, hour=hour)
		except ValueError:
			return pd.NaT  # Return a NaT (Not a Time) if there's an error

	# Create a new datetime column using vectorized operations
	all_years_data = []

	# iterate over each year in the solar dataframe
	for year in solar_df.index.year.unique():
		df_year = solar_df.loc[solar_df.index.year == year] # get just that specific year into a dataframe
		all_years_data.append(df_year[['charges']]/3600) # add chagrgses to the all_years_dat list

		# switch index to 2023 so all years can be plotted with the same x axis
		df_year.index = [safe_datetime(2023, month, day, hour) \
			for month, day, hour in zip(df_year.index.month, df_year.index.day, df_year.index.hour)] 

		# add the plot trace for the charge over time for that year
		fig.add_trace(go.Scatter(
			x=df_year.index,
			y=df_year['charges']/3600,
			mode='lines',
			name=f"Modelled Charge {year}",
			visible='legendonly'
		))

	# create dataframe concatenating data from each year
	concatenated_df = pd.concat(all_years_data)

	# resample to the 10 minute resolution in case finer resolution model is used
	concatenated_df = concatenated_df.resample('10min').mean()

	# set time columns for ease of grouping
	concatenated_df['month'] = concatenated_df.index.month
	concatenated_df['day'] = concatenated_df.index.day
	concatenated_df['hour'] = concatenated_df.index.hour
	concatenated_df['minute'] = concatenated_df.index.minute

	# group by time of year ato get the mean
	grouped_df = concatenated_df.groupby(['month', 'day','hour','minute'])['charges'].mean().reset_index()

	# exclude leap days
	grouped_df = grouped_df.loc[(grouped_df['month'] != 2) | (grouped_df['day'] != 29)]

	# set datetime index
	grouped_df.index = pd.to_datetime(
		grouped_df[['month', 'day','hour','minute']].assign(Year=2023),
		format='%Y-%m-%d %H:%M'
	)

	# Create a trace for the mean charge
	fig.add_trace(go.Scatter(
		x=grouped_df.index,
		y=grouped_df['charges'],
		mode='lines',
		name='Mean Charge',
		line=dict(color='black', width=2)
	))
	# Update figure layout
	fig.update_layout(
		title='Variation in Battery Charge Captured Over a Year',
		xaxis_title='Date',
		yaxis_title='Charge (W*H)',
		xaxis=dict(
			tickformat='%m',
			tickmode='auto',  # Adjust as needed
			nticks=12  # Limit to 12 date labels
		),
		xaxis_tickangle=-45,  # Rotate x-axis labels for readability
		margin=dict(l=40, r=40, t=40, b=60),
		height=400,
		autosize=True,
		legend=dict(
			x=0.99, y=0.01,  # Position the legend in the bottom right corner
			xanchor='right',
			yanchor='bottom',
			bgcolor='rgba(255, 255, 255, 0.5)'  # Set background color to semi-transparent white
		),
	)

	return fig

def plot_data(model_df, model_df_daily_max, bounds = False, lower_df_daily_max = False, upper_df_daily_max = False, battery_threshold = 10, batt_min_threshold = 0, model_df_daily_min = False):
	"""
	Char hourly failure rates for solar and heat transfer model predictions

	Parameters:
	model_df - the model output dataframe of heat transfer predictions
	model_df_daily_max - model output dataframe of daily maximum temperatures
	bounds - boolean value storing if bounds were modeled and are to be plotted
	lower_df_daily_max - 10% historical data case model minimum daily temperature output
	upper_df_daily_max - 90% historical data case model maximum daily temperature output
	batt_threshold - the maximum battery temperature allowed (prior user input)
	batt_min_threshold - the minimum battery temperature allowed (prior user input)
	model_df_daily_min - model output dataframe of daily minimum temperatures

	Returns:
	fig - plotly figure of model predicted temperatures over time
	"""
	
	# Clear previous figure traces and layout settings
	fig = go.Figure()

	# plot modeled battery temperature
	fig.add_trace(go.Scatter(
		x=model_df.index,
		y=model_df['internal'] - 273.15,
		mode='lines',
		name='Modelled Internal Temp For 2023 Weather Conditions',
		line=dict(color='orange')
	))

	# plot NASA Power API outdoor temperature (drybulb)
	fig.add_trace(go.Scatter(
		x=model_df.index,
		y=model_df['outside_temp'] - 273.15,
		mode='lines',
		name="Outdoor Temperature",
		line=dict(color='green')
	))

	# create array for battery threshold values
	batt_threshold_array = np.full(len(model_df.index), battery_threshold)
	batt_min_threshold_array = np.full(len(model_df.index), batt_min_threshold)

	# Plot dashed lines for battery upper threshold
	fig.add_trace(go.Scatter(
		x=model_df.index,
		y=batt_threshold_array,
		mode='lines',
		name='Battery Max Temp',
		line=dict(color='black', dash='dash')
	))

	# Plot dashed lines for battery lower threshold
	fig.add_trace(go.Scatter(
		x=model_df.index,
		y=batt_min_threshold_array,
		mode='lines',
		name='Battery Min Temp',
		line=dict(color='black', dash='dash')
	))

	# if modeled historical 10% and 90% bounds, plot predictions
	if bounds:
		fig.add_trace(go.Scatter(
			x=upper_df_daily_max.index + pd.DateOffset(hours=15),
			y=upper_df_daily_max['internal'] - 273.15,
			mode='lines',
			name="Modelled Daily Max Internal Temp 90% ",
			line=dict(color='red')
		))

		fig.add_trace(go.Scatter(
			x= lower_df_daily_max.index + pd.DateOffset(hours=15),
			y= lower_df_daily_max['internal'] - 273.15,
			mode='lines',
			name="Modelled Daily Min Internal Temp 10% ",
			line=dict(color='blue')
		))

	else:
		# otherwise, plot te ddaily min and max values for the standard 2023 model predictions
		fig.add_trace(go.Scatter(
			x=model_df_daily_max.index + pd.DateOffset(hours=15),
			y=model_df_daily_max['internal'] - 273.15,
			mode='lines',
			name='Modelled Daily Max Internal Temp',
			line=dict(color='red')
		)),
	
		fig.add_trace(go.Scatter(
			x=model_df_daily_min.index + pd.DateOffset(hours=3),
			y=model_df_daily_min['internal'] - 273.15,
			mode='lines',
			name='Modelled Daily Min Internal Temp',
			line=dict(color='blue')
		))


	# Update figure layout
	fig.update_layout(
		title='Variation in Temperature Over Time',
		xaxis_title='Date',
		yaxis_title='Temperature (C)',
		xaxis=dict(
			tickformat='%m-%d %H:%M',
			tickmode='auto',  # Adjust as needed
			nticks=12  # Limit to 12 date labels
		),
		xaxis_tickangle=-45,  # Rotate x-axis labels for readability
		margin=dict(l=40, r=40, t=40, b=60),
		height=400,
		autosize=True,
		legend=dict(
			x=0.99, y=0.01,  # Position the legend in the bottom right corner
			xanchor='right',
			yanchor='bottom',
			bgcolor='rgba(255, 255, 255, 0.5)'  # Set background color to semi-transparent white
		),
	)

	return fig

def run_model(logger, q, latitude, longitude, battery_capacity, battery_threshold, length, width, height, thickness, fan_flow, heat_gen, fan_heat_gen, threshold, start_date_time, end_date_time, stop_event, material, bounds,
					solar = False, solar_panel_area = 0, solar_panel_tilt = 0, solar_panel_azimuth = 0, solar_panel_efficiency = 0, power_consumption = 0, battery_rating = 0.8, batt_min_threshold = 0, box_shading= False, shading_ranges = []):
	"""
	Run heat transfer and solar radiation models as applicable, generate verbose results, generate figures, and save data

	Parameters:
	logger - logging temporary file instance for outputting text to app console
	q - the queue.Queue() for storing outputs (can't return from the thread)
	latitude - the user input latitude for running the model
	longitude - the user input longitude for running the model
	battery capacity - W*H the battery can hold (user input)
	battery threshold - maximum battery temperature allowed (user input) [C]
	length - enclosure length (user input) [m]
	width - enclosure width (user input) [m]
	height - enclosure height (user input) [m]
	thickness - enclosure thickness (user input) [m]
	fan_flow - fan volumetric flow (user input) [CFM]
	heat_gen - internal heat generation from things in enclosure (user input) [W]
	fan_heat_gen - internal heat generation from the fan specifically (user input) [W]
	threshold - temperature threshold for fan to turn on (user input) [C]
	start_date_time - datetime string for starting the HT model run period (user input)
	end_date_time - datetime string for ending the HT model run period (user input)
	stop_event - the threading stop event for stopping execution of the model when cancel button is pressed
	material - dictionary of material properties (user chosen)
	bounds - boolean containing if 10% and 90% historical weather bounds will be run for historical model (user input)
	solar - boolean containing if solar model will be run (user input)
	solar_panel_area - area of solar panel (user input) [m]
	soalar_panel_tilt - tilt of solar panel (user input) [degrees]
	solar_panel_azimuth - orientation of solar panel (degrees clockwise from north) (user input)
	solar_panel_efficiency - efficency of solar panel (user input)
	power_consumption - sensor power consumption (assumed to be constant) (user input) [W]
	battery_rating - true capacity fraction of rated battery capacity (user input)
	batt_min_threshold - minimum operating temperature for battery (user input) [C]
	box_shading - boolean containing weather or not the sensor/enclosure is shaded by the solar panel (user input)
	shading_ranges - list of tuples containing time strings of time periods within a day during which the setup is shaded (user input)
	"""
	
	# Create a temporary directory to store files
	temp_dir = tempfile.mkdtemp()

	# Path to save pickle files within the temporary directory
	pickle_dir = os.path.join(temp_dir, 'pickle_files')

	# Ensure the directory exists
	os.makedirs(pickle_dir, exist_ok=True)

	# Default number of steps is 4, use 3 if the material is highly thermally conductive
	N = 4
	if material['thermal_conductivity'] > 10:
		N = 3
		if thickness < 0.01:
			thickness = 0.01 # if material is super conductive, set upper bound of thickness modelled

	# run model with no solar components
	if not solar:
		try:
			logger.info("Function Started")
			lower_dir = "3" #arbitrary definition just to initiate
			upper_dir = "3" #arbitrary definition just to initiate

			# model without bounds
			if not bounds:
				dir_2023 = run_anywhere(latitude=latitude, longitude=longitude, logging = logger,
											T_max=45 + 273.15, T_initial=20 + 273.15, # these can also stay the same
											k = material['thermal_conductivity'], 
											rho = material['density'],
											Cp = material['Cp'], 
											emissivity = material['emissivity'], 
											absorptivity = material['emissivity'],  # can keep these the same for any run
											B_air=0.00367, v_air=15.89e-6, k_air=0.024, alpha_air=22.5e-6, h_max=100, # constant air properties
											L=length, W=width, H=height, N=N,  # 
											thicknesses=[thickness, thickness, thickness, thickness, thickness, thickness], #
											fan_flow=fan_flow, heat_generation=heat_gen, fan_heat_generation=fan_heat_gen, fan_threshold = threshold, # specific to his fan setup
											start_date_time = start_date_time, #
											end_date_time = end_date_time,
											bounded = False,
											quantiles = [0.1, 0.9],
											stop_event = stop_event,
											box_shading = box_shading,
											shading_ranges = shading_ranges) 
				if dir_2023 == "":
					logger.info("Function canceled due to error \n")
					return dash.no_update

			# model with bounds
			elif bounds:
				lower_dir, upper_dir, dir_2023 = run_anywhere(latitude=latitude, longitude=longitude, logging = logger,
										T_max=45 + 273.15, T_initial=20 + 273.15, # these can also stay the same
										k = material['thermal_conductivity'], 
										rho = material['density'],
										Cp = material['Cp'], 
										emissivity = material['emissivity'], 
										absorptivity = material['emissivity'],  # can keep these the same for any run
										B_air=0.00367, v_air=15.89e-6, k_air=0.024, alpha_air=22.5e-6, h_max=100, # constant air properties
										L=length, W=width, H=height, N=N,  # 
										thicknesses=[thickness, thickness, thickness, thickness, thickness, thickness], #
										fan_flow=fan_flow, heat_generation=heat_gen, fan_heat_generation=fan_heat_gen, fan_threshold = threshold, # specific to his fan setup
										start_date_time = start_date_time, #
										end_date_time = end_date_time,
										bounded = True,
										quantiles = [0.1, 0.9],
										stop_event = stop_event,
										box_shading = box_shading,
										shading_ranges = shading_ranges)
				
				if dir_2023 == "" or lower_dir == "" or upper_dir == "":
					logger.info("Function canceled due to error \n")
					return dash.no_update
				
		except Exception as e:
			logger.info("Error running function. Double check inputs and try again. " + str(e))
			return dash.no_update

		else:
			logger.info(f"Function complete \n")

		#load data from results directories (which are temporary)
		import time
		file_name = f"{latitude}_{longitude}_{start_date_time}_{end_date_time}"
		model_df = load_data(dir_2023, base_time = dt.strptime(start_date_time, '%Y-%m-%d %H:%M:%S'))
		
		# save dataframes to pickles
		model_df = model_df.resample('min').mean()  #resample/downsample to the minute resoltuion
		model_df.to_pickle(os.path.join(pickle_dir,file_name+"_mean.pkl"))
		model_df_daily_max = model_df.resample('D').max()
		model_df_daily_min = model_df.resample('D').min()

		# if bounds modeled, do the same with historical bound model predictions
		if bounds:
			upper_df = load_data(upper_dir, base_time = dt.strptime(start_date_time, '%Y-%m-%d %H:%M:%S'))
			lower_df = load_data(lower_dir, base_time = dt.strptime(start_date_time, '%Y-%m-%d %H:%M:%S'))

			lower_df = lower_df.resample('min').mean()  #resample/downsample to the minute resolution
			upper_df = upper_df.resample('min').mean()  #resample/downsample to the minute resolution

			upper_df.to_pickle(os.path.join(pickle_dir,file_name+"_upper.pkl"))
			lower_df.to_pickle(os.path.join(pickle_dir,file_name+"_lower.pkl"))

			lower_df_daily_min = lower_df.resample('D').min()
			upper_df_daily_max = upper_df.resample('D').max()

			plotly_fig = plot_data(model_df, model_df_daily_max, bounds = bounds,
							lower_df_daily_max = lower_df_daily_min, 
							upper_df_daily_max = upper_df_daily_max, battery_threshold = battery_threshold, batt_min_threshold = batt_min_threshold)
			chart_fig = chart_failure(model_df, batt_threshold = battery_threshold, batt_min_threshold = batt_min_threshold, bounds = True, upper_df = upper_df, lower_df = lower_df)

			verbose_results(logger, model_df, battery_threshold, batt_min_threshold = batt_min_threshold, with_upper = True, upper_df = upper_df, lower_df = lower_df)
		else:
			plotly_fig = plot_data(model_df, model_df_daily_max, bounds = bounds, battery_threshold = battery_threshold, batt_min_threshold = batt_min_threshold, model_df_daily_min = model_df_daily_min)
			chart_fig =chart_failure(model_df, batt_threshold = battery_threshold, batt_min_threshold = batt_min_threshold, bounds = False)
			verbose_results(logger, model_df, battery_threshold, batt_min_threshold = batt_min_threshold,  with_upper = False)
		# plot the results
		figures = [plotly_fig, chart_fig]
	
	# if running solar model
	elif solar:
		try:
			logger.info("Function Started")
			lower_dir = "3" #arbitrary definition
			upper_dir = "3" #arbitrary definition

			# run without historical data bounds for HT model
			if not bounds:
				model_df, solar_df = run_combined_anywhere(latitude=latitude, longitude=longitude, logging = logger,
											T_max=45 + 273.15, T_initial=20 + 273.15, # these can also stay the same
											k = material['thermal_conductivity'], 
											rho = material['density'],
											Cp = material['Cp'], 
											emissivity = material['emissivity'], 
											absorptivity = material['emissivity'],  # can keep these the same for any run
											B_air=0.00367, v_air=15.89e-6, k_air=0.024, alpha_air=22.5e-6, h_max=100, # constant air properties
											L=length, W=width, H=height, N=N,  # 
											thicknesses=[thickness, thickness, thickness, thickness, thickness, thickness], #
											fan_flow=fan_flow, heat_generation=heat_gen, fan_heat_generation=fan_heat_gen, fan_threshold = threshold, # specific to his fan setup
											start_date_time = start_date_time, #
											end_date_time = end_date_time,
											solar_panel_area = solar_panel_area,
											solar_panel_tilt = solar_panel_tilt,
											solar_panel_azimuth = solar_panel_azimuth,
											solar_panel_efficiency = solar_panel_efficiency,
											battery_rated_capacity = battery_capacity,
											battery_efficiency = battery_rating,
											dt_solar = 60 * 10,
											power_consumption = power_consumption,
											bounded = False,
											quantiles = [0.1, 0.9],
											stop_event = stop_event,
											box_shading = box_shading,
											shading_ranges = shading_ranges) 
			
			# run with historical data bounds for HT model
			elif bounds:
				model_df_lower, model_df_upper, model_df, solar_df\
					 = run_combined_anywhere(latitude=latitude, longitude=longitude, logging = logger,
											T_max=45 + 273.15, T_initial=20 + 273.15, # these can also stay the same
											k = material['thermal_conductivity'], 
											rho = material['density'],
											Cp = material['Cp'], 
											emissivity = material['emissivity'], 
											absorptivity = material['emissivity'],  # can keep these the same for any run
											B_air=0.00367, v_air=15.89e-6, k_air=0.024, alpha_air=22.5e-6, h_max=100, # constant air properties
											L=length, W=width, H=height, N=N,  # 
											thicknesses=[thickness, thickness, thickness, thickness, thickness, thickness], #
											fan_flow=fan_flow, heat_generation=heat_gen, fan_heat_generation=fan_heat_gen, fan_threshold = threshold, # specific to his fan setup
											start_date_time = start_date_time, #
											end_date_time = end_date_time,
											solar_panel_area = solar_panel_area,
											solar_panel_tilt = solar_panel_tilt,
											solar_panel_azimuth = solar_panel_azimuth,
											solar_panel_efficiency = solar_panel_efficiency,
											battery_rated_capacity = battery_capacity,
											battery_efficiency = battery_rating,
											dt_solar = 60 * 10,
											power_consumption = power_consumption,
											bounded = True,
											quantiles = [0.1, 0.9],
											stop_event = stop_event,
											box_shading = box_shading,
											shading_ranges = shading_ranges) 
				logger.info("Finished running combined function")

		except Exception as e:
			logger.info("Error running function. Double check inputs and try again. " + str(e))
			return dash.no_update

		else:
			logger.info(f"Function complete \n")

		#load data
		import time

		#save dataframes as pickles
		file_name = f"{latitude}_{longitude}_{start_date_time}_{end_date_time}"
		model_df = model_df.resample('min').mean()  #resample/downsample to the minute resoltuion
		model_df.to_pickle(os.path.join(pickle_dir,file_name+"_2023.pkl"))
		model_df_daily_max = model_df.resample('D').max()

		model_df_daily_min = model_df.resample('D').min()

		solar_df.to_pickle(os.path.join(pickle_dir,file_name+"_solar_2023.pkl"))

		# get results
		if bounds:
			model_df_lower = model_df_lower.resample('min').mean()  #resample/downsample to the minute resoltuion
			model_df_upper = model_df_upper.resample('min').mean()  #resample/downsample to the minute resoltuion

			model_df_upper.to_pickle(os.path.join(pickle_dir,file_name+"_upper.pkl"))
			model_df_lower.to_pickle(os.path.join(pickle_dir,file_name+"_lower.pkl"))

			lower_df_daily_min = model_df_lower.resample('D').min()
			upper_df_daily_max = model_df_upper.resample('D').max()

			plotly_fig = plot_data(model_df, model_df_daily_max, bounds = bounds,
							lower_df_daily_max = lower_df_daily_min, 
							upper_df_daily_max = upper_df_daily_max, battery_threshold = battery_threshold, batt_min_threshold = batt_min_threshold)

			verbose_results(logger, model_df, battery_threshold, batt_min_threshold = batt_min_threshold, with_upper = True, upper_df = model_df_upper, lower_df = model_df_lower)
			verbose_results_solar(logger, solar_df)
			fig_solar = plot_solar_data(solar_df)
			chart_fig = chart_failure(model_df, batt_threshold = battery_threshold, batt_min_threshold = batt_min_threshold, bounds = True, upper_df = model_df_upper, lower_df = model_df_lower, solar = True, solar_df = solar_df)
			fig_solar_failure = chart_solar_failure(solar_df)
			figures = [plotly_fig, fig_solar, chart_fig, fig_solar_failure]
		else:
			plotly_fig = plot_data(model_df, model_df_daily_max, bounds = bounds, battery_threshold = battery_threshold, batt_min_threshold = batt_min_threshold, model_df_daily_min =model_df_daily_min)
			fig_solar = plot_solar_data(solar_df)
			verbose_results(logger, model_df, battery_threshold, batt_min_threshold = batt_min_threshold, with_upper = False)
			verbose_results_solar(logger, solar_df)
			chart_fig = chart_failure(model_df, batt_threshold = battery_threshold, batt_min_threshold = batt_min_threshold, \
							 bounds = False, solar = True, solar_df = solar_df)
			fig_solar_failure = chart_solar_failure(solar_df)
			figures = [plotly_fig, fig_solar, chart_fig, fig_solar_failure]

	# put figures and directory of pickled saved dataframes in the q (equivalent of returning but this function is run via thread)
	q.put(figures)
	q.put(pickle_dir)

# Material properties dictionary  from https://www.matweb.com/, https://www.engineeringtoolbox.com/emissivity-coefficients-d_447.html
material_properties = {
    'Aluminum': {'emissivity': 0.77, 'Cp': 963, 'density': 2740, 'thermal_conductivity': 96}, #https://www.sunrise-metal.com/aluminum-alloy-a383 https://www.polycase.com/an-19pm, https://www.matweb.com/search/DataSheet.aspx?MatGUID=392ac894c0d642c0947f8d6b975d55de
    'Stainless Steel': {'emissivity': 0.6, 'Cp': 500, 'density': 8030, 'thermal_conductivity': 16.3}, #https://www.matweb.com/search/DataSheet.aspx?MatGUID=0cf4755fe3094810963eaa74fe812895&ckck=1
    'Fiberglass': {'emissivity': 0.92, 'Cp': 1300, 'density': 1370, 'thermal_conductivity': 0.216}, #https://www.matweb.com/search/datasheet.aspx?matguid=c2d2a46a0d774ed99d074f0929e9863b#:~:text=27600%20psi,POC010%20/%2055588
    'Polycarbonate': {'emissivity': 0.92, 'Cp': 1650, 'density': 1200, 'thermal_conductivity': 0.216}, #https://www.matweb.com/search/DataSheet.aspx?MatGUID=84b257896b674f93a39596d00d999d77
    'ABS': {'emissivity': 0.94, 'Cp': 1990, 'density': 1070, 'thermal_conductivity': 0.162}, #https://www.matweb.com/search/DataSheet.aspx?MatGUID=eb7a78f5948d481c9493a67f0d089646
	'PVC': {'emissivity': 0.92, 'Cp': 900, 'density': 1390, 'thermal_conductivity': 0.157}, #https://www.matweb.com/search/DataSheet.aspx?MatGUID=bb6e739c553d4a34b199f0185e92f6f7, https://www.matweb.com/search/datasheet.aspx?matguid=9e30fe79a7d74a168c8c970381ac6b99
}


# Define allowed and warning bounds for each dcc.Input input id
input_bounds = {
    'Latitude': {'allowed': (-90, 90), 'warning': (-90, 90)},
    'Longitude': {'allowed': (-180, 180), 'warning': (-180, 180)},
    'input-emissivity': {'allowed': (0, 1), 'warning': (0.5, 0.98)},
    'input-cp': {'allowed': (0, 10000), 'warning': (400, 2200)},
    'input-density': {'allowed': (0, 10000), 'warning': (500, 9000)},
    'input-thermal-conductivity': {'allowed': (0, 250), 'warning': (0.03, 150)},
    'length': {'allowed': (0.01, 10), 'warning': (0.05, 0.5)},
    'width': {'allowed': (0.01, 10), 'warning': (0.05, 0.5)},
    'height': {'allowed': (0.01, 10), 'warning': (0.05, 0.5)},
    'thickness': {'allowed': (0.001, 10), 'warning': (0.0025, 0.02)},
    'batt_threshold': {'allowed': (0, 150), 'warning': (30, 70)},
    'fan_flow': {'allowed': (0, 50), 'warning': (0, 20)},
    'threshold': {'allowed': (-50, 100), 'warning': (-40, 90)},
    'heat_gen': {'allowed': (0, 100), 'warning': (0, 10)},
	'fan_heat_gen': {'allowed': (0, 100), 'warning': (0, 5)},
    'batt_min_threshold': {'allowed': (-50, 100), 'warning': (-40, 90)},
    'solar_panel_area': {'allowed': (0, 100), 'warning': (0.05, 1)},
    'solar_panel_tilt': {'allowed': (0, 90), 'warning': (0, 70)},
    'solar_panel_azimuth': {'allowed': (0, 360), 'warning': (0, 360)},
    'solar_panel_efficiency': {'allowed': (0.01, 0.5), 'warning': (0.1, 0.3)},
    'power_consumption': {'allowed': (0, 100), 'warning': (0.2, 10)},
    'capacity': {'allowed': (0.1, 10000), 'warning': (5, 200)},
    'battery_rating': {'allowed': (0.01, 1), 'warning': (0.5, 0.95)},
}

# Dash app initialization with Bootstrap theme
app = dash.Dash(__name__, server = server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout 
app.layout = dbc.Container([
	dcc.Store(id='pickle_dir', storage_type='session'),
	dcc.Store(id='error-store', data=False, storage_type='session'),
	dcc.Store(id='unique-tab-id', storage_type='session'),  # Store component to hold unique tab ID in session storage
	dcc.Store(id='shading_ranges', data = [], storage_type='session'),
	dcc.Input(id="tab-id", type="hidden"),  # Hidden input for storing tab ID,
	html.H1("Air Quality Monitor Heat Transfer Model", className='text-center mb-4'),
	dbc.Row([
		dbc.Col([
			dcc.Input(id='Latitude', type='number', placeholder='Latitude (degrees)', min=-90, max=90, debounce=True, className='form-control'),
			html.Div(id='Latitude-warning', className='text-danger')
		], width=3),
		dbc.Tooltip("Enter Latitude in degrees [-90, 90]", target='Latitude'),
		dbc.Col([
			dcc.Input(id='Longitude', type='number', placeholder='Longitude (degrees)', min=-180, max=180, debounce=True, className='form-control'),
			html.Div(id='Longitude-warning', className='text-danger')
		], width=3),
		dbc.Tooltip("Enter Longitude in degrees [-180, 180]", target='Longitude'),
		dbc.Col(dcc.Dropdown(
			id='material-dropdown',
			options=[{'label': material, 'value': material} for material in material_properties.keys()] + [{'label': 'Other', 'value': 'Other'}],
			value='ABS'
		)),
		dbc.Tooltip("Select the material for the enclosure", target='material-dropdown'),
		dbc.Col(dcc.DatePickerRange(
			id='Date Range',
			start_date=dt(2023, 7, 1),
			end_date=dt(2023, 7, 31),
			min_date_allowed=dt(2001, 1, 1),
			max_date_allowed=dt(2023, 12, 31),
			display_format='MM-DD',
			style={'width': '100%'}
		), width=4),
		dbc.Tooltip("Select the date range for analysis. Typical year weather is modeled.", target='Date Range'),
	], className='mb-3'),

	dbc.Row(id='other-inputs', style={'display': 'none'}, children=[
		dbc.Col([
			dbc.Input(id='input-emissivity', type='number', placeholder='Emissivity', className='form-control'),
			html.Div(id='input-emissivity-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter emissivity of the enclosure material", target='input-emissivity'),
		dbc.Col([
			dbc.Input(id='input-cp', type='number', placeholder='Cp [J/kg K]', className='form-control'),
			html.Div(id='input-cp-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter specific heat capacity of the enclosure material in J/kg K", target='input-cp'),
		dbc.Col([
			dbc.Input(id='input-density', type='number', placeholder='Density [kg/m^3]', className='form-control'),
			html.Div(id='input-density-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter density of the enclosure material in kg/m^3", target='input-density'),
		dbc.Col([
			dbc.Input(id='input-thermal-conductivity', type='number', placeholder='Thermal Conductivity [W/m K]', className='form-control'),
			html.Div(id='input-thermal-conductivity-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter thermal conductivity of the enclosure material in W/mK", target='input-thermal-conductivity'),
	], className='mb-3'),

	dbc.Row([
		dbc.Col([
			dcc.Input(id='length', type='number', placeholder='Enclosure Length (m)', className='form-control'),
			html.Div(id='length-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter enclosure length in meters", target='length'),
		dbc.Col([
			dcc.Input(id='width', type='number', placeholder='Enclosure Width (m)', className='form-control'),
			html.Div(id='width-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter enclosure width in meters", target='width'),
		dbc.Col([
			dcc.Input(id='height', type='number', placeholder='Enclosure Height (m)', className='form-control'),
			html.Div(id='height-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter enclosure height in meters", target='height'),
		dbc.Col([
			dcc.Input(id='thickness', type='number', placeholder='Wall thickness (m)', className='form-control'),
			html.Div(id='thickness-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter wall thickness in meters", target='thickness'),
	], className='mb-3'),

	dbc.Row([
		dbc.Col([
			dcc.Input(id='heat_gen', type='number', placeholder='Internal Heat Generation (W)', className='form-control'),
			html.Div(id='heat_gen-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter internal heat generation in watts", target='heat_gen'),

		dbc.Col([
			dcc.Input(id='batt_threshold', type='number', placeholder='Battery Maximum Allowed Temperature (C)', className='form-control'),
			html.Div(id='batt_threshold-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter battery maximum allowed temperature in Celsius", target='batt_threshold'),
		
		dbc.Col([
			dbc.Button("Add Shading Range", id="add-range-selector", color="primary", className="mb-2"),
		]),

		dbc.Col([
			dbc.Button("Remove Shading Range", id="remove-range-selector", color="danger", className="mb-2"),
		]),

		html.Div(id="range-selectors-container"),
		html.Div(id="output-range-values"),

		dbc.Tooltip("Add time range of day shading on sensor", target='add-range-selector'),
		dbc.Tooltip("Remove time range of day shading on sensor", target='remove-range-selector'),

	], className='mb-3'),

	dbc.Row([
		dbc.Col([
			dcc.Input(id='fan_flow', type='number', placeholder='Fan flow (cfm)', className='form-control'),
			html.Div(id='fan_flow-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter fan flow in cubic feet per minute", target='fan_flow'),
		dbc.Col([
			dcc.Input(id='threshold', type='number', placeholder='Fan temp threshold (C)', className='form-control'),
			html.Div(id='threshold-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter fan temperature threshold in Celsius", target='threshold'),
		dbc.Col([
			dcc.Input(id='fan_heat_gen', type='number', placeholder='Fan Heat Generation (W)', className='form-control'),
			html.Div(id='fan_heat_gen-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter fan heat generation in watts", target='fan_heat_gen'),
		dbc.Col([
			dcc.Input(id='batt_min_threshold', type='number', placeholder='Battery Minimum Allowed Temperature (C)', className='form-control'),
			html.Div(id='batt_min_threshold-warning', className='text-danger')
		]),
		dbc.Tooltip("Enter battery minimum allowed temperature in Celsius", target='batt_min_threshold'),
	], className='mb-3'),

	dbc.Row([
		dbc.Col(dbc.Checkbox(id='bounds', value=True, label='Run With Bounds')),
		dbc.Tooltip("Select to run the model with bounds", target='bounds'),
		dbc.Col(dbc.Checkbox(id='solar', value=False, label='Run With Solar')),
		dbc.Tooltip("Select to run the model with bounds", target='bounds'),
		dbc.Row(id='solar_row_1', style={'display': 'none'}, children=[
			dbc.Col([
				dcc.Input(id='solar_panel_area', type='number', placeholder='Solar Area (m³)', className='form-control'),
				html.Div(id='solar_panel_area-warning', className='text-danger')
			]),
			dbc.Tooltip("Enter solar panel area in cubic meters", target='solar_panel_area'),

			dbc.Col([
				dcc.Input(id='solar_panel_tilt', type='number', placeholder='Solar Panel Tilt (Degrees)', className='form-control'),
				html.Div(id='solar_panel_tilt-warning', className='text-danger')
			]),
			dbc.Tooltip("Enter solar panel tilt above flat (0 if flat, 90 if vertical)", target='solar_panel_tilt'),

			dbc.Col([
				dcc.Input(id='solar_panel_azimuth', type='number', placeholder='Solar Panel Azimuth (Degrees)', className='form-control'),
				html.Div(id='solar_panel_azimuth-warning', className='text-danger')
			]),
			dbc.Tooltip("Enter solar panel azimuthal angle in degrees from due north (0 if facing north)", target='solar_panel_azimuth'),

			dbc.Col([
				dcc.Input(id='solar_panel_efficiency', type='number', placeholder='Panel Efficiency', className='form-control'),
				html.Div(id='solar_panel_efficiency-warning', className='text-danger')
			]),
			dbc.Tooltip("Enter solar panel efficiency (from 0 to 1)", target='solar_panel_efficiency'),
		], className='mb-3'),

		dbc.Row(id='solar_row_2', style={'display': 'none'}, children=[
			dbc.Col([
				dcc.Input(id='power_consumption', type='number', placeholder='Power Consumption [W]', className='form-control'),
				html.Div(id='power_consumption-warning', className='text-danger')
			]),
			dbc.Tooltip("Enter system power consumption in W", target='power_consumption'),

			dbc.Col([
				dcc.Input(id='capacity', type='number', placeholder='Battery Capacity (w*h)', className='form-control'),
				html.Div(id='capacity-warning', className='text-danger')
			]),
			dbc.Tooltip("Enter battery capacity in watt-hours", target='capacity'),

			dbc.Col([
				dcc.Input(id='battery_rating', type='number', placeholder='Battery Rated Efficiency', className='form-control'),
				html.Div(id='battery_rating-warning', className='text-danger')
			]),
			dbc.Tooltip("Enter battery rated efficiency (from 0 to 1, amount of capacity usable)", target='battery_rating'),

			dbc.Col(dbc.Checkbox(id='box_shading', value=False, label='Sensor Shaded by Panel')),
			dbc.Tooltip("Is the sensor and enclosure shaded by the solar panel? ", target='box_shading'),

		], className='mb-3'),

	], className='mb-3'),

	dbc.Row([
		dbc.Col(dbc.Button('Run', color="primary", id='run-button', n_clicks=0, className='button-margin')),
		dbc.Tooltip("Run the simulation", target='run-button'),
		dbc.Col(dbc.Button('Cancel', color="warning", id='cancel-button', n_clicks=0, className='button-margin')),
		dbc.Tooltip("Cancel the simulation", target='cancel-button'),
		dbc.Col(dbc.Button('Download Files', id='download-button', n_clicks=0, className='button-margin')),
		dbc.Tooltip("Download the results", target='download-button'),
		# dbc.Col(dbc.Button('Update Log', id='log-button', n_clicks=0, className='button-margin')),
		# dbc.Tooltip("Refresh the log console", target='log-button'),

	], className='mb-3 justify-content-center'),

	dbc.Row([
		dbc.Col(dcc.Textarea(id='log-output', value='', className='form-control',
				style={
					'font-family': 'monospace',
					'width': '100%',
					'height': '400px',
					'resize': 'both',
					'overflow': 'auto',
					'display': 'block',
					'minHeight': '400px',
					'maxHeight': '400px',
					'maxWidth': '100%',
					'border': '1px solid #ccc',
					'fontSize': '10px'
				})),
		dbc.Tooltip("Log output will be displayed here", target='log-output'),
		dbc.Col(dcc.Graph(id='figure',
				style={'width': '100%', 'height': '400px', 'border': '1px solid #ccc', 'border-radius': '5px', 'padding': '3px'})),
		dbc.Tooltip("Graphical representation of the results", target='figure'),
	], className='mb-3'),

	dbc.Row([
		dbc.Col(id='solar_figure_row', style={'display': 'none'}, children=[
			dcc.Graph(id='solar_figure', style={'width': '100%', 'height': '400px', 'border': '1px solid #ccc', 'border-radius': '5px', 'padding': '3px'})
			]),
		dbc.Tooltip("Graphical representation of the results", target='solar_figure'),
		
		dbc.Col(id='solar_figure_row2', style={'display': 'none'}, children=[
			dcc.Graph(id='solar_failure_figure', style={'width': '100%', 'height': '400px', 'border': '1px solid #ccc', 'border-radius': '5px', 'padding': '3px'}),		
		]),

		dbc.Col([dcc.Graph(id='failure_chart',
				style={'width': '100%', 'height': '400px', 'border': '1px solid #ccc', 'border-radius': '5px', 'padding': '3px'})]),
		dbc.Tooltip("Graphical representation of the results", target='failure-chart'),
	], className='mb-3'),

	#dcc.Interval(id='interval-component', interval=100, n_intervals=0),
	html.Div(id='dummy-output', style={'display': 'none'}),
	dcc.Interval(
        id='log_interval',
        interval=100,  # in milliseconds
        n_intervals=0  # Number of intervals passed
    ),
	dcc.Download(id='download-data'),
])

# JavaScript code to generate a unique tab ID and store it in sessionStorage
app.clientside_callback(
    """
    function() {
        // Check if tabID is not already in sessionStorage
        if (!sessionStorage.getItem('tabID')) {
            // Generate and store a unique tab ID
            var tabID = 'tab-' + Math.random().toString(36).substr(2, 9);
            sessionStorage.setItem('tabID', tabID);
        }
        // Return the unique tab ID
        return sessionStorage.getItem('tabID');
    }
    """,
    Output('unique-tab-id', 'data'),
    Input('run-button', 'n_clicks')
)

# callback to handle input bounding
@app.callback(
	[Output(f"{input_id}-warning", 'children') for input_id in input_bounds.keys()] +\
	[Output(f"{input_id}-warning", 'className') for input_id in input_bounds.keys()] +\
	[Output('error-store', 'data')],
	[Input(input_id, 'value') for input_id in input_bounds.keys()]
	)
def check_input_bounds(*args):
	"""
	Prevents users from inpuing physically impossible or unlikely values

	Parameters:
	input_ids for each input in the input_bounds dictionary

	Returns:
	warnings - list of warning text for each input (empty string if no warning)
	classes - class of text for each input (error, warning, normal)
	error - boolean storing whether or not one of the inputs outside of error thresholds 
	"""

	warnings = []
	classes = []
	error = False
	
	# iterate through the dictionary of inputs
	for idx, value in enumerate(args):
		input_id = list(input_bounds.keys())[idx]
		if value is not None:

			# Check if value within allowed range (to avoid error, which won't allow the function to run)
			if not (input_bounds[input_id]['allowed'][0] <= value <= input_bounds[input_id]['allowed'][1]):
				warnings.append(f"Allowed bounds {input_bounds[input_id]['allowed']}.")
				classes.append('error-text')
				error = True

			# Check if value within warning range, which would produce warning but not prevent function from running
			elif not (input_bounds[input_id]['warning'][0] <= value <= input_bounds[input_id]['warning'][1]):
				warnings.append(f"Recommended range {input_bounds[input_id]['warning']}.")
				classes.append('warning-text')
			else:
				warnings.append("")
				classes.append('')
		else:
			warnings.append("")
			classes.append('')
	return warnings + classes + [error]

# call back to show other solar items
@app.callback(
    Output('solar_row_1', 'style'),
	Output('solar_row_2', 'style'),
	Output('solar_figure_row', 'style'),
	Output('solar_figure_row2', 'style'),
    Input('solar', 'value')
)
def toggle_inputs(solar):
	"""
	Toggle whether solar inputs are visible

	Parameters:
	solar - bool containing whether or not running solar checkbox is checked

	Returns:
	styles - dictionary for solar div or row styles 
	"""

	if solar == True:
		return {'display': 'flex'}, {'display': 'flex'},  {'display': 'flex'}, {'display': 'flex'}
	else:
		return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

def consolidate_ranges(ranges):
	"""
	Consolidate ranges of time range tuples in case any overlap

	Parameters:
	ranges - list of tuples containing start and end time strings

	Returns:
	ranges - consolidated list of tuples containing start and end time strings
	"""

	if not ranges:
		return []

	# Sort ranges by start time
	ranges = sorted(ranges, key=lambda x: x[0])

	# initialize list of consolidated ranges, starting with the first range
	consolidated = [ranges[0]]

	# iterate through the rest of the ranges
	for current in ranges[1:]:

		# get last time range in consolidated list
		last = consolidated[-1]

		# Check for overlap or contiguous ranges
		if current[0] <= last[1]:
			consolidated[-1] = (last[0], max(last[1], current[1])) # combine into a single range
		else:
			consolidated.append(current) # otherwise add the range to the consolidated list of ranges
	return consolidated

# Callback to add or remove range sliders
@app.callback(
    Output('range-selectors-container', 'children'),
    Output('output-range-values', 'children'),
    Input('add-range-selector', 'n_clicks'),
    Input('remove-range-selector', 'n_clicks'),
    State('range-selectors-container', 'children'),
    State('output-range-values', 'children'),
    prevent_initial_call=True
)
def update_range_sliders(add_clicks, remove_clicks, range_sliders, output_values):
	"""
	Update range sliders when button pressed

	Parameters:
	add_clicks - button press callback to add rangeslider
	remove_clicks - button press callback to remove rangeslider
	range_sliders - list of the range sliders for selecting time periods
	output_values - list of the descriptions of time ranges in from range_sliders 

	Returns:
	range_sliders - list of the range sliders for selecting time periods
	output_values - list of the descriptions of time ranges in from range_sliders 
	"""

	ctx = dash.callback_context # get callback context

	if not ctx.triggered:
		raise dash.exceptions.PreventUpdate #make sure callback was triggered

	button_id = ctx.triggered[0]['prop_id'].split('.')[0] # get button id from callback


	# Generate marks for every hour
	marks = {i * 60: f'{i:02d}:00' for i in range(25)}

	# add range slider
	if button_id == 'add-range-selector':
		range_sliders = range_sliders or []
		output_values = output_values or []
		index = len(range_sliders)
		range_sliders.append( # add div object with rangeslider and dynamic text output inside
			html.Div(
				children=[
					dcc.RangeSlider(
						id={'type': 'dynamic-range-slider', 'index': index},
						min=0,
						max=24*60,  # Total minutes in a day
						step=5,  # 5-minute intervals
						value=[480, 1020],  # Default to 8:00 AM - 5:00 PM
						marks=marks
					),
					html.Div(id={'type': 'dynamic-output', 'index': index})
				]
			)
		)
		output_values.append(html.Div(id={'type': 'dynamic-output-value', 'index': index})) # add

	# remove range slider and text description 
	elif button_id == 'remove-range-selector' and range_sliders:
		range_sliders.pop()
		output_values.pop()

	return range_sliders, output_values

# Callback to update the output values dynamically and store datetimes
@app.callback(
    Output({'type': 'dynamic-output', 'index': dash.dependencies.ALL}, 'children'),
	Output('shading_ranges', 'data'),
    Input({'type': 'dynamic-range-slider', 'index': dash.dependencies.ALL}, 'value'),
    Input({'type': 'dynamic-range-slider', 'index': dash.dependencies.ALL}, 'drag_value'),
    prevent_initial_call=True
)
def display_selected_ranges(ranges, drag_ranges):
	"""
	Update range sliders when range slider dragged

	Parameters:
	ranges - list of range sliders
	drag_ranges - range slider dragged callback

	Returns:
	range_sliders_text_descriptions - list of strings describing times chosen by range sliders
	consolidated - consolidated list of time range tuples
	"""

	all_datetimes = [] # list of datetimes
	
	# ensure range was dragged and callback occurred
	if not drag_ranges:
		raise dash.exceptions.PreventUpdate
	
	# iterate through drag ranges and add tuple of time ranges to all_datetimes list 
	for value in drag_ranges:
		start = timedelta(minutes=value[0]) # start time in minutes
		end = timedelta(minutes=value[1]) # end time
		start_time = (datetime.min + start).time() # 
		end_time = (datetime.min + end).time()
		all_datetimes.append((start_time, end_time))

	# Consolidate overlapping datetime ranges
	consolidated = consolidate_ranges(all_datetimes)

	return [f'Selected range: {start_time.strftime("%H:%M")} - {end_time.strftime("%H:%M")}' for start_time, end_time in all_datetimes], consolidated


#callback to handle "other" option for materials dropdown
@app.callback(
    Output('other-inputs', 'style'),
    Input('material-dropdown', 'value')
)
def display_other_inputs(selected_material):
	"""
	Update material selection input options

	Parameters:
	selected material - material selected in dropdown option

	Returns:
	style of row containing the material inputo ptions 
	"""

	if selected_material == 'Other':
		return {}
	else:
		return {'display': 'none'}

# Function to start the long process
def start_long_function(stop_event, logger, user_id, log_file_path, latitude, longitude, battery_capacity, battery_threshold, length, width, height, thickness, fan_flow, heat_gen, fan_heat_gen, threshold, start_date_time, end_date_time, material, bounds, 
solar = False, solar_panel_area = 0, solar_panel_tilt = 0, solar_panel_azimuth = 0, solar_panel_efficiency = 0, power_consumption = 0, battery_rating = 0.8, batt_min_threshold = 0, box_shading = False, shading_ranges = []):
	"""
	Start thread to run model

	Parameters:
	logger - logging temporary file instance for outputting text to app console
	latitude - the user input latitude for running the model
	longitude - the user input longitude for running the model
	battery capacity - W*H the battery can hold (user input)
	battery threshold - maximum battery temperature allowed (user input) [C]
	length - enclosure length (user input) [m]
	width - enclosure width (user input) [m]
	height - enclosure height (user input) [m]
	thickness - enclosure thickness (user input) [m]
	fan_flow - fan volumetric flow (user input) [CFM]
	heat_gen - internal heat generation from things in enclosure (user input) [W]
	fan_heat_gen - internal heat generation from the fan specifically (user input) [W]
	threshold - temperature threshold for fan to turn on (user input) [C]
	start_date_time - datetime string for starting the HT model run period (user input)
	end_date_time - datetime string for ending the HT model run period (user input)
	stop_event - the threading stop event for stopping execution of the model when cancel button is pressed
	material - dictionary of material properties (user chosen)
	bounds - boolean containing if 10% and 90% historical weather bounds will be run for historical model (user input)
	solar - boolean containing if solar model will be run (user input)
	solar_panel_area - area of solar panel (user input) [m]
	soalar_panel_tilt - tilt of solar panel (user input) [degrees]
	solar_panel_azimuth - orientation of solar panel (degrees clockwise from north) (user input)
	solar_panel_efficiency - efficency of solar panel (user input)
	power_consumption - sensor power consumption (assumed to be constant) (user input) [W]
	battery_rating - true capacity fraction of rated battery capacity (user input)
	batt_min_threshold - minimum operating temperature for battery (user input) [C]
	box_shading - boolean containing weather or not the sensor/enclosure is shaded by the solar panel (user input)
	shading_ranges - list of tuples containing time strings of time periods within a day during which the setup is shaded (user input)
	"""
		
	q = queue.Queue()

	thread = Thread(target=run_model, args=(logger, q, latitude, longitude, battery_capacity, battery_threshold, length, width, height, thickness, fan_flow, heat_gen,fan_heat_gen, threshold, start_date_time, end_date_time, stop_event, material, bounds, solar, solar_panel_area, solar_panel_tilt, solar_panel_azimuth, solar_panel_efficiency, power_consumption, battery_rating, batt_min_threshold, box_shading, shading_ranges))

	processes[user_id] = {'thread': thread, 'stop_event': stop_event, 'temp_file_path': log_file_path}
	thread.start()
	thread.join()  # Wait for the thread to finish
	figs = q.get()
	pickle_dir = q.get()
	clear_queue(q)

	return figs, pickle_dir

# Function to clear the queue
def clear_queue(q):
    while not q.empty():
        q.get()
        q.task_done()  # Mark the item as processed

# Callback to start/stop the function and update log output
@app.callback(
	Output('figure', 'figure'),
	Output('solar_figure', 'figure'),
	Output('failure_chart', 'figure'),
	Output('solar_failure_figure', 'figure'),
	Output('pickle_dir', 'data'),
	Input('unique-tab-id', 'data'),
    Input('run-button', 'n_clicks'),
    Input('cancel-button', 'n_clicks'),
	Input('Latitude', 'value'),
    Input('Longitude', 'value'),
    Input('capacity', 'value'),
	Input('batt_threshold', 'value'),
    Input('length', 'value'),
    Input('width', 'value'),
    Input('height', 'value'),
    Input('thickness', 'value'),
    Input('fan_flow', 'value'),
    Input('heat_gen', 'value'),
	Input('fan_heat_gen', 'value'),
    Input('threshold', 'value'),
	Input('Date Range', 'start_date'),
    Input('Date Range', 'end_date'),
	Input('material-dropdown', 'value'),
	Input('bounds', 'value'),
	Input('solar', 'value'),
	Input('solar_panel_area', 'value'),
	Input('solar_panel_tilt', 'value'),
	Input('solar_panel_azimuth', 'value'),
	Input('solar_panel_efficiency', 'value'),
	Input('power_consumption', 'value'),
	Input('battery_rating', 'value'),
	Input('batt_min_threshold', 'value'),
	Input('error-store', 'data'),
	Input('box_shading', 'value'),
	Input('shading_ranges', 'data'),
	State('input-emissivity', 'value'),
    State('input-cp', 'value'),
    State('input-density', 'value'),
    State('input-thermal-conductivity', 'value'),
    prevent_initial_call=True
)
def run_or_cancel_function(tab_id, run_clicks, cancel_clicks, latitude, longitude, battery_capacity, battery_threshold, length, width, height, thickness, fan_flow, heat_gen, fan_heat_gen, threshold, start_date_time, end_date_time, material_name, bounds, 
						solar, solar_panel_area, solar_panel_tilt, solar_panel_azimuth, solar_panel_efficiency, power_consumption, battery_rating, batt_min_threshold, error_store, box_shading, shading_ranges, input_emissivity, input_cp, input_density, input_k):

	"""
	Run or cancel model

	Parameters:
	tab_id - unique tab id for the session instance
	run_clicks - run button clicked callback
	cancel_clicks - cancel button clicked callback
	latitude - the user input latitude for running the model
	longitude - the user input longitude for running the model
	battery capacity - W*H the battery can hold (user input)
	battery threshold - maximum battery temperature allowed (user input) [C]
	length - enclosure length (user input) [m]
	width - enclosure width (user input) [m]
	height - enclosure height (user input) [m]
	thickness - enclosure thickness (user input) [m]
	fan_flow - fan volumetric flow (user input) [CFM]
	heat_gen - internal heat generation from things in enclosure (user input) [W]
	fan_heat_gen - internal heat generation from the fan specifically (user input) [W]
	threshold - temperature threshold for fan to turn on (user input) [C]
	start_date_time - datetime string for starting the HT model run period (user input)
	end_date_time - datetime string for ending the HT model run period (user input)
	stop_event - the threading stop event for stopping execution of the model when cancel button is pressed
	material - dictionary of material properties (user chosen)
	bounds - boolean containing if 10% and 90% historical weather bounds will be run for historical model (user input)
	solar - boolean containing if solar model will be run (user input)
	solar_panel_area - area of solar panel (user input) [m]
	soalar_panel_tilt - tilt of solar panel (user input) [degrees]
	solar_panel_azimuth - orientation of solar panel (degrees clockwise from north) (user input)
	solar_panel_efficiency - efficency of solar panel (user input)
	power_consumption - sensor power consumption (assumed to be constant) (user input) [W]
	battery_rating - true capacity fraction of rated battery capacity (user input)
	batt_min_threshold - minimum operating temperature for battery (user input) [C]
	box_shading - boolean containing weather or not the sensor/enclosure is shaded by the solar panel (user input)
	shading_ranges - list of tuples containing time strings of time periods within a day during which the setup is shaded (user input)
	input_emissivity - manual input material emissivity
	input_cp - manual input material specific heat capacity
	input_density - manual input material density
	input_k - manual input material thermal conductivity

	Returns:
	figure - heat transfer model output figure of temperatures
	solar figure - solar battery charge output figure
	failure_chart - hourly failure rate bar plot
	solar_failure_chart - heatmap of solar failure rates by hour by month
	"""

	# get call back context to ensure button was clicked
	ctx = dash.callback_context
	if not ctx.triggered:
		return dash.no_update, dash.no_update, dash.no_update,  dash.no_update,  dash.no_update

	triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
	user_id = tab_id  # Use the tab ID as the user ID¸

	# get datetime start and end from the date range dash input
	start_date_time = start_date_time.split('T')[0] + ' 00:00:00'
	end_date_time = end_date_time.split('T')[0] + ' 23:59:59'

	# if the start button is pressed
	if triggered_id == 'run-button':

		# if already running
		if user_id in processes and processes[user_id]['thread'].is_alive(): 
			return dash.no_update, dash.no_update, dash.no_update,  dash.no_update,  dash.no_update

		logger, log_file_path = setup_unique_logger(user_id)  # Set up a unique logger for this session
		stop_event = Event()

		if error_store:
			logger.info("Fix inputs before running.")
			return dash.no_update, dash.no_update, dash.no_update,  dash.no_update,  dash.no_update

		# select material
		if material_name == "Other" and input_emissivity is not None and input_cp is not None and input_density is not None and input_k is not None:
			material_properties['Other'] = {'emissivity': input_emissivity, 'Cp': input_cp, 'density': input_density, 'thermal_conductivity': input_k}
			material = material_properties[material_name]
		elif material_name != "Other":
			material = material_properties[material_name]
		else:
			logger.info("Choose and define material first")
			return dash.no_update, dash.no_update, dash.no_update,  dash.no_update,  dash.no_update

		if not solar:
			figs, pickle_dir = start_long_function(
								stop_event = stop_event,
								logger = logger,
								user_id = user_id,
								log_file_path = log_file_path,
								latitude = latitude, 
								longitude = longitude, 
								battery_capacity = battery_capacity,
								battery_threshold = battery_threshold, 
								length = length, 
								width =width, 
								height = height, 
								thickness = thickness, 
								fan_flow = fan_flow, 
								heat_gen = heat_gen, 
								fan_heat_gen = fan_heat_gen,
								threshold = threshold, 
								start_date_time = start_date_time, 
								end_date_time = end_date_time,
								material = material,
								bounds = bounds,
								batt_min_threshold = batt_min_threshold,
								box_shading = box_shading,
								shading_ranges = shading_ranges)
			logger.info("Plotting Figure")
			return figs[0], dash.no_update, figs[1], dash.no_update, pickle_dir
		elif solar:
			figs, pickle_dir = start_long_function(
							stop_event = stop_event,
							logger = logger,
							user_id = user_id,
							log_file_path = log_file_path,
							latitude = latitude, 
							longitude = longitude, 
							battery_capacity = battery_capacity, 
							battery_threshold = battery_threshold, 
							length = length, 
							width =width, 
							height = height, 
							thickness = thickness, 
							fan_flow = fan_flow, 
							heat_gen = heat_gen, 
							fan_heat_gen = fan_heat_gen,
							threshold = threshold, 
							start_date_time = start_date_time, 
							end_date_time = end_date_time,
							material = material,
							bounds = bounds,
							solar = solar, 
							solar_panel_area = solar_panel_area,
							solar_panel_tilt = solar_panel_tilt,
							solar_panel_azimuth = solar_panel_azimuth,
							solar_panel_efficiency = solar_panel_efficiency,
							power_consumption = power_consumption,
							battery_rating = battery_rating,
							batt_min_threshold = batt_min_threshold,
							box_shading = box_shading,
							shading_ranges = shading_ranges)
			logger.info("Plotting Figure")
			return figs[0], figs[1], figs[2], figs[3], pickle_dir

	elif triggered_id == 'cancel-button':
		# Stop button clicked
		if user_id in processes:
			# If a process is running for this user, stop it
			processes[user_id]['stop_event'].set()  # Set the stop event
			processes[user_id]['thread'].join()  # Wait for the thread to finish

			# Delete the temporary file
			os.remove(processes[user_id]['temp_file_path'])

			del processes[user_id]  # Remove the process from the dictionary
		return dash.no_update, dash.no_update, dash.no_update,  dash.no_update, dash.no_update

	return dash.no_update, dash.no_update, dash.no_update,  dash.no_update,  dash.no_update

# Function to read the log file
def read_log_file(log_file_path):
    with open(log_file_path, 'r') as log_file:
        return log_file.read()

# Helper function to create a zip file of pickle files
def create_zip_file(pickle_dir):
	"""
	Create zip file for downloading 

	Parameters:
	pickle_dir - the string directory of the local pickle files

	Returns:
	temp_zip.name - the zipfile name of pickled dataframes
	"""

	try:
		# Create a temporary file to store the zip
		temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)

		# Create a zip file
		with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
			for file_name in os.listdir(pickle_dir):
				file_path = os.path.join(pickle_dir, file_name)
				zipf.write(file_path, arcname=file_name)

		return temp_zip.name
	except Exception as e:
		return 

# Callback to handle button click to download files
@app.callback(
	Output("download-data", "data"),
	Input('download-button', 'n_clicks'),
	Input('pickle_dir', 'data'),
	prevent_initial_call=True
)
def download_files(n_clicks, pickle_dir):
	"""
	Download zipped pickled files when download data button pressed

	Parameters:
	n_clicks - the callback for data download button clicked
	pickle_dir - the directory for the pickle files

	Returns:
	dcc.send_file - object for downloading files
	"""

	if n_clicks:
		zip_file = create_zip_file(pickle_dir) # zip files 
		return dcc.send_file(zip_file, filename='data.zip') # send to downlaod dcc object

# Update output-div content with logfile text
@app.callback(
    Output('log-output', 'value'),
    Input('log_interval', 'n_intervals'),
	State('unique-tab-id', 'data')
)
def update_output_div(n, tab_id):
	if tab_id not in processes:
		return dash.no_update
	else:
		temp_file_path = processes[tab_id]['temp_file_path']
		log_content = read_log_file(temp_file_path)
		return log_content

@app.callback(
    Output('Date Range', 'end_date'),
    [Input('Date Range', 'start_date'),
     Input('Date Range', 'end_date'),
	 Input('material-dropdown', 'value')]
)
def update_end_date(start_date, end_date, material):
	"""
	Call back to ensure dates seleted is max one month\


	Parameters:
	start_date - start date selected with input daterange widget
	end_date - end date selected with input daterange widget
	material - the material chosen

	Returns:
	end_date - updated end_date ensuring date range doesn't surpass maximum time period for model run
	"""

	from dash.exceptions import PreventUpdate
	if start_date is None or end_date is None:
		raise PreventUpdate

	start_date_obj = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d')
	end_date_obj = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d')

	# Calculate duration
	duration = end_date_obj - start_date_obj

	max_days = 30
	if material == "Aluminum" or material == "Stainless Steel":
		max_days = 5

	# Check if duration is less than 30 days
	if duration.days >= max_days:
		end_date = start_date_obj + timedelta(days=max_days)
		return end_date.strftime('%Y-%m-%d')
	else:
		return dash.no_update

if __name__ == '__main__':
	app.run_server(debug=True, port = '7777', use_reloader = False)