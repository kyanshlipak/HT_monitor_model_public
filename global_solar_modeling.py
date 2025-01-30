"""
File Name: solar_simulation_model.py

Description:
This script performs solar power generation and storage simulations across grid points on the globe. 
The simulations include parameters such as solar panel area, tilt, azimuth, efficiency, and battery capacity. 
It generates time-series data for energy production and consumption, logging results for each grid point. 
The output is saved in pickle format for further analysis.

Key Features:
- Simulates solar energy production and battery usage for specified geographic locations.
- Utilizes meteorological data for realistic modeling.
- Generates grid points, filtered for land regions using the Natural Earth dataset.
- Outputs time-series results for solar energy simulations.

Requirements:
- Python 3.x
- Libraries: numpy, pandas, matplotlib, geopandas, shapely, logging
- Natural Earth dataset for land boundaries (`ne_110m_admin_0_countries.shp`).

Usage:
1. Ensure the required libraries are installed (`pip install numpy pandas matplotlib geopandas shapely`).
2. Provide the Natural Earth shapefile in the same directory or update the `shapefile_path` variable.
3. Execute the script to generate solar simulation results across grid points.
4. Output data for each grid point is saved in a directory named by the timestamp.

Author: Kyan Shlipak
Date: 1/6/2025
"""


import numpy as np
import matplotlib.pyplot as plt
from numerical_modelling import *
from datetime import datetime
import pandas as pd


def multiple_solar_simulations(latitude=40.5853, longitude=-105.0844, logging = None,
                             start_solar_time='2023-01-01 00:00:00', 
                             end_solar_time='2023-12-31 23:00:00',
							 solar_parameters = [{
								'simulation_name':'',
								'solar_panel_area': 0.2, 
								'solar_panel_tilt': 0,
								'solar_panel_azimuth': 0,
								'solar_panel_efficiency': 0.18,
								'battery_rated_capacity': 120,
								'battery_efficiency': 0.8,
								'power_consumption': 5,
								'shading_ranges': [],
							 }],
                            dt_solar = 10,
                            stop_event = None,
                            ):
	print(f"*** Modelling Solar ({latitude}, {longitude}) ***")
	start_solar_time = datetime.strptime(start_solar_time, '%Y-%m-%d %H:%M:%S')
	end_solar_time = datetime.strptime(end_solar_time, '%Y-%m-%d %H:%M:%S')
	
	timezone = get_timezone(latitude, longitude)
	df_meterological = get_data(latitude=latitude, 
						longitude=longitude, 
						start_date_time = start_solar_time,
						end_date_time = end_solar_time,
						timezone = timezone,
						logging = logging,
						stop_event = stop_event
	)



	def run_and_output(solar_panel_area, solar_panel_tilt, solar_panel_azimuth, solar_panel_efficiency, battery_rated_capacity, battery_efficiency, shading_ranges, power_out_func):
		
		solar_time_steps, charges, power_ins, power_outs = solar_panel_model(latitude = latitude, longitude = longitude, logging = logging, 
							start_date_time = start_solar_time, end_date_time = end_solar_time, df =df_meterological,
							solar_panel_area = solar_panel_area, solar_panel_tilt = solar_panel_tilt, solar_panel_azimuth = solar_panel_azimuth, 
							solar_panel_efficiency = solar_panel_efficiency, 
							battery_rated_capacity = battery_rated_capacity, battery_efficiency = battery_efficiency, dt = dt_solar,
							power_out_func = power_out_func, stop_event = stop_event, shading_ranges = shading_ranges)

		solar_df = pd.DataFrame({'seconds_past': solar_time_steps})
		solar_df['datetime'] = solar_df['seconds_past'].apply(lambda x: start_solar_time + timedelta(seconds=x))
		solar_df['charges'] = charges
		#solar_df['power_ins'] = power_ins
		#solar_df['power_outs'] = power_outs
		solar_df.set_index('datetime', inplace=True)
		solar_df = solar_df.resample('30min').mean()
		return solar_df
	
	data_output = {}

	for solar_parameter_set in solar_parameters:
		simulation_name = solar_parameter_set['simulation_name']
		solar_panel_area = solar_parameter_set['solar_panel_area']
		solar_panel_tilt = solar_parameter_set['solar_panel_tilt']
		solar_panel_azimuth = solar_parameter_set['solar_panel_azimuth']
		solar_panel_efficiency = solar_parameter_set['solar_panel_efficiency']
		battery_rated_capacity = solar_parameter_set['battery_rated_capacity']
		battery_efficiency = solar_parameter_set['battery_efficiency']
		power_consumption = solar_parameter_set['power_consumption']
		shading_ranges = solar_parameter_set['shading_ranges']
		
		def power_out_func(*args, **kwargs):
			return power_consumption
		
		logging.info(f"Modelling at ({round(latitude,4)}, {round(longitude, 4)})")

		try: 
			solar_df = run_and_output(solar_panel_area = solar_panel_area, 
									 solar_panel_tilt = solar_panel_tilt, 
									 solar_panel_azimuth = solar_panel_azimuth, 
									 solar_panel_efficiency = solar_panel_efficiency, 
									 battery_rated_capacity = battery_rated_capacity, 
									 battery_efficiency = battery_efficiency, 
									 shading_ranges = shading_ranges, 
									 power_out_func = power_out_func)
			data_output[simulation_name] = solar_parameter_set
			data_output[simulation_name]['data'] = solar_df
		except Exception as e:
			logging.error("Error running model" + str(e))
			data_output[simulation_name]['data'] = False

	return data_output
	

def generate_grid_points(latitude_step = 2, 
						 longitude_step = 2, 
						 latitude_bounds = [-180, 180], 
						 longitude_bounds = [-180, 180]):

	import numpy as np
	import geopandas as gpd
	from shapely.geometry import Point

	# Load Natural Earth dataset for land boundaries
	shapefile_path = "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
	world = gpd.read_file(shapefile_path)
	#world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
	africa = world.loc[(world['CONTINENT'] != 'Seven seas (open ocean)') & (world['CONTINENT'] != 'Antarctica')]

	# Define grid resolution
	latitude_step = 2
	longitude_step = 2

	# Define bounds for Africa
	min_latitude, max_latitude = latitude_bounds
	min_longitude, max_longitude = longitude_bounds

	# Generate grid
	latitudes = np.arange(min_latitude, max_latitude + latitude_step, latitude_step)
	longitudes = np.arange(min_longitude, max_longitude + longitude_step, longitude_step)

	# Create grid points
	grid_points = [Point(lon, lat) for lat in latitudes for lon in longitudes]

	# Filter for points within Africa land boundaries
	grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs='EPSG:4326')
	land_points = grid_gdf[grid_gdf.within(africa.unary_union)]


	from shapely.geometry import Point
	import matplotlib.pyplot as plt

	# Plot the map
	fig, ax = plt.subplots(figsize=(12, 12))
	africa.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)
	land_points.plot(ax=ax, color='blue', markersize=1, label='Grid Points')

	# Customize plot
	plt.title('Grid Points Over Land Only', fontsize=16)
	plt.xlabel('Longitude', fontsize=12)
	plt.ylabel('Latitude', fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend(loc='lower left')
	#plt.show()

	return land_points


land_points = generate_grid_points()

import logging
from datetime import datetime
import os

time = datetime.strftime(datetime.now(), "%m_%d_%H_%M")
dir_name = f'global_solar_modeling_2_degrees_{time}'
os.mkdir(dir_name)

# Initialize the logger
logging.basicConfig(
    filename='global_solar_model.log',  # Name of the log file
    level=logging.INFO,                   # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'           # Time format
)

# Create logger instance
logger = logging.getLogger()

	
for i in land_points.geometry[:]:
	lon = int(str(i).split(' ')[1][1:])
	lat = int(str(i).split(' ')[2][:-1])
	
	results = multiple_solar_simulations(latitude=lat, longitude=lon, logging = logger,
							solar_parameters = [{
								'simulation_name':'base',
								'solar_panel_area': 0.5, 
								'solar_panel_tilt': 0,
								'solar_panel_azimuth': 0,
								'solar_panel_efficiency': 0.18,
								'battery_rated_capacity': 300,
								'battery_efficiency': 0.8,
								'power_consumption': 5,
								'shading_ranges': [],
							}],
							start_solar_time='2023-01-01 00:00:00', 
							end_solar_time='2023-12-31 23:59:59',
							dt_solar = 10,
							stop_event = None,
							)
	
	name = f'{dir_name}/{lat}_{lon}.pkl'
	results['base']['data'].to_pickle(name)