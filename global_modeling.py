import numpy as np
import matplotlib.pyplot as plt
from numerical_modelling import *
from datetime import datetime
import pandas as pd

def multiple_temperature_simulations(latitude=40.5853, longitude=-105.0844, logging = None,
                             T_max=45 + 273.15, T_initial=20 + 273.15, 
                             B_air=0.00367, v_air=15.89e-6, k_air=0.024, alpha_air=22.5e-6, h_max=200, N=4, 
							 box_parameters = [{
								'simulation_name': "basic_ABS",
								'L':0.2032,
								'W':0.152, 
								'H':0.10922,
								'thicknesses':[0.004, 0.004, 0.004, 0.004, 0.004, 0.004],
								'fan_flow': 0, 
								'heat_generation': 0, 
								'fan_heat_generation': 0, 
								'fan_threshold': 40, 
								'box_shading': False,
								'shading_ranges': [],
								'k': 0.18, 
								'rho': 1070, 
								'Cp': 2000, 
								'emissivity': 0.66, 
								'absorptivity': 0.66, 
							 }],
                             start_date_time='2023-01-01 00:00:00', 
                             end_date_time='2023-12-31 23:00:00',
                             stop_event = None,
							):


	def run_and_output(data_2020, fan_speed_func, constant_q_flux,
					k, rho, Cp, emissivity, absorptivity, thicknesses, L, W, H, N, box_shading, shading_ranges):
		time_steps, _, avg_temp, _, _, outside_temp = \
			run_model_reduced_complexity_and_memory(latitude, longitude, logging, 30, 7.5, start_date_time, data_2020, 
										fan_speed_func, air_velocity_func, avg_T, angle_func, constant_q_flux, h_external, 
										t_final, T_max, T_infinity_func, T_initial, 
										k, rho, Cp, G_func, emissivity, absorptivity, B_air, v_air, k_air, alpha_air, 
										thicknesses = thicknesses, L = L, W =W, H=H,N= N,h_max = h_max, stop_event = stop_event, 
										box_shading = box_shading, shading_ranges = shading_ranges)

		base_time = start_date_time
		model_df = pd.DataFrame({'seconds_past': time_steps})


		# Convert seconds_past to Timedelta and add to base_time
		model_df['datetime'] = model_df['seconds_past'].apply(lambda x: base_time + timedelta(seconds=x))
		del model_df['seconds_past']
		model_df['outside_temp'] = outside_temp
		model_df['internal'] = avg_temp

		model_df.set_index('datetime', inplace=True)

		# save dataframes to pickles
		model_df_ten_min = model_df.resample('30min').mean()
		return model_df_ten_min

	start_date_time = datetime.strptime(start_date_time, '%Y-%m-%d %H:%M:%S')
	end_date_time = datetime.strptime(end_date_time, '%Y-%m-%d %H:%M:%S')

	timezone = get_timezone(latitude, longitude)
	df_mean = get_data(latitude=latitude, 
													longitude=longitude, 
													start_date_time = start_date_time,
													end_date_time = end_date_time,
													timezone = timezone,
													logging = logging,
													stop_event = stop_event
	)

	data_output = {}

	for box_parameter_set in box_parameters:
		simulation_name = box_parameter_set['simulation_name']
		L = box_parameter_set['L']
		W = box_parameter_set['W']
		H = box_parameter_set['H']
		thicknesses = box_parameter_set['thicknesses']
		fan_flow = box_parameter_set['fan_flow']
		heat_generation = box_parameter_set['heat_generation']
		fan_heat_generation = box_parameter_set['fan_heat_generation']
		fan_threshold = box_parameter_set['fan_threshold']
		box_shading = box_parameter_set['box_shading']
		shading_ranges = box_parameter_set['shading_ranges']
		k = box_parameter_set['k']
		rho = box_parameter_set['rho']
		Cp = box_parameter_set['Cp']
		emissivity = box_parameter_set['emissivity']
		absorptivity = box_parameter_set['absorptivity']

		logging.info(f"Modelling {simulation_name} with at ({round(latitude,4)}, {round(longitude, 4)})")

		### Calculated Params
		fan_flow_m3s = fan_flow * 0.0004719471999802417
		fan_speed = fan_flow_m3s / np.mean([L*W, L*H, W*H])
		t_final = (end_date_time - start_date_time).total_seconds()
		
		def constant_q_flux(T_internal): 
			if T_internal >= (fan_threshold + 273.15):
				return heat_generation + fan_heat_generation # from fan assembly
			else:
				return heat_generation

		def fan_speed_func(T_internal):
			if T_internal > (fan_threshold + 273.15):
				return fan_speed
			else:
				return 0

		try: 
			model_df_ten_min = run_and_output(df_mean, fan_speed_func = fan_speed_func, constant_q_flux = constant_q_flux,
									 k = k, rho = rho, Cp = Cp, emissivity = emissivity, absorptivity=absorptivity, thicknesses=thicknesses,
									 L = L, W= W, H= H, N = N, box_shading=box_shading, shading_ranges=shading_ranges)
			data_output[simulation_name] = box_parameter_set
			data_output[simulation_name]['data'] = model_df_ten_min
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
dir_name = f'global_modeling_2_degrees_{time}'
os.mkdir(dir_name)

# Initialize the logger
logging.basicConfig(
    filename='global_temp_model.log',  # Name of the log file
    level=logging.INFO,                   # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'           # Time format
)

# Create logger instance
logger = logging.getLogger()


for i in land_points.geometry:
	lon = int(str(i).split(' ')[1][1:])
	lat = int(str(i).split(' ')[2][:-1])
	
	results = multiple_temperature_simulations(latitude=lat, longitude=lon, logging = logger,
                             T_max=45 + 273.15, T_initial=20 + 273.15, 
                             B_air=0.00367, v_air=15.89e-6, k_air=0.024, alpha_air=22.5e-6, h_max=200, N=4, 
							 box_parameters = [
								 {
								'simulation_name': "no_shading",
								'L':0.2032,
								'W':0.152, 
								'H':0.10922,
								'thicknesses':[0.006, 0.006, 0.006, 0.006, 0.006, 0.006],
								'fan_flow': 0, 
								'heat_generation': 5, 
								'fan_heat_generation': 0, 
								'fan_threshold': 40, 
								'box_shading': False,
								'shading_ranges': [],
								'k': 0.162, 
								'rho': 1070, 
								'Cp': 1990, 
								'emissivity': 0.94, 
								'absorptivity': 0.94, 
							 },
								{
								'simulation_name': "solar_shield",
								'L':0.2032,
								'W':0.152, 
								'H':0.10922,
								'thicknesses':[0.006, 0.006, 0.006, 0.006, 0.006, 0.006],
								'fan_flow': 0, 
								'heat_generation': 5, 
								'fan_heat_generation': 0, 
								'fan_threshold': 40, 
								'box_shading': True,
								'shading_ranges': [],
								'k': 0.162, 
								'rho': 1070, 
								'Cp': 1990, 
								'emissivity': 0.94, 
								'absorptivity': 0.94, 
							 },
							 								 {
								'simulation_name': "full_shade",
								'L':0.2032,
								'W':0.152, 
								'H':0.10922,
								'thicknesses':[0.006, 0.006, 0.006, 0.006, 0.006, 0.006],
								'fan_flow': 0, 
								'heat_generation': 5, 
								'fan_heat_generation': 0, 
								'fan_threshold': 40, 
								'box_shading': False,
								'shading_ranges': [('00:00:00', '23:59:59')],
								'k': 0.162, 
								'rho': 1070, 
								'Cp': 1990, 
								'emissivity': 0.94, 
								'absorptivity': 0.94, 
							 }
							 ],
                             start_date_time='2023-01-01 00:00:00', 
                             end_date_time='2023-12-31 23:59:00',
                             stop_event = None,
							)
	
	sub_dir = f'{dir_name}/{lat}_{lon}'
	os.mkdir(sub_dir)

	results['no_shading']['data'].to_pickle(sub_dir + '/no_shading.pkl')
	results['solar_shield']['data'].to_pickle(sub_dir + '/solar_shield.pkl')
	results['full_shade']['data'].to_pickle(sub_dir + '/full_shade.pkl')

