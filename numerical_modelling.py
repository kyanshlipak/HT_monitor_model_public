"""
numerical_modelling.py

Main python file containing functions for running models, including the numerical heat transfer and solar models themselves

Author: Kyan Shlipak
Date: 09/28
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from timezonefinder import TimezoneFinder
import pytz
import ephem
from datetime import datetime, timedelta
import requests

sigma = 5.67 * 10 ** (-8)

def free_convection(T_surface, T_infinity, area, perimeter, horizontal, v_air, Beta, alpha_air, upper):
  """
  Calculate the free convection Nusselt number
  
  Parameters:
  T_surface - surface temperature of the face in K
  T_infinity - temperature of the surrounding air in K
  area - area of the face in m^2
  perimeter - the perimeter of the face in m
  horizontal - boolean if the plate is horizontal (for buoyance purposes) as upposed to upright and vertical
  v_air - kinematic viscosity of the air 
  Beta - thermal expansion coefficient of the air
  alpha_air - thermal diffusivity of the air
  upper - if the plate is horizontal, whether the air is above or below the plate

  Returns:
  Nu_free - Nusselt number for free convection case
  Gr - Graschof number (dimensionless ratio of buoyancy forces to viscous forces in a fluid)
  """

  # convection correlation for free convection of a vertical plate
  def vertical_plate_correlations():
    #https://volupe.se/empirical-correlations-for-convective-heat-transfer-coefficients/
    if Ra <= 10**9:
      Nu = 0.68 + (0.67 * Ra ** 0.25) / (1 + (0.492/Pr)**(9/16) ) ** (4/9) #correlation with higher accuracy for just laminar
    else:
      Nu = (0.825 + 0.387 * Ra ** (1/6) / (1 + (0.492/Pr)**(9/16) )**8/27)**2  #correlation for laminar or turbulent
    return Nu
    
  # convection correlations for free convection of a horizontal plate
  def horizontal_plate_correlations():
    #https://volupe.se/empirical-correlations-for-convective-heat-transfer-coefficients/
    if (T_surface > T_infinity and upper == True) or (T_surface < T_infinity and upper == False): # upper surface of hot plate or lower surface of cold plate
      if Ra <= 10**7:
        Nu = 0.54 * Ra ** 0.25 # laminar correlation
      else:
        Nu = 0.15 * Ra ** (1/3) # turbulent correlation
    else: # upper surface of cold plate or lower surface of hot plate
      Nu = 0.27 * Ra ** 0.25 #combined correlation
    return Nu
    
  Pr = v_air/alpha_air
  Gr = 9.8 * Beta * abs(T_surface - T_infinity) * (area/perimeter) ** 3 / v_air**2 #free convection meaurement of buoyant flows
  Ra = Gr*Pr # Gr*Pr
  
  if horizontal: Nu_free = horizontal_plate_correlations()
  else: Nu_free = vertical_plate_correlations()

  return Nu_free, Gr

def h_external(T_surface, T_infinity, characteristic_length, area, perimeter, horizontal, v_air, Beta, alpha_air, k_air, velocity, upper):
  """
  Calculate the convection coefficient for a plate on the outside of the sensor enclosure
  
  Parameters:
  T_surface - surface temperature of the face in K
  T_infinity - temperature of the surrounding air in K
  characteristic_length - characteristic length of the face in m
  area - area of the face in m^2
  perimeter - the perimeter of the face in m
  horizontal - boolean if the plate is horizontal (for buoyance purposes) as upposed to upright and vertical
  v_air - kinematic viscosity of the air 
  Beta - thermal expansion coefficient of the air
  alpha_air - thermal diffusivity of the air
  k_air - thermal conductivity of the air
  velocity - speed of the air flow in m/s
  upper - if the plate is horizontal, whether the air is above or below the plate

  Returns:
  h - convection coefficient for the plate in W/m^2 K
  """


  def forced_correlation(): # assumes each is a flat plate because we can't really calculate orientation w.r.t. wind
    Re_critical = 5 * 10**5 # turbulent flow for Re_L
    if Re < Re_critical:
      Nu = 0.664 * Re ** 0.5 * Pr ** (1/3)
    else:
      Nu = (0.664 * Re_critical ** 0.5 + 0.037 * (Re ** 0.8 - Re_critical ** 0.8)) * Pr ** (1/3)
    return Nu

  Pr = v_air/alpha_air
  Re = velocity * characteristic_length / v_air #forced convectino measurement of inertial flows

  # find forced convection coefficient
  Nu_forced = forced_correlation()
  h_forced = Nu_forced * k_air / characteristic_length

  Nu_free, Gr = free_convection(T_surface, T_infinity, area, perimeter, horizontal, v_air, Beta, alpha_air, upper)
  
  # calculate h_free free convection coefficient
  if horizontal: 
    h_free =  Nu_free * k_air / (area/perimeter) # horizontal correlations have this specific characteristic length
  else:
    h_free = Nu_free * k_air / characteristic_length

  # determine which type of convection is dominant
  if Re == 0:
    dominance = 100000000
  else:
    dominance = Gr / Re**2 

  if dominance < 0.1: # forced convection dominant
    return h_forced
  elif dominance > 10: # free convection dominant
    return h_free
  else:
    # calcualte mixed convection Nu for mixed dominance case
    n = 3
    h_mixed = (h_free ** n + h_forced ** n) ** (1/n) # combined / mixed case empirical relation (approximation)
    return h_mixed

def stability_criterion(thickness, N, h_max, k, alpha, emissivity, T_max, max_q_flux):
  """
  Calculate the minimum timestep required for stable finite difference method modelling
  
  Parameters:
  thickness - the thickness of the enclosure wall in m
  N - number of discrete intervals modelled through the thickness of the wall
  h_max - maximum expected convection coefficient
  k - thermal conductivity of the material
  alpha_ABS - thermal diffusivity of the material
  emissivity - emissivity of the material
  T_max - the maximum expected temperature

  Returns:
  dt - the timestep for the finite difference methods numerical model
  """

  sigma = 5.67 * 10**(-8)
  dx = thickness / (N - 1)  # Grid spacing
  Bi = h_max*dx/k

  # this can be derived by hand based on finite difference methods equation
  dt =  0.5 * dx**2 / alpha / (1 + Bi + dx/k * emissivity * sigma * T_max**3)/1.25 #dt has to be under this number, safety facture 1.2, external side
  return dt

def hydraulic_diameter(length_1, length_2):
    """
    Calculate the hydraulic diameter of a 2d rectangular surface given both lenghts

    Parameters:
    length_1 - length in meters
    length_2 - length in meters

    Returns:
    hydraulic diameter
    """
    return 2 * length_1 * length_2 / (length_1 + length_2) # 4 * A / P

def internal_convective_heat_transfer_func(fan_speed, v_air, k_air, alpha_air, perp_length1, perp_length2, T_surface, T_infinity, characteristic_length, area, perimeter, horizontal, Beta, upper):

  """
  Calculate the convection coefficient for a plate on the inside of the sensor enclosure
  
  Parameters:
  fan_speed - approximate speed of air flow in m/s
  v_air - kinematic viscosity of the air 
  k_air - thermal conductivity of the air
  alpha_air - thermal diffusivity of the air
  L - length of enclosure in m
  W - width of the enclosure in m
  H - height of the enclosure in m
  T_surface - surface temperature of the face in K
  T_infinity - temperature of the surrounding air in K
  characteristic_length - characteristic length of the face in m
  area - area of the face in m^2
  perimeter - the perimeter of the face in m
  horizontal - boolean if the plate is horizontal (for buoyance purposes) as upposed to upright and vertical
  Beta - thermal expansion coefficient of the air
  upper - if the plate is horizontal, whether the air is above or below the plate

  Returns:
  h - convection coefficient for the plate (inside surface) in W/m^2 K
  """
  
  Pr = v_air/alpha_air
  if fan_speed == 0:
    return 0
  def forced_internal_convection():
    """
    Calculate the convection coefficient for a plate on the inside of the sensor enclosure
    
    Returns:
    h - convection coefficient for the plate for forced convection only
    """
    
    def parallel_plate(fan_speed, length, v_air, Pr, k_air): # assumes each is a flat plate because we can't really calculate orientation w.r.t. wind
      """
      flat plate correlation for forced convection parallel to plate

      returns h
      """
      Re = fan_speed * length / v_air # reynolds number
      Re_critical = 5 * 10**5 # turbulent case for Re_L
      if Re < Re_critical:
        Nu = 0.664 * Re ** 0.5 * Pr ** (1/3) #laminar case
      else:
        Nu = (0.664 * Re_critical ** 0.5 + 0.037 * (Re ** 0.8 - Re_critical ** 0.8)) * Pr ** (1/3) #turbulent case
      h = Nu * k_air / length # convection HT coefficient
      return h, Re

    # flat plate correlation for forced convection parallel to plate
    def perpendicular_plate(fan_speed, v_air, Pr, k_air, perp_length_1, perp_length_2):
      """
      flat plate correlation for forced convection perpendicular to plate

      returns h
      """

      D = hydraulic_diameter(perp_length_1, perp_length_2) #hydraulic diameter for perpendicular face
      Re = fan_speed * D / v_air # reynolds number
      Nu = 0.3 + (0.62 * Re ** 0.5 * Pr ** (1/3) / (1 + (0.4/Pr) ** (2/3) ) ** 0.25) * ( 1 + (Re/282000) ** 0.625) ** 0.8 # churchill bernstein correlation
      h = Nu * k_air / D # convection HT coefficient
      return h
    
    """
    We don't know box orientation for face flow direction, so we approximate h 
    by averaging the h found for each of the three axial fan flow directions
    """
    h_side1, Re_L1 = parallel_plate(fan_speed, perp_length1, v_air, Pr, k_air)
    h_side2, Re_L2 = parallel_plate(fan_speed, perp_length2, v_air, Pr, k_air)
    h_end = perpendicular_plate(fan_speed, v_air, Pr, k_air, perp_length1, perp_length2)
    avg_h = (h_side1 + h_side2 + h_end)/3 
    Re_L = (Re_L1 + Re_L2) /2
    return avg_h, Re_L


  # weighted average h as calculated above for fan
  h_forced, Re_avg = forced_internal_convection()

  # calculate free convection Nusselt number
  Nu_free, Gr = free_convection(T_surface, T_infinity, area, perimeter, horizontal, v_air, Beta, alpha_air, upper)

  # calculate h_free
  if horizontal: 
    h_free =  Nu_free * k_air / (area/perimeter) # horizontal correlations have this specific characteristic length
  else:
    h_free = Nu_free * k_air / characteristic_length

  # determine which type of convection is dominant
  dominance = Gr / Re_avg**2 


  if dominance < 0.1: # forced convection dominant
    return h_forced
  elif dominance > 10: # free convection dominant
    return h_free
  else:
    # calcualte mixed convection Nu
    n = 3
    h_mixed = (h_free ** n + h_forced ** n) ** (1/n) # combined / mixed case empirical relation (approximation)
    return h_mixed

def get_timezone(latitude, longitude):
  """
  Find the timezone

  Parameters:
  latitude - in degrees, -90 to 90
  longitude - in degrees, -180 to 180
  
  Returns:
  tiemzone - the pytz timezone object for the given location
  """

  tf = TimezoneFinder() 
  timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
  timezone = pytz.timezone(timezone_str)
  return timezone


# get data from FoCO roof weather site
def get_local_data(start_date, end_date):
  def api_call(start_date = "2024-06-17", end_date = "2024-06-20"):
    import requests
    import json

    #notes:

    #__author__ = "Christian L'Orange"
    #__copyright__ = "Copyright 2021"
    #__credits__ = ["Chrisitan L'Orange"]
    #__license__ = "GPL"
    #__version__ = "1.0.8"
    #__email__ = "lorange.christian@gmail.com"
    #__status__ = "Production"
    #__description__ = "Code to pull the roof weather data"
    #__updated__ = "12 Oct 2021"
    api_path = "local_api_key.txt"
    key = open(api_path).read()[:-1]
    inst = "weather"
    headers = {
      'X-Parse-Application-Id': 'Mimir',
      'X-Parse-REST-API-Key': 'undefined',
      }

    params = (
      ('order', 'createdAt'), #show oldest elements first,
      ('limit', '10000'), #how many records to return
      ('where', '{\"createdAt\":{\"$gte\":{\"__type\":\"Date\",\"iso\":\"' + start_date + '\"},\"$lte\":{\"__type\":\"Date\",\"iso\":\"' + end_date + '\"}}}')#set start date (lt==less than gt==greater than)
      )

    response = requests.get('https://makai-hub.com/api/instruments/' + inst + '/?&key=' + key, headers=headers, params=params)

    # Returns data which matches the parameters (params) listed above
    json_data = json.loads(response.text)

    # Extract the weather data
    weather_data = json_data.get('weather', [])

    # Convert the weather data to a DataFrame
    weather_df = pd.DataFrame(weather_data)
    return weather_df
    
  df = api_call(start_date, end_date)
  df['datetime'] = pd.to_datetime(df['sampleTime'])
  df['temp_out_F'] = df['temp_out_F'].astype(float)
  df['Temperature'] = round( ( df['temp_out_F'] -32 )*  5/9, 4)
  df['Wind Speed'] = df['windspeed'].astype(float)
  df['DNI'] = df['solarrad_Wm2'].astype(float)
  df['seconds'] = (df['datetime'] - df['datetime'].iloc[-1]).dt.total_seconds()
  df.sort_values(by='datetime', inplace=True)
  return df

def convert_to_local(dt_utc, timezone):
    """
    convert pandas timezone object to local timezone 
    without changing the time (so 09:00 UTC becomes 09:00 Denver)
    """
    dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(timezone).replace(tzinfo=None)
    return dt_local

def convert_to_local_with_zone(dt_utc, timezone):
    """
    convert pandas timezone object to local timezone 
    while changing the time (so 09:00 UTC becomes 03:00 Denver)
    """
    dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(timezone).replace()
    return dt_local

def remove_timezone(dt_aware):
    """
    remove timezone specification from pd datetime object
    so 09:00 Denver just becmoes 09:00 (which assmues UTC but is used for plotting and merging dfs)
    """
    return dt_aware.replace(tzinfo=None)

def get_data(start_date_time, end_date_time, latitude, longitude, timezone, logging, stop_event = False):
    """
    Retrieves solar radiation and weather data from NASA's POWER API for a specified date range and location. 
    This data includes direct normal irradiance (DNI), temperature, and wind speed, which is processed into a DataFrame.

    Parameters:
        start_date_time (datetime obj): Start of the date range for data retrieval.
        end_date_time (datetime obj): End of the date range for data retrieval.
        latitude (float): Latitude of the location for data retrieval.
        longitude (float): Longitude of the location for data retrieval.
        timezone (str): Timezone to convert the data to (currently unused in the function).
        logging (logging.Logger): Logger for capturing information and errors.
        stop_event (threading.Event, optional): Event to stop data retrieval early (default is False).

    Returns:
        DataFrame: A DataFrame containing solar radiation, temperature, wind speed, and associated timestamps 
                   for the specified period.
    """
    
    def get_solar_radiation_data(start_date_time, end_date_time, latitude, longitude):
        """
        Fetches solar radiation, temperature, and wind speed data from NASA's POWER API for a specific date range and location.

        Parameters:
            start_date_time (datetime): Start of the date range for data retrieval.
            end_date_time (datetime): End of the date range for data retrieval.
            latitude (float): Latitude of the location for data retrieval.
            longitude (float): Longitude of the location for data retrieval.

        Returns:
            DataFrame: A DataFrame containing DNI (Direct Normal Irradiance), temperature, wind speed, and timestamps, 
                       or None if an error occurs.
        """
        start_date = start_date_time.strftime("%Y%m%d")
        end_date = end_date_time.strftime("%Y%m%d")

        logging.info(f"*** Pulling NASA Radiation Data for {start_date} ***")
        api_path = "NASA_api_key.txt"
        api_key = open(api_path).read()[:-1]

        # Define the parameters
        params = {
          'parameters': 'ALLSKY_SFC_SW_DNI,T2M,WS10M',
          'community': 'RE',
          'longitude': longitude,
          'latitude': latitude,
          'start': start_date,
          'end': end_date,
          'format': 'JSON'
        }

        # Define the headers
        headers = {
          'Authorization': f'Bearer {api_key}'
        }

        # Define the API endpoint
        url = 'https://power.larc.nasa.gov/api/temporal/hourly/point'

        # Make the request
        response = requests.get(url, params=params, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
          data = response.json()

          # Extract relevant data
          parameters = data['properties']['parameter']
          radiation_data = parameters['ALLSKY_SFC_SW_DNI']
          temp = parameters['T2M'].values()
          windspeed = parameters['WS10M'].values()
        
          # Create a DataFrame
          df = pd.DataFrame(list(radiation_data.items()), columns=['Datetime', 'DNI'])
          df['Temperature'] = temp
          df['Wind Speed'] = windspeed

          df['datetime'] = df['Datetime'].apply(lambda x: datetime.strptime(x, '%Y%m%d%H'))
          #df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(timezone)
          df['datetime'] = df['datetime'].apply(lambda x: remove_timezone(x))
          #df['datetime'] = df['datetime'].apply(lambda x: remove_timezone(x))
          logging.info("*** Successful data pull ***")
          return df
        else:
          logging.error(f"Error: {response.status_code} for date {start_date}")
          return None

    def chunking(start_datetime, end_datetime, latitude, longitude, func, stop_event):
      """
      Splits a date range into smaller chunks and retrieves data for each chunk, concatenating the results into a single DataFrame.
      NASA API only allows a year at a time (ish)

      Parameters:
          start_datetime (datetime): Start of the date range.
          end_datetime (datetime): End of the date range.
          latitude (float): Latitude of the location.
          longitude (float): Longitude of the location.
          func (function): The data retrieval function (either NASA or ERA5 in this case).
          stop_event (threading.Event): Event to stop the process prematurely (optional).

      Returns:
          DataFrame: A concatenated DataFrame of all data retrieved over the date chunks, or None if stopped or no data is returned.
      """
              
      # Initialize an empty list to store individual dataframes
      dfs = []

      # Iterate through date range in chunks
      current_date = start_datetime

      if "era5" in func.__name__:
        maxdays = 250
        logging.info("ERA5 Function Call")
      else:
        maxdays = 365
      
      while current_date <= end_datetime:
          if stop_event and stop_event.is_set():
            logging.info("Query stopped")
            return None

          # Define chunk end date (7 days ahead or end_date, whichever is earlier)
          chunk_end_date = min(current_date + timedelta(days=maxdays), end_datetime)

          # Fetch data for the current chunk
          df_chunk = func(current_date, chunk_end_date, latitude, longitude)
          if df_chunk is not None:
              dfs.append(df_chunk)

          # Move to the next chunk
          current_date = chunk_end_date + timedelta(days=1)

      # Concatenate all dataframes into a single dataframe
      if dfs:
          final_df = pd.concat(dfs, ignore_index=True)
          return final_df
      else:
          return None

    df = chunking(start_date_time, end_date_time, latitude, longitude, get_solar_radiation_data, stop_event)

    merged_df = df
    merged_df['seconds'] = (merged_df['datetime'] - merged_df['datetime'].iloc[0]).dt.total_seconds()
    return merged_df

def get_data_bound(latitude, longitude, start_date_time, end_date_time, timezone, quantiles, logging, stop_event, all = False):
    """
    Retrieves bounded solar radiation, temperature, and wind speed data for a given location and time range,
    then computes the upper and lower quantiles and mean values. The data is returned for each hour, day, and month,
    and can optionally include all the original data retrieved.

    Parameters:
        latitude (float): Latitude of the location for data retrieval.
        longitude (float): Longitude of the location for data retrieval.
        start_date_time (datetime): Start of the date range the data.
        end_date_time (datetime): End of the date range for the data.
        timezone (str): Timezone to use for data conversion.
        quantiles (tuple): A tuple specifying the lower and upper quantiles for bounding the data (e.g., (0.1, 0.9)).
        logging (logging.Logger): Logger for capturing information and errors.
        stop_event (threading.Event): Event to stop data retrieval early if triggered, optional.
        all (bool, optional): If True, returns the entire dataset along with the bounded data (default is False).

    Returns:
        tuple: A tuple containing:
            - df_bound_lower (DataFrame): DataFrame with lower bound quantiles of DNI, temperature, and wind speed.
            - df_bound_upper (DataFrame): DataFrame with upper bound quantiles of DNI, temperature, and wind speed.
            - df_mean (DataFrame): DataFrame with the mean values of DNI, temperature, and wind speed.
            - whole_df (DataFrame, optional): If `all` is True, the entire original dataset is also returned.
    """
    
    # start of data query for bounded data (21 years)
    logging.info("*** Retrieving bounded data ***")
    start_query = datetime.strptime('2003-01-01 00:00:00',"%Y-%m-%d %H:%M:%S")
    end_query = datetime.strptime('2023-12-31 23:00:00',"%Y-%m-%d %H:%M:%S")

    # full 2001 - 2023 df
    whole_df = get_data(start_query, end_query, latitude, longitude, timezone, logging, stop_event)
    whole_df = whole_df.set_index('datetime')

    # set columns of time vars
    whole_df['hour'] = whole_df.index.hour
    whole_df['day'] = whole_df.index.day
    whole_df['month'] = whole_df.index.month

    # calculate lower quantile vals for temp and DNI and upper quantile for wind
    dni_df_lower = whole_df.groupby(['hour','day','month'])[['DNI', 'Temperature']].quantile(quantiles[0])
    wind_df_lower = whole_df.groupby(['hour','day','month'])['Wind Speed'].quantile(1-quantiles[0])
    
    # create lower bound df
    df_bound_lower = pd.merge(dni_df_lower, wind_df_lower, left_index=True, right_index=True)
    df_bound_lower = df_bound_lower.reset_index()

    # calculate upper quantile vals for temp and DNI and lower quantile for wind
    dni_df_upper = whole_df.groupby(['hour','day','month'])[['DNI', 'Temperature']].quantile(quantiles[1])
    wind_df_upper = whole_df.groupby(['hour','day','month'])['Wind Speed'].quantile(1-quantiles[1])
    
    # create upper bound df
    df_bound_upper= pd.merge(dni_df_upper, wind_df_upper, left_index=True, right_index=True)
    df_bound_upper = df_bound_upper.reset_index()

    # create mean meteorological vars dataframe
    dni_df_mean = whole_df.groupby(['hour','day','month'])[['DNI', 'Temperature']].mean()
    wind_df_mean = whole_df.groupby(['hour','day','month'])['Wind Speed'].mean()
    df_mean = pd.merge(dni_df_mean, wind_df_mean, left_index=True, right_index=True)
    df_mean = df_mean.reset_index()

    year = start_date_time.year

    # Function to handle day out of range errors
    def safe_datetime(year, month, day, hour):
        """
        Safely creates a datetime object, handling out-of-range day errors. If invalid, returns NaT.

        Parameters:
            year (int): Year of the datetime.
            month (int): Month of the datetime.
            day (int): Day of the datetime.
            hour (int): Hour of the datetime.

        Returns:
            datetime or NaT: Returns a valid datetime object or NaT if an error occurs.
        """
        try:
            return pd.Timestamp(year=year, month=month, day=day, hour=hour)
        except ValueError:
            return pd.NaT  # Return a NaT (Not a Time) if there's an error

    # Create a new datetime column using vectorized operations for each df
    df_bound_upper['datetime'] = [
        safe_datetime(year, month, day, hour)
        for month, day, hour in zip(df_bound_upper['month'], df_bound_upper['day'], df_bound_upper['hour'])
    ]
    df_bound_upper = df_bound_upper.drop(columns=['hour', 'day', 'month'])

    df_bound_lower['datetime'] = [
        safe_datetime(year, month, day, hour)
        for month, day, hour in zip(df_bound_lower['month'], df_bound_lower['day'], df_bound_lower['hour'])
    ]
    df_bound_lower = df_bound_lower.drop(columns=['hour', 'day', 'month'])

    df_mean['datetime'] = [
        safe_datetime(year, month, day, hour)
        for month, day, hour in zip(df_mean['month'], df_mean['day'], df_mean['hour'])
    ]
    df_mean = df_mean.drop(columns=['hour', 'day', 'month'])
    
    # Create a "seconds" column (used for data querying from HT model)
    df_bound_lower['seconds'] = (df_bound_lower['datetime'] - df_bound_lower['datetime'].iloc[0]).dt.total_seconds()
    df_bound_upper['seconds'] = (df_bound_upper['datetime'] - df_bound_upper['datetime'].iloc[0]).dt.total_seconds()
    df_mean['seconds'] = (df_mean['datetime'] - df_mean['datetime'].iloc[0]).dt.total_seconds()

    # sort df and reset index so datetime col exists
    df_bound_upper = df_bound_upper.sort_values(by = 'datetime').reset_index()
    df_bound_lower = df_bound_lower.sort_values(by = 'datetime').reset_index()
    df_mean = df_mean.sort_values(by = 'datetime').reset_index()

    # return
    logging.info('data collection complete')
    if all:
      whole_df = whole_df.reset_index()
      return df_bound_lower, df_bound_upper, df_mean, whole_df
    return df_bound_lower, df_bound_upper, df_mean

# Get the weighted average of internal temperature faces for radiation heat transfer to the battery
def weighted_avg_T(Temperatures, areas, n):
    """
    Calculate the weighted average temperature of internal faces for radiation heat transfer to the battery.
    
    Parameters:
        Temperatures (ndarray): A 3D array where each element represents temperature values of internal faces.
        areas (ndarray): A 1D array of areas corresponding to each face.
        n (int): The index for selecting specific internal faces.

    Returns:
        float: The weighted average temperature based on the areas of internal faces.
    """
    return (Temperatures[:, n, -1] * areas).sum(axis=0) / areas.sum()  # weighted avg

# Get the average of internal temperature faces for radiation heat transfer to the battery
def avg_T(Temperatures, areas, n):
    """
    Calculate the simple average temperature of internal faces for radiation heat transfer to the battery.
    
    Parameters:
        Temperatures (ndarray): A 3D array where each element represents temperature values of internal faces.
        areas (ndarray): A 1D array of areas corresponding to each face.
        n (int): The index for selecting specific internal faces.

    Returns:
        float: The simple average temperature of the internal faces.
    """
    return np.mean(Temperatures[:, n, -1])  # avg

# Get the minimum of internal temperature faces for radiation heat transfer to the battery
def min_T(Temperatures, areas, n):
    """
    Retrieve the minimum temperature from the internal faces for radiation heat transfer to the battery.
    
    Parameters:
        Temperatures (ndarray): A 3D array where each element represents temperature values of internal faces.
        areas (ndarray): A 1D array of areas corresponding to each face.
        n (int): The index for selecting specific internal faces.

    Returns:
        float: The minimum temperature among the selected internal faces.
    """
    return Temperatures[:, n, -1].min()  # min

# Get the maximum of internal temperature faces for radiation heat transfer to the battery
def max_T(Temperatures, areas, n):
    """
    Retrieve the maximum temperature from the internal faces for radiation heat transfer to the battery.
    
    Parameters:
        Temperatures (ndarray): A 3D array where each element represents temperature values of internal faces.
        areas (ndarray): A 1D array of areas corresponding to each face.
        n (int): The index for selecting specific internal faces.

    Returns:
        float: The maximum temperature among the selected internal faces.
    """
    return Temperatures[:, n, -1].max()  # max

# query dataframe from api for data
def data_query(starttime, n, dt, column, df):
  """
  Queries the DataFrame to retrieve an interpolated data point for a specified time.
  
  Parameters:
      starttime (float): The starting timestamp (in seconds) from which to calculate the query time.
      n (int): A multiplier to determine how far from the starttime to query.
      dt (float): The time step size (in seconds).
      column (str): The name of the column in the DataFrame for which to retrieve the data.
      df (DataFrame): The DataFrame containing the data, where 'seconds' is a column representing timestamps.

  Returns:
      float: The interpolated value from the specified column at the calculated timestamp.
  """
  timestamp = starttime + n * dt

  # Find the index where the given number would be inserted
  insertion_index = df['seconds'].searchsorted(timestamp)

  # Find the rows on either side
  if insertion_index == 0:
      # The given number is smaller than all numbers in the DataFrame
      closest_below = df.iloc[0]
      closest_above = df.iloc[0]
  elif insertion_index == len(df):
      # The given number is larger than all numbers in the DataFrame
      closest_below = df.iloc[-1]
      closest_above = df.iloc[-1]
  else:
      closest_below = df.iloc[insertion_index - 1]
      closest_above = df.iloc[insertion_index]

  lower_data = closest_below[column]
  upper_data = closest_above[column]

  factor = (timestamp - closest_below.seconds) /(closest_above.seconds - closest_below.seconds + 0.00001)
  data_point = lower_data + (upper_data - lower_data) * factor
  return data_point

# get the solar angle above the horizon
def get_solar_angles(latitude, longitude, date_time, timezone):
  """
    Computes the solar altitude (angle above the horizon) and azimuth for a given location, date, and time.

    Parameters:
        latitude (float): The latitude of the observer's location (in degrees).
        longitude (float): The longitude of the observer's location (in degrees).
        date_time (datetime): The local date and time to compute the solar angles.
        timezone (pytz.timezone): The time zone of the location.

    Returns:
        tuple: A tuple containing:
            - float: Solar altitude (angle above the horizon in radians).
            - float: Solar azimuth (direction relative to north in radians).
    """
  dt_with_timezone = timezone.localize(date_time)
  observer = ephem.Observer() # create sun observer
  observer.lat = str(latitude) # set observer lat
  observer.lon = str(longitude) # set observer lon
  observer.date = dt_with_timezone.astimezone(pytz.utc) # set observing datetime

  sun = ephem.Sun() # create sun
  sun.compute(observer) # find sun location
  return float(sun.alt), float(sun.az) # return angle above the horizon and azimuth

# get the angle of the sun above the horizon given time and location
def angle_func(start_time, n, dt, df, latitude, longitude, timezone):
  """
  Calculates the solar altitude (angle of the sun above the horizon) for a specific time and location.

  Parameters:
      start_time (datetime): The starting reference time.
      n (int): The index multiplier for the time step.
      dt (timedelta): The time step duration.
      df (DataFrame): The data frame containing the queried data.
      latitude (float): The latitude of the location (in degrees).
      longitude (float): The longitude of the location (in degrees).
      timezone (pytz.timezone): The time zone of the location.

  Returns:
      float: Solar altitude (angle above the horizon in radians).
  """
  date_time = data_query(start_time, n, dt, "datetime", df)
  angle, _ = get_solar_angles(latitude, longitude, date_time, timezone) #in radians
  return angle

# get the azimuthal angle of the sun relative to north
def azimuthal_angle_func(start_time, n, dt, df, latitude, longitude, timezone):
  """
  Calculates the solar azimuth (direction of the sun relative to north) for a specific time and location.

  Parameters:
      start_time (datetime): The starting reference time.
      n (int): The index multiplier for the time step.
      dt (timedelta): The time step duration.
      df (DataFrame): The data frame containing the queried data.
      latitude (float): The latitude of the location (in degrees).
      longitude (float): The longitude of the location (in degrees).
      timezone (pytz.timezone): The time zone of the location.

  Returns:
      float: Solar azimuth (direction relative to north in radians).
  """
  date_time = data_query(start_time, n, dt, "datetime", df)
  _, azimuthal_angle = get_solar_angles(latitude, longitude, date_time, timezone) # in radians
  return azimuthal_angle

# get the external temperature
def T_infinity_func(start_time, n, dt, df):
  """
  Retrieves the external ambient temperature (T∞) in Kelvin for a given time.

  Parameters:
      start_time (datetime): The starting reference time.
      n (int): The index multiplier for the time step.
      dt (timedelta): The time step duration.
      df (DataFrame): The data frame containing temperature data in Celsius.

  Returns:
      float: External temperature in Kelvin.
  """
  temp =  data_query(start_time, n, dt, "Temperature", df) + 273.15
  return temp

# get the total incident radiation
def G_func(start_time, n, dt, df): 
  """
  Retrieves the total incident direct normal irradiance (DNI) for a given time.

  Parameters:
      start_time (datetime): The starting reference time.
      n (int): The index multiplier for the time step.
      dt (timedelta): The time step duration.
      df (DataFrame): The data frame containing DNI data.

  Returns:
      float: The total incident direct normal irradiance (DNI) in W/m².
  """
  radiation = data_query(start_time, n, dt, "DNI", df)
  return radiation

# get the wind speed
def air_velocity_func(start_time, n, dt, df): 
  """
  Retrieves the wind speed at a specific time.

  Parameters:
      start_time (datetime): The starting reference time.
      n (int): The index multiplier for the time step.
      dt (timedelta): The time step duration.
      df (DataFrame): The data frame containing wind speed data.

  Returns:
      float: Wind speed in meters per second.
  """
  wind_speed = data_query(start_time, n, dt, "Wind Speed", df)
  return wind_speed

### Check if it is time to collect new environmental data
def data_collect_now(n, interval, run_interval, time_in_interval, dt, data_collection_time_interval = 60): # data_collection_time_interval how often in seconds to collect data
  """
  Determines whether it's time to collect new environmental data based on the simulation time.

  Parameters:
      n (int): The current time step.
      interval (int): Total time of a cycle in seconds.
      run_interval (int): Time in seconds for how long the simulation runs per cycle.
      time_in_interval (int): Time indicator for the current position in the interval.
      dt (float): Time step duration (in seconds).
      data_collection_time_interval (int): How often to collect data in seconds (default is 60 seconds).

  Returns:
      bool: True if it's time to collect data, False otherwise.
  """


  ### Helper function to check if there's a multiple of x in the range [a, b]
  def exists_multiple_within_range(a, b, x):
    """
    Checks if there exists a multiple of x within the range [a, b].

    Parameters:
        a (int): Start of the range.
        b (int): End of the range.
        x (int): The value to check multiples of.

    Returns:
        bool: True if a multiple of x exists within the range, False otherwise.
    """
    # Calculate the smallest integer k such that k * x >= a
    smallest_int = np.ceil(a / x)
    
    # Compute the smallest multiple of x within the range
    n_min = smallest_int * x
    
    # Check if this multiple is within the range [a, b]
    return n_min + 1 <= b

  # Check if time to recollect data
  if int(n/dt) % data_collection_time_interval == 1:
    return True
  
  # interval is number of seconds for cycle, run_interval is number of seconds to run simulation per cycle
  n_lower =  n - int(interval/dt) + int(run_interval/dt)

  # Check if we missed the data collection time based on the step interval
  # By ensuring new interval has just started and before skipping with linear approx,
  # data collection would at some poit have been int(n/dt) % data_collection_time_interval == 1:
  if time_in_interval == 0 and exists_multiple_within_range(n_lower, n, int(data_collection_time_interval/dt)):
    return True
  else:
    return False

def solar_panel_model(latitude, longitude, logging, start_date_time, end_date_time, df,
                    solar_panel_area, solar_panel_tilt, solar_panel_azimuth, solar_panel_efficiency, 
                    battery_rated_capacity, battery_efficiency, dt,
                    power_out_func, stop_event = False, shading_ranges = []):
    """
    Simulates the operation of a solar panel and battery system over a time period.

    Parameters:
        latitude (float): Latitude of the solar panel location.
        longitude (float): Longitude of the solar panel location.
        logging (logging.Logger): Logger for information output.
        start_date_time (datetime): Simulation start date and time.
        end_date_time (datetime): Simulation end date and time.
        df (DataFrame): meterological data (DNI, Temperature, etc.).
        solar_panel_area (float): Surface area of the solar panel in square meters.
        solar_panel_tilt (float): Tilt angle of the solar panel.
        solar_panel_azimuth (float): Azimuth angle of the solar panel.
        solar_panel_efficiency (float): Efficiency of the solar panel (0 to 1).
        battery_rated_capacity (float): Battery rated capacity in watt-hours.
        battery_efficiency (float): Efficiency of the battery.
        dt (float): Time step duration (in seconds).
        power_out_func (function): Function to calculate power output, assumed constant.
        stop_event (threading.Event, optional): Event to stop the simulation (default is False).
        shading_ranges (list, optional): List of time ranges (dictionary) where shading occurs.

    Returns:
        tuple: Arrays of time steps, battery charges, power input, and power output.
    """
    import numpy as np
    from math import cos, sin, radians, degrees, acos

    # Function to calculate incident angle on the panel
    def incident_angle_factor(elevation, azimuth, tilt_angle, panel_azimuth):
        tilt_angle = radians(tilt_angle)
        panel_azimuth = radians(panel_azimuth)

        # downward incident radiation + horizontal incident radiation factor
        normal_term = sin(elevation) * cos(tilt_angle)
        horizontal_term = cos(elevation) * sin(tilt_angle) * cos(azimuth - panel_azimuth)
        
        if horizontal_term < 0: horizontal_term = 0
        if normal_term < 0: normal_term = 0

        angle_factor = normal_term + horizontal_term

        if elevation < 0: 
            return 0 #no solar radiation
        else:
          return angle_factor # incident angle factor in radians

    #set up stuff

    t_final = (end_date_time - start_date_time).total_seconds()
    current_date_time = datetime.now().strftime('%H:%M:%S')
    logging.info(f"{current_date_time} *** RUNNING SOLAR MODEL: starting at {start_date_time} ***")
    start_time = time.time() #start of run_time
    timezone = get_timezone(latitude, longitude)

    # initilization
    n=1
    start_time_simulation_seconds = (pd.to_datetime(start_date_time) - df['datetime'].iloc[0]).total_seconds()

    #create arrays
    max_charge = battery_rated_capacity * 3600 * battery_efficiency # approximating J of battery
    time_steps = np.linspace(0,t_final,int(t_final/dt) +1)
    charges =  np.full((time_steps.shape[0]), max_charge)
    power_ins =  np.zeros((time_steps.shape[0]))
    power_outs =  np.zeros((time_steps.shape[0]))

    # print progress (at 10 discrete intervals)
    print_now = True
    print_step = 1

    query_interval = 300 #default
    if query_interval <= dt * 2:
      query_interval = dt * 2

    logging.info("dt " + str(dt))
    logging.info("query_interval " + str(query_interval))

    while n < time_steps.shape[0]:
        if stop_event and stop_event.is_set():
            return

    # Data queries perform only every 10 seconds to save substantial computing time
        if n % int(query_interval/dt) == 1 or n == 1:
            elevation_angle = angle_func(start_time_simulation_seconds, n, dt, df, latitude, longitude, timezone)
            azimuthal_angle = azimuthal_angle_func(start_time_simulation_seconds, n, dt, df, latitude, longitude, timezone)
            DNI = G_func(start_time_simulation_seconds, n, dt, df) 

            simulation_date_time = start_date_time + timedelta(seconds=(n * dt))
            if is_datetime_in_ranges(simulation_date_time, shading_ranges):
              DNI = 0
              
            power_in = solar_panel_area * DNI * incident_angle_factor(elevation_angle,  azimuthal_angle, solar_panel_tilt, solar_panel_azimuth) * solar_panel_efficiency
            power_out = power_out_func(start_time_simulation_seconds, n, dt, df)
            



        # print progress (at 10 discrete intervals)
        if n  >= int(print_step * time_steps.shape[0]/10) and print_now == True:
            logging.info(f"{datetime.now().strftime('%H:%M:%S')}  -  {round(n / time_steps.shape[0]* 100)}%: {round(time.time() - start_time)}")
            print_now = False
            print_step += 1
        elif n  >= int(print_step * time_steps.shape[0]/10) and print_now == False: 
            print_now = True

        charges[n] = charges[n-1] + (power_in - power_out) * dt

        # max charge case
        if charges[n] >= max_charge:
          charges[n] = max_charge # cap charging
        # min charge case
        elif charges[n] < 0:
          charges[n] = 0 # cant go below zero      
        power_ins[n] = power_in
        power_outs[n] = power_out
        
        n+=1
        
    logging.info(f"*** TOTAL TIME *** {round(time.time() - start_time)} seconds")
    return time_steps, charges, power_ins, power_outs

def is_datetime_in_ranges(dt, ranges):
    """
    Check if the given datetime is within any of the datetime ranges.

    :param dt: The datetime to check.
    :param ranges: List of tuples, where each tuple is (start_dt, end_dt).
    :return: True if datetime is within any range, False otherwise.
    """
    for start_dt, end_dt in ranges:
        start_dt = datetime.strptime(start_dt, "%H:%M:%S").time()
        end_dt = datetime.strptime(end_dt, "%H:%M:%S").time()
        if start_dt <= dt.time() <= end_dt:
            return True
    return False

# run the full model with reduced memory (no full Temperatures array)
def run_model_reduced_complexity_and_memory(latitude, longitude, logging, interval, run_interval, start_date_time, df, fan_speed_func, air_velocity_func, internal_radiation_T_func, angle_of_incidence_func, q_flux_func,
              convection_func, t_final, T_max, T_infinity_func, T_initial, k, rho, Cp, G_func, emissivity, absorptivity, B_air, v_air, k_air, alpha_air,
              thicknesses, L = 0.2, W = 0.1, H = 0.1, N = 50, h_max = 100, include_internal_effects = False, stop_event = None, box_shading = False, shading_ranges = [], fan_mixing_ratio = 0.1):
  # print that it is running
  """
  Run a full numerical model for thermal dynamics with reduced memory usage by omitting the temperatures array.
  
  Parameters:
  - latitude (float): Latitude for calculating solar geometry.
  - longitude (float): Longitude for calculating solar geometry.
  - logging (Logger): Logger for outputting simulation progress and information.
  - interval (float): total interval for cycling linear approximations to reduce complexity
  - run_interval (float): interval to run full fidelity for running the model while reducing computational expense.
  - start_date_time (datetime): Starting date and time for the simulation.
  - df (DataFrame): DataFrame containing meteorological data.
  - fan_speed_func (function): Function to determine fan speed based on internal temperature and time.
  - air_velocity_func (function): Function to calculate air velocity based on conditions.
  - internal_radiation_T_func (function): Function to query internal radiation (currently unused).
  - angle_of_incidence_func (function): Function to query the angle of incidence of sunlight on each face.
  - q_flux_func (function): Function for the internal heat generation or heat flux.
  - convection_func (function): Function to calculate the convection coefficient.
  - t_final (float): Final simulation time.
  - T_max (float): Maximum assumed temperature for stability criterion.
  - T_infinity_func (function): Function to query external temperature over time.
  - T_initial (float): Initial temperature at the start of the simulation.
  - k (float): Thermal conductivity of the material.
  - rho (float): Density of the material.
  - Cp (float): Specific heat capacity of the material.
  - G_func (function): Function to query DNI solar radiation.
  - emissivity (float): Emissivity of the material surface.
  - absorptivity (float): Absorptivity of the material surface.
  - B_air (float): Thermal expansion coefficient of air (for free convection).
  - v_air (float): kinematic viscosity of the air for convection calculations.
  - k_air (float): Thermal conductivity of air.
  - alpha_air (float): Thermal diffusivity of air.
  - thicknesses (list): List of thicknesses for each geometry face.
  - L (float): Length of the enclosure (default 0.2 m).
  - W (float): Width of the enclosure (default 0.1 m).
  - H (float): Height of the enclosure (default 0.1 m).
  - N (int): Number of discrete points for FDM one dimensionally through box thickness.
  - h_max (float): Maximum heat transfer coefficient (default 100) for stability criterion.
  - include_internal_effects (bool): Whether to include internal heat generation effects (default False), currently unused.
  - stop_event (Event): Event to signal stopping the simulation (default None).
  - box_shading (bool): Whether to account for shading of the box by a solar shield(default False).
  - shading_ranges (list): List of time ranges (dictionaries) for shading effects.

  Returns:
  - tuple: A tuple containing the following:
      - time_steps (list): List of timesteps in seconds  for the simulation.
      - previous_temperatures (array): Array of temperature distribution from the previous step.
      - avg_temp (float): Internal temperature at each step during the simulation.
      - battery_temp (float): Temperature of the battery at each step during the simulation (equals internal now).
      - dt (float): Time step used in the simulation.
      - outside_temp (float): Outside temperature at each step of the simulation.
  """

  N= int(N)
  current_date_time = datetime.now().strftime('%H:%M:%S')
  logging.info(f" *** RUNNING FULL NUMERICAL MODEL: starting at {start_date_time} ***")
  
  start_time = time.time() #start of run_time
  theta = np.arctan(W/L) # angle in the xy plane of sun
  angle_of_incidence = 0 # angle above horizon

  # get timezone from lat, lon
  timezone = get_timezone(latitude, longitude)

  alpha_ABS = k/(rho*Cp)

  # get start time of simulation in seconds relative to dataframe
  start_time_simulation_seconds = (pd.to_datetime(start_date_time) - df['datetime'].iloc[0]).total_seconds()
  def get_geometry(geo):
    upper = False

    if "WH" in geo:
      area = W * H
      perimeter = 2*W + 2*H
      if (angle_of_incidence < np.pi/2) == (int(geo[-1]) == 1): #check that radiation would be shining on this face
        rad_factor = np.cos(angle_of_incidence) * np.cos(theta) # amount of radiation flux this face sees (A* = A x rad_factor)
      else: rad_factor = 0 # otherwise no radiation
      characteristic_length = (H * W) ** 0.5
      horizontal = False
      lengths = [H, W]

    elif "LH" in geo:
      area = L * H
      perimeter = 2*L + 2*H
      if (angle_of_incidence < np.pi/2) == (int(geo[-1]) == 1): #check that radiation would be shining on this face
        rad_factor =  np.cos(angle_of_incidence) * np.sin(theta) # amount of radiation flux this face sees (A* = A x rad_factor)
      else: rad_factor = 0 # otherwise no radiation
      characteristic_length = (L * H) ** 0.5
      horizontal = False
      lengths = [H, L]

    elif "LW" in geo or "WL" in geo:
      area = L * W
      perimeter = 2*L + 2*W
      if int(geo[-1]) == 1: #check that radiation would be shining on this face
        rad_factor = np.sin(angle_of_incidence) # amount of radiation flux this face sees (A* = A x rad_factor)
        upper = True
      else: 
        rad_factor = 0 # otherwise no radiation
        upper = False
      characteristic_length = (L * W) ** 0.5
      horizontal = True
      lengths = [L, W]

    if angle_of_incidence < 0 or angle_of_incidence > 180:
      rad_factor = 0
    return rad_factor, area, perimeter, characteristic_length, horizontal, upper, lengths

  geometries = ["WH1", "LH1", "LW1", "WH2", "LH2", "LW2"]

  geometry_results = {}

  for geo in geometries:
    results = get_geometry(geo)
    geometry_results[geo] = {
        "rad_factor": results[0],
        "area": results[1],
        "perimeter": results[2],
        "characteristic_length": results[3],
        "horizontal": results[4],
        "upper": results[5],
        "lengths": results[6]
    }

  def get_minimum_timestep():
    dts = []
    for i in range(len(geometries)):
      thickness = thicknesses[i]
      dts.append(stability_criterion(thickness, N, h_max, k, alpha_ABS, emissivity, T_max, q_flux_func(T_max+273.15)))

    return min(dts)

  areas = np.array([area for _, area, _, _, _, _, _ in (get_geometry(geo) for geo in geometries)]) # store areas

  #first: find the minimum timestep
  dt = get_minimum_timestep()
  if dt > 30: dt = 30

  # incase the timestep is too big relative to the run intervals
  while dt * 5 > (run_interval):
    run_interval = run_interval * 2
    interval = interval * 2

  logging.info(f"*** Timestep {round(dt, 3)} seconds ***")

  #create arrays
  time_steps = np.linspace(0,t_final,int(t_final/dt) +1)

  avg_temp =  np.zeros((time_steps.shape[0]), dtype='float32')
  avg_temp[:] = T_initial

  battery_temp =  np.zeros((time_steps.shape[0]), dtype='float32')
  battery_temp[:] = T_initial

  outside_temp = np.zeros((time_steps.shape[0]), dtype='float32')
  outside_temp[:] = T_infinity_func(start_time_simulation_seconds, 0, dt, df)


  time_in_interval = 0
  interval_steps = int(interval/dt)
  # Perform time-stepping
  n = 1

  # print progress (at 10 discrete intervals)
  print_now = True
  print_step = 1

  previous_temperatures = np.zeros((6, 2, int(N)), dtype = 'float32')
  previous_temperatures[:, :, :] = T_initial
  
  while n < time_steps.shape[0]:
  #for n in range(1, time_steps.shape[0]):
    if stop_event and stop_event.is_set():
      return

    # Data queries perform only every 10 seconds to save substantial computing time
    if data_collect_now(n, interval, run_interval, time_in_interval, dt, data_collection_time_interval = 60):
      T_infinity = T_infinity_func(start_time_simulation_seconds, n, dt, df)
      angle_of_incidence = angle_of_incidence_func(start_time_simulation_seconds, n, dt, df, latitude, longitude, timezone)
      G = G_func(start_time_simulation_seconds, n, dt, df)
      air_velocity = air_velocity_func(start_time_simulation_seconds, n, dt, df)
      fan_speed = fan_speed_func(avg_temp[n-1])

      simulation_date_time = start_date_time + timedelta(seconds=(n * dt))

      if is_datetime_in_ranges(simulation_date_time, shading_ranges):
        G = G * 0.25

    outside_temp[n] = T_infinity

    # print progress (at 10 discrete intervals)
    if n  >= int(print_step * time_steps.shape[0]/10) and print_now == True:
      logging.info(f"  -  {round(n / time_steps.shape[0]* 100)}%: {round(time.time() - start_time)}")
      print_now = False
      print_step += 1
    elif n  >= int(print_step * time_steps.shape[0]/10) and print_now == False: 
      print_now = True

      
    # iterate through each geometry (each of the 6 faces of the box)#
    for i in range(len(geometries)):
      geo = geometries[i]
      rad_factor = geometry_results[geo]['rad_factor']
      area = geometry_results[geo]['area']
      perimeter = geometry_results[geo]['perimeter']
      characteristic_length = geometry_results[geo]['characteristic_length']
      horizontal = geometry_results[geo]['horizontal']
      upper = geometry_results[geo]['upper']
      lengths = geometry_results[geo]['lengths']


      thickness = thicknesses[i]
      dx = thickness / (N - 1)  # Grid spacing
      Fo = alpha_ABS * dt / dx**2

      # if box is shaded by solar panel
      if geo == "WH1" and box_shading: 
        new_G = G * 0
      else:
        new_G = G

      # Create copy of temperature array for the next time step
      T_next = previous_temperatures[i, 0, :] # (shape N)
      
      # T_previous = Temperatures[i, n-1, :]
      T_previous = previous_temperatures[i, 0, :] # (shape N)

      # get h using convection correlations for external convection
      convection_coefficient = convection_func(T_surface = T_previous[0],
            T_infinity = T_infinity,
            characteristic_length = characteristic_length,
            area = area,
            perimeter = perimeter,
            v_air = v_air,
            Beta = B_air, 
            alpha_air = alpha_air,
            k_air = k_air,
            horizontal = horizontal,
            velocity = air_velocity,
      		upper = upper)

            # find fan speed and convection as a function of fan speed
      upper_inside = False
      if horizontal == True and upper == False: upper_inside = True

      convection_coefficient_internal = internal_convective_heat_transfer_func(
        fan_speed = fan_speed, 
        v_air = v_air, 
        k_air = k_air, 
        alpha_air = alpha_air, 
        perp_length1 = lengths[0],
        perp_length2 = lengths[1],
        T_surface = T_previous[0], 
        T_infinity = avg_temp[n-1], 
        characteristic_length = characteristic_length,
        area = area, 
        perimeter = perimeter, 
        horizontal = horizontal,  
        Beta = B_air, 
        upper = upper_inside
        )
      
      q = q_flux_func(avg_temp[n-1]) # get the radiation heat flux
      q_flux = q/np.sum(areas)

      Bi = convection_coefficient*dx/k #biot number for outside surface
      Bi_internal = convection_coefficient_internal*dx/k #biot number for inside surface

      # outer surface heat constraints
      # Conduction (2 * Fo * T), Convection (2 * Fo * Bi * T), Irradiation (2 * Fo * dx /k  * a * G), Radiation (2 * Fo * dx /k * sigma * a * T^4) 
      T_next[0] = 2 * Fo * (T_previous[1] + Bi * T_infinity + dx / k * absorptivity * new_G * rad_factor + dx /k * sigma * absorptivity * T_infinity ** 4) + (1 - 2 * Fo - 2 * Bi * Fo - Fo * 2 * dx / k * emissivity * sigma * T_previous[0]**3) * T_previous[0]

      if fan_speed > 0:
        bulk_air_temp = avg_temp[n-1] * (1-fan_mixing_ratio) + outside_temp[n-1] * fan_mixing_ratio
      else:
        bulk_air_temp = avg_temp[n-1]

      # Update temperature at interior points using finite difference scheme
      for m in range(1, N):
        if m == N-1:
          # conduction term (2 * Fo * T) | Heat gen term (2 Fo * dx /k U q_flux) | Internal convection term: 2 * Fo * Bi * T 
          if include_internal_effects:
            T_next[m] = T_previous[m] + 2 * Fo * (T_previous[m-1] - T_previous[m] + dx /k * q_flux + Bi_internal * (bulk_air_temp - T_previous[m]) )#+ dx/k * inner_q_rad_battery_flux) # for the internal heat generation constraint / term
          else: 
            T_next[m] = T_previous[m] + 2 * Fo * (T_previous[m-1] - T_previous[m] + dx /k * q_flux + Bi_internal * (bulk_air_temp - T_previous[m]))
        else:
          T_next[m] = T_previous[m] + Fo * (T_previous[m+1] + T_previous[m-1] - 2 * T_previous[m]) #
      
      # update temperature arrays
      previous_temperatures[i, 1, :] = T_next


    avg_temp[n] = internal_radiation_T_func(previous_temperatures, areas, 0) # weighted average by internal surface area

    battery_dT =  avg_temp[n] - avg_temp[n-1]
    battery_temp[n] = battery_temp[n-1] + battery_dT

    time_in_interval += dt

    #  *** to reduce complexity do linear approximations every interval ***
    if time_in_interval >= run_interval:
      end_of_write = n + 1 + interval_steps - int(time_in_interval/dt) # extrapolate form end of run_interval to end of total interval
      if end_of_write >= time_steps.shape[0]: 
        end_of_write = time_steps.shape[0] # if at end of simulation 

      # change in temperature at most recent timestep
      dT_previous = ((previous_temperatures[:, 1, :] - previous_temperatures[:, 0, :])).reshape(6, N) # (shape 6 x N)
      
      previous = previous_temperatures[:, 1, :]
      previous_temperatures[:, 1, :] = previous +  dT_previous  * (end_of_write - (n+1)) # (the linear approximated next T_previous)

      #linear approximations 
      avg_temp[n+1:end_of_write] = avg_temp[n] + np.linspace(1, end_of_write - (n+1), end_of_write - (n+1))  * (avg_temp[n] - avg_temp[n-1]) 
      battery_temp[n+1:end_of_write] = battery_temp[n] + np.linspace(1, end_of_write - (n+1), end_of_write - (n+1)) * battery_dT 
      
      # try to do linear approximation for outside temperature
      try:
        outside_temp[n+1:end_of_write] = outside_temp[n] + np.linspace(1, end_of_write - (n+1), end_of_write - (n+1))  * ((outside_temp[n] - outside_temp[n-int(300/dt)])/(int(300/dt)))
      except Exception as e:
        outside_temp[n+1:end_of_write] = outside_temp[n]
      
      #reset interval times
      time_in_interval = 0
      n = end_of_write -1
    else:
      n += 1 # otherwise just proceed to next timestep
      previous_temperatures[:, 0, :]  = previous_temperatures[:, 1, :]

  logging.info(f"*** TOTAL TIME *** {round(time.time() - start_time)} seconds")
  return time_steps, previous_temperatures, avg_temp, battery_temp, dt, outside_temp #return time steps and temperature distribution

# run the full model with linear approximations to reduce computational time, currently unused
def run_model_reduced_complexity(latitude, longitude, logging, interval, run_interval, start_date_time, df, fan_speed_func, air_velocity_func, internal_radiation_T_func, internal_radiative_heat_transfer_func, angle_of_incidence_func, q_flux_func,
              convection_func, t_final, T_max, T_infinity_func, T_initial, k, rho, Cp, G_func, emissivity, absorptivity, B_air, v_air, k_air, alpha_air,
              battery_Cp, battery_mass, battery_lengths, battery_emissivity, thicknesses, L = 0.2, W = 0.1, H = 0.1, N = 50, h_max = 100, include_internal_effects = False, box_shading = False, shading_ranges = [], fan_mixing_ratio = 0.1):
  """
  Run a full numerical model for thermal dynamics with reduced memory usage by omitting the temperatures array.
  
  Parameters:
  - latitude (float): Latitude for calculating solar geometry.
  - longitude (float): Longitude for calculating solar geometry.
  - logging (Logger): Logger for outputting simulation progress and information.
  - interval (float): total interval for cycling linear approximations to reduce complexity
  - run_interval (float): interval to run full fidelity for running the model while reducing computational expense.
  - start_date_time (datetime): Starting date and time for the simulation.
  - df (DataFrame): DataFrame containing meteorological data.
  - fan_speed_func (function): Function to determine fan speed based on internal temperature and time.
  - air_velocity_func (function): Function to calculate air velocity based on conditions.
  - internal_radiation_T_func (function): Function to query internal radiation (currently unused).
  - angle_of_incidence_func (function): Function to query the angle of incidence of sunlight on each face.
  - q_flux_func (function): Function for the internal heat generation or heat flux.
  - convection_func (function): Function to calculate the convection coefficient.
  - t_final (float): Final simulation time.
  - T_max (float): Maximum assumed temperature for stability criterion.
  - T_infinity_func (function): Function to query external temperature over time.
  - T_initial (float): Initial temperature at the start of the simulation.
  - k (float): Thermal conductivity of the material.
  - rho (float): Density of the material.
  - Cp (float): Specific heat capacity of the material.
  - G_func (function): Function to query DNI solar radiation.
  - emissivity (float): Emissivity of the material surface.
  - absorptivity (float): Absorptivity of the material surface.
  - B_air (float): Thermal expansion coefficient of air (for free convection).
  - v_air (float): kinematic viscosity of the air for convection calculations.
  - k_air (float): Thermal conductivity of air.
  - alpha_air (float): Thermal diffusivity of air.
  - thicknesses (list): List of thicknesses for each geometry face.
  - L (float): Length of the enclosure (default 0.2 m).
  - W (float): Width of the enclosure (default 0.1 m).
  - H (float): Height of the enclosure (default 0.1 m).
  - N (int): Number of discrete points for FDM one dimensionally through box thickness.
  - h_max (float): Maximum heat transfer coefficient (default 100) for stability criterion.
  - include_internal_effects (bool): Whether to include internal heat generation effects (default False), currently unused.
  - stop_event (Event): Event to signal stopping the simulation (default None).
  - box_shading (bool): Whether to account for shading of the box by a solar shield(default False).
  - shading_ranges (list): List of time ranges (dictionaries) for shading effects.

  Returns:
  - tuple: A tuple containing the following:
      - time_steps (list): List of timesteps in seconds  for the simulation.
      - previous_temperatures (array): Array of temperature distribution from the previous step.
      - avg_temp (float): Internal temperature at each step during the simulation.
      - battery_temp (float): Temperature of the battery at each step during the simulation (equals internal now).
      - dt (float): Time step used in the simulation.
      - outside_temp (float): Outside temperature at each step of the simulation.
  """
    
  current_date_time = datetime.now().strftime('%H:%M:%S')
  logging.info(f" *** RUNNING FULL NUMERICAL MODEL: starting at {start_date_time} ***")
  N = int(N)
  start_time = time.time() #start of run_time
  theta = np.arctan(W/L) # angle in the xy plane of sun
  angle_of_incidence = 0 # initial condition

  # get timezone from lat, lon
  timezone = get_timezone(latitude, longitude)

  alpha_ABS = k/(rho*Cp)

  # get start time of simulation in seconds relative to dataframe
  start_time_simulation_seconds = (pd.to_datetime(start_date_time) - df['datetime'].iloc[0]).total_seconds()

  def get_geometry(geo):
    upper = False
    if "WH" in geo:
      area = W * H
      perimeter = 2*W + 2*H
      if (angle_of_incidence < np.pi/2) == (int(geo[-1]) == 1): #check that radiation would be shining on this face
        rad_factor = np.cos(angle_of_incidence) * np.cos(theta) # amount of radiation flux this face sees (A* = A x rad_factor)
      else: rad_factor = 0 # otherwise no radiation
      characteristic_length = (H * W) ** 0.5
      horizontal = False
      lengths = [H, W]

    elif "LH" in geo:
      area = L * H
      perimeter = 2*L + 2*H
      if (angle_of_incidence < np.pi/2) == (int(geo[-1]) == 1): #check that radiation would be shining on this face
        rad_factor =  np.cos(angle_of_incidence) * np.sin(theta) # amount of radiation flux this face sees (A* = A x rad_factor)
      else: rad_factor = 0 # otherwise no radiation
      characteristic_length = (L * H) ** 0.5
      horizontal = False
      lengths = [L, H]

    elif "LW" in geo or "WL" in geo:
      area = L * W
      perimeter = 2*L + 2*W
      if int(geo[-1]) == 1: #check that radiation would be shining on this face
        rad_factor = np.sin(angle_of_incidence) # amount of radiation flux this face sees (A* = A x rad_factor)
        upper = True
      else: 
        rad_factor = 0 # otherwise no radiation
        upper = False
      characteristic_length = (L * W) ** 0.5
      horizontal = True
      lengths = [W, L]


    if angle_of_incidence < 0 or angle_of_incidence > 180:
      rad_factor = 0
    return rad_factor, area, perimeter, characteristic_length, horizontal, upper, lengths

  geometries = ["WH1", "LH1", "LW1", "WH2", "LH2", "LW2"]

  geometry_results = {}

  for geo in geometries:
    results = get_geometry(geo)
    geometry_results[geo] = {
        "rad_factor": results[0],
        "area": results[1],
        "perimeter": results[2],
        "characteristic_length": results[3],
        "horizontal": results[4],
        "upper": results[5],
        "lengths": results[6]
    }

  def get_minimum_timestep():
    dts = []
    for i in range(len(geometries)):
      thickness = thicknesses[i]
      dts.append(stability_criterion(thickness, N, h_max, k, alpha_ABS, emissivity, T_max, q_flux_func(T_max+273.15)))

    return min(dts)

  areas = np.array([area for _, area, _, _, _, _ ,_ in (get_geometry(geo) for geo in geometries)]) # store areas

  #first: find the minimum timestep
  dt = get_minimum_timestep()

  #first: find the minimum timestep
  dt = get_minimum_timestep()
  if dt > 10: dt = 10

  # incase the timestep is too big relative to the run intervals
  while dt * 5 > (run_interval):
    run_interval = run_interval * 2
    interval = interval * 2

  logging.info(f"*** Timestep {round(dt, 3)} seconds ***")

  #create arrays
  time_steps = np.linspace(0,t_final,int(t_final/dt) +1)
  time_steps.shape

  Temperatures =  np.zeros((6, time_steps.shape[0], N), dtype='float32') #initialize 3D numpy array
  Temperatures[:,0,:] = T_initial # set first timestep of each face to the initial temperature

  avg_temp =  np.zeros((time_steps.shape[0]), dtype='float32')
  avg_temp[:] = T_initial

  battery_temp =  np.zeros((time_steps.shape[0]), dtype='float32')
  battery_temp[:] = T_initial

  outside_temp = np.zeros((time_steps.shape[0]), dtype='float32')
  outside_temp[:] = T_infinity_func(start_time_simulation_seconds, 0, dt, df)

  time_in_interval = 0
  interval_steps = int(interval/dt)
  # Perform time-stepping
  n = 1

  # print progress (at 10 discrete intervals)
  print_now = True
  print_step = 1
  
  while n < time_steps.shape[0]:
  #for n in range(1, time_steps.shape[0]):
    #logging.info(str(n) + str(time_in_interval))
    # Data queries perform only every 10 seconds to save substantial computing time
    if data_collect_now(n, interval, run_interval, time_in_interval, dt, data_collection_time_interval = 60):
      T_infinity = T_infinity_func(start_time_simulation_seconds, n, dt, df)
      angle_of_incidence = angle_of_incidence_func(start_time_simulation_seconds, n, dt, df, latitude, longitude, timezone)
      G = G_func(start_time_simulation_seconds, n, dt, df)
      #logging.info("G: " + str(G))
      air_velocity = air_velocity_func(start_time_simulation_seconds, n, dt, df)
      fan_speed = fan_speed_func(avg_temp[n-1])

      simulation_date_time = start_date_time + timedelta(seconds=(n * dt))

      if is_datetime_in_ranges(simulation_date_time, shading_ranges):
        G = G * 0.25

    outside_temp[n] = T_infinity
    
    # print progress (at 10 discrete intervals)
    if n  >= int(print_step * time_steps.shape[0]/10) and print_now == True:
      logging.info(f"  -  {round(n / time_steps.shape[0]* 100)}%: {round(time.time() - start_time)}")
      print_now = False
      print_step += 1
    elif n  >= int(print_step * time_steps.shape[0]/10) and print_now == False: 
      print_now = True
     
    # iterate through each geometry (each of the 6 faces of the box)#
    for i in range(len(geometries)):
      geo = geometries[i]
      rad_factor = geometry_results[geo]['rad_factor']
      area = geometry_results[geo]['area']
      perimeter = geometry_results[geo]['perimeter']
      characteristic_length = geometry_results[geo]['characteristic_length']
      horizontal = geometry_results[geo]['horizontal']
      upper = geometry_results[geo]['upper']
      lengths = geometry_results[geo]['lengths']

      if geo == "WH1" and box_shading: 
        new_G = G * 0.1
      else:
        new_G = G 
      thickness = thicknesses[i]
      dx = thickness / (N - 1)  # Grid spacing
      Fo = alpha_ABS * dt / dx**2


      # Create copy of temperature array for the next time step
      T_next = Temperatures[i,n, :] # (shape N)
      T_previous = Temperatures[i, n-1, :]

      # get h using convection correlations for external convection
      convection_coefficient = convection_func(T_surface = T_previous[0],
            T_infinity = T_infinity,
            characteristic_length = characteristic_length,
            area = area,
            perimeter = perimeter,
            v_air = v_air,
            Beta = B_air, 
            alpha_air = alpha_air,
            k_air = k_air,
            horizontal = horizontal,
            velocity = air_velocity,
      		upper = upper)

      # find fan speed and convection as a function of fan speed
      upper_inside = False
      if horizontal == True and upper == False: upper_inside = True

      convection_coefficient_internal = internal_convective_heat_transfer_func(
        fan_speed = fan_speed, 
        v_air = v_air, 
        k_air = k_air, 
        alpha_air = alpha_air, 
        perp_length1 = lengths[0],
        perp_length2 = lengths[1],
        T_surface = T_previous[0], 
        T_infinity = avg_temp[n-1], 
        characteristic_length = characteristic_length,
        area = area, 
        perimeter = perimeter, 
        horizontal = horizontal,  
        Beta = B_air, 
        upper = upper_inside
        )
      
      q = q_flux_func(avg_temp[n-1]) # get the radiation heat flux
      q_flux = q/np.sum(areas)

      Bi = convection_coefficient*dx/k #biot number for outside surface
      Bi_internal = convection_coefficient_internal*dx/k #biot number for inside surface

      # outer surface heat constraints
      # Conduction (2 * Fo * T), Convection (2 * Fo * Bi * T), Irradiation (2 * Fo * dx /k  * a * G), Radiation (2 * Fo * dx /k * sigma * a * T^4) 
      #logging.info(new_G)
      T_next[0] = 2 * Fo * (T_previous[1] + Bi * T_infinity + dx / k * absorptivity * new_G * rad_factor + dx /k * sigma * absorptivity * T_infinity ** 4) + (1 - 2 * Fo - 2 * Bi * Fo - Fo * 2 * dx / k * emissivity * sigma * T_previous[0]**3) * T_previous[0]

      if fan_speed > 0:
        bulk_air_temp = avg_temp[n-1] * (1- fan_mixing_ratio) + outside_temp[n-1] * fan_mixing_ratio
      else:
        bulk_air_temp = avg_temp[n-1]

      # Update temperature at interior points using finite difference scheme
      for m in range(1, N):
        if m == N-1:
          # conduction term (2 * Fo * T) | Heat gen term (2 Fo * dx /k U q_flux) | Internal convection term: 2 * Fo * Bi * T 
          if include_internal_effects:
            T_next[m] = T_previous[m] + 2 * Fo * (T_previous[m-1] - T_previous[m] + dx /k * q_flux * absorptivity + Bi_internal * (bulk_air_temp - T_previous[m]) )#+ dx/k * inner_q_rad_battery_flux) # for the internal heat generation constraint / term
          else: 
            T_next[m] = T_previous[m] + 2 * Fo * (T_previous[m-1] - T_previous[m] + dx /k * q_flux * absorptivity + Bi_internal * (bulk_air_temp - T_previous[m]))
        else:
          T_next[m] = T_previous[m] + Fo * (T_previous[m+1] + T_previous[m-1] - 2 * T_previous[m]) #

      Temperatures[i,n, :] = T_next

    avg_temp[n] = internal_radiation_T_func(Temperatures, areas, n-1) # weighted average by internal surface area

    battery_dT =  avg_temp[n] - avg_temp[n-1]
    battery_temp[n] = battery_temp[n-1] + battery_dT

    time_in_interval += dt

    #  *** to reduce complexity do linear approximations every interval ***
    if time_in_interval >= run_interval:
      end_of_write = n + 1 + interval_steps - int(time_in_interval/dt) # extrapolate form end of run_interval to end of total interval
      if end_of_write >= time_steps.shape[0]: end_of_write = time_steps.shape[0] # if at end of simulation 

      # change in temperature at most recent timestep
      dT = ((Temperatures[:, n, :] - Temperatures[:, n-1, :])).reshape(6, 1, Temperatures.shape[2])

      #linear approximations 
      Temperatures[:, n+1:end_of_write, :] = Temperatures[:, n, :].reshape(6, 1, Temperatures.shape[2]) + ( dT  *  np.linspace(1, end_of_write - (n+1), end_of_write - (n+1)).reshape(1, end_of_write - (n+1), 1) )      
      avg_temp[n+1:end_of_write] = avg_temp[n] + np.linspace(1, end_of_write - (n+1), end_of_write - (n+1))  * (avg_temp[n] - avg_temp[n-1]) 
      battery_temp[n+1:end_of_write] = battery_temp[n] + np.linspace(1, end_of_write - (n+1), end_of_write - (n+1))  * battery_dT 
      
      # try to do linear approximation for outside temperature
      try:
        outside_temp[n+1:end_of_write] = outside_temp[n] + np.linspace(1, end_of_write - (n+1), end_of_write - (n+1))  * ((outside_temp[n] - outside_temp[n-int(300/dt)])/(int(300/dt)))
      except Exception as e:
        outside_temp[n+1:end_of_write] = outside_temp[n]
      
      #reset interval times
      time_in_interval = 0
      n = end_of_write -1
    else:
      n += 1 # otherwise just proceed to next timestep

  logging.info(f"*** TOTAL TIME *** {round(time.time() - start_time)} seconds")
  return time_steps, Temperatures, avg_temp, battery_temp, dt, outside_temp #return time steps and temperature distribution