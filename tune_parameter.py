"""
This script tunes parameters for a model using real sensor data, comparing 
model results with real data to minimize error.
"""

# Import necessary libraries for numerical modeling and data processing
import numpy as np
import matplotlib.pyplot as plt
from numerical_modelling import *
from datetime import datetime
from validation import *
import pandas as pd
from itertools import product

def tune_parameters(start_date_time, end_date_time, raspi_path, parameters, values_list, run_func):
    """
    Tune model parameters by running simulations with different parameter values and
    comparing the results to real sensor data. The function identifies the best 
    combination of parameters by minimizing the root mean squared error (RMSE).

    Arguments:
    - start_date_time (str): The start date and time for the data to be analyzed.
    - end_date_time (str): The end date and time for the data to be analyzed.
    - raspi_path (str): Path to the real sensor readings CSV file.
    - parameters (list): A list of model parameters to tune (string versions of the param names).
    - values_list (list of lists): A list of value ranges for each parameter to test.
    - run_func (function): Function to execute the model with given parameters.

    Returns:
    - final_parameters (list): The optimized parameters that yielded the lowest RMSE.
    """

    # Set up logging for the tuning process
    import logging
    filename = 'validation.log'
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # Clear the log file before running
    with open(filename, 'w'):
        pass  # This opens and immediately closes the file, truncating it

    # Load and smooth real sensor data to use in validation
    _, smoothed_df = import_data(raspi_path, logging, 9, 2)

    def objective_function(params, run_func):
        """
        Evaluate the model's performance by running it with the given parameter set
        and calculating the RMSE between the model's output and real data.

        Arguments:
        - params (tuple): Current set of parameter values being tested.
        - run_func (function): Function that runs the model.

        Returns:
        - RMSE (float): The root mean squared error between the model's predictions
                        and real sensor data.
        - model_df (DataFrame): The model's output data as a pandas DataFrame.
        """
        # Map parameter names to their values
        param_dict = {parameters[i]: params[i] for i in range(len(parameters))}
        print(f"Parameters = {param_dict}")
         
        # Run the model with the current parameter set
        results_dir = run_func(start_date_time=start_date_time, end_date_time=end_date_time, logging=logging, **param_dict)

        # Load the model output and merge with real data
        model_df = load_data_FoCO(results_dir, datetime.strptime(start_date_time, '%Y-%m-%d %H:%M:%S'))
        shortened_df = smoothed_df.loc[(smoothed_df.index >= start_date_time) & (smoothed_df.index <= end_date_time)]
        merged = pd.merge_asof(shortened_df, model_df, left_on='timestamp', right_on='datetime', direction='nearest')

        # Calculate RMSE between the model's predictions and real sensor data
        RMSE = np.sqrt(((merged['4_smooth']+273.15) - merged['internal'])**2).mean()
        return RMSE, model_df

    # Generate all combinations of parameter values to test
    parameter_combinations = list(product(*values_list))

    # Initialize variables to track the best parameters and their error
    results_dicts = []
    best_params = None
    best_error = float('inf')
    best_model_df = None

    # Iterate through each combination of parameter values
    for params in parameter_combinations:
        mean_error, model_df = objective_function(params, run_func)
        results_dicts.append({'data': model_df, 'label': f"Params {params}"})
        print(params, "error:", mean_error)
        
        # Update best parameters if the current set yields a lower error
        if mean_error < best_error:
            best_error = mean_error
            best_params = params
            best_model_df = model_df

    # Plot the comparison of model outputs for all parameter sets
    plot_comparison(smoothed_df, results_dicts, start_date_time, end_date_time)

    # Output the best parameters and their corresponding error
    final_parameters = best_params
    print(f"\nFinal optimized parameters: {final_parameters} with error = {best_error}")

    # Plot the results with the optimized parameters
    plot_comparison(smoothed_df, [{'data': best_model_df, 'label': "Optimized Parameters"}], start_date_time, end_date_time)

    return final_parameters

# Example function call with sample data and parameters
tune_parameters(start_date_time='2024-06-26 17:00:00',
               end_date_time='2024-06-28 15:00:00',
               raspi_path="../real_data/sensor_readings_2024_06_26_16_45.csv",
               parameters=["N"],
               values_list=[np.linspace(2, 10, 9)],
               run_func=run_foco)
