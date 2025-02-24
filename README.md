# Heat Transfer and Solar Charging Tool for Air Pollutant Monitors (Version 1.0 [2025])

## Overview

This repository contains a suite of Python scripts for running the predictive solar radiation and heat transfer models designed for air pollutant monitors. These models are integrated into a Plotly-Dash web application, allowing users to simulate heat transfer dynamics, solar energy generation, and other related physical processes to predict the internal temperature and solar charging of air pollutant monitors.. The application supports multiple simultaneous user sessions and can be deployed on local machines or compute servers.

## Main Features

- **Interactive Web Application**: A Plotly-Dash app that allows users to input simulation parameters and visualize results in real-time.
- **Multiple Simulation Models**: Includes separate models for heat transfer, solar energy generation, and combined systems, with support for geographic customization.
- **User Session Management**: Automatically creates unique log files for each user session to track inputs, outputs, and errors.
- **Data Collection and Logging**: Code for Raspberry Pi devices to collect and log temperature data using thermocouples, integrated into the application.
- **Uncertainty Analysis**: Includes options for bounding simulation results using quantiles to account for uncertainty.
- **Real-Time Data Comparison**: Models can be validated using real-world data and visualized through the app.

## Installation Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/kyanshlipak/HT_monitor_model_public.git
cd HT_monitor_model_public
```

### Step 2: Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Running the Application

```bash
python app.py
```

### Step 5: Demo inputs

Test a demo run by setting the latitude and longitude to your location and using the below parameters: Below is an image of the web application interface with example parameters:

![Demo parameters](demo_parameters.png)

The model will typically take just a few minutes to run, depending on the conductivity of the material chosen, duration of modeling, whether or not a solar panel is included, and if 10% & 90% bounds are used. A log output will print updates in real time.

### Step 6: Demo outputs

The demo outputs for the simulation run through the app in Step 5 and downloaded via the "Download Files" button is included in the demo_data folder. Outputs include the upper temperature bound, lower temperature bound, 2023 period modeled, and 10 year solar charging dataframes in the form of pickle files. "Pickle" .pkl files are used due to their low data storage requirements. They can be read into a dataframe using the pandas python library, among other options.

### Expected Installation Time

The complete installation process, including dependency installation, should take approximately 5-10 minutes, depending on your system and internet speed.

### Compatible Python Versions

This tool is designed for Python 3.8.8, but it is also compatible with at least Python 3.7+. Other Python3 versions may be compatible but are not recommended.


## Usage

- **General tool usa cases**: We recommend using this tool to get a better prediction of anticipated internal temperatures over time in a given air pollutant monitor for deployment somewhere. The failure rates outputted should be a valuable statistic when designing low-cost sensor campaigns, especially in hot places or locations without wall power.
- **Heat Transfer and Solar Charging Modeling**: For basic, default modeling, use app.py or our locally hosted site (will be published shortly). For more specific modeling of a known location with a higher resolution data source than the NASA power api, use numerical_modelling.py and the validation.py script to slightly adjust the run_foco.py script for your specific location and data inputs.
- **Tune parameter for future modeling**: Use tune_parameter.py with real collected data and run the model to tune specific model inputs empirically, like emissivity of the monitor enclosure.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE.md) file for details.
