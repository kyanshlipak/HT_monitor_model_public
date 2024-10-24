"""
raspi_code.py

Python code used to collect and log temperature data using two 8 thermocouple breakout boards

Author: Kyan Shlipak
Date: 09/28
"""

import spidev
import RPi.GPIO as GPIO
import time
import pandas as pd
from datetime import datetime
import Adafruit_GPIO.SPI as SPI
import Adafruit_MAX31855.MAX31855 as MAX31855
import numpy as np

# Raspberry Pi hardware SPI configuration.
SPI_PORT = 0
SPI_DEVICE_1 = 0  # CE0
SPI_DEVICE_2 = 1  # CE1

# Create a MAX31855 sensor instance.
spi_1 = SPI.SpiDev(SPI_PORT, SPI_DEVICE_1, max_speed_hz=5000000)
spi_2 = SPI.SpiDev(SPI_PORT, SPI_DEVICE_2, max_speed_hz=5000000)

sensor_1 = MAX31855.MAX31855(spi=spi_1)
sensor_2 = MAX31855.MAX31855(spi=spi_2)

GPIO.setwarnings(True)
GPIO.cleanup()  # Clean up GPIO on exit
GPIO.setmode(GPIO.BCM)


# Setup GPIO
T0 = 17  # Example GPIO pin for T0
T1 = 27  # Example GPIO pin for T1
T2 = 22  # Example GPIO pin for T2
ceo_pin = 8
ceo_pin2 = 7

GPIO.setup(T0, GPIO.OUT) # output pins to determine which sensor to read from
GPIO.setup(T1, GPIO.OUT)
GPIO.setup(T2, GPIO.OUT)

# SPI Comms Set chip select pins
GPIO.setup(ceo_pin, GPIO.OUT)
GPIO.setup(ceo_pin2, GPIO.OUT)
GPIO.output(ceo_pin, GPIO.LOW)
GPIO.output(ceo_pin2, GPIO.LOW)

# current  time
now = datetime.now()
date_string = now.strftime("%Y_%m_%d_%H_%M")
file_name = "sensor_readings_" + date_string + ".csv"

# Function to set T0, T1, and T2 pins to select a sensor
def select_sensor(sensor_num):
	assert 0 <= sensor_num < 8, "Sensor number must be between 0 and 7"
	binary = bin(sensor_num).zfill(3)[-3:].replace('b', '0')
	binary = list(binary)
	GPIO.output(T0, to_GPIO(int(binary[2])))
	GPIO.output(T1, to_GPIO(int(binary[1])))
	GPIO.output(T2, to_GPIO(int(binary[0])))

def to_GPIO(num):
	if num == 0: return GPIO.LOW
	else: return GPIO.HIGH

# Function to read from the selected sensor
def read_sensor(sensor):
	temp =  sensor.readTempC()
	return temp

# Initialize DataFrame
columns = ['timestamp'] + [f'sensor_{i}' for i in range(8)] + [f'sensor_{8+i}' for i in range(8)]

df = pd.DataFrame(columns=columns)

# Main loop to read from each sensor
interval = 1  # how often to get data from the next sensor
save_interval = 60  # Save data every 60 seconds
next_save_time = time.time() + save_interval
try:
	start_time = time.time() # start of process
	while True:
		data_row = {'timestamp': datetime.now()} #dictionary for row of pandas data
		for sensor_num in range(8):
			select_sensor(sensor_num) # set GPIO T output pins to select sensor
			time.sleep(0.125)  # Small delay to ensure the sensor selection settles
			sensor_value_1 = read_sensor(sensor_1) # read data from the sensor
			sensor_value_2 = read_sensor(sensor_2) # read data from the sensor
			now = datetime.now()
			date_string = now.strftime("%Y/%M/%D %H:%M:%S")
			data_row[f'sensor_{sensor_num}'] = sensor_value_1 # set to sensor value
			data_row[f'sensor_{sensor_num+8}'] = sensor_value_2 # set to sensor value
			print(f"timestamp {date_string} sensor {sensor_num} value: {sensor_value_1}") # print output
			print(f"timestamp {date_string} sensor {sensor_num+8} value: {sensor_value_2}") # print output

			elapsed_time = time.time() - start_time # check how much time this took
			sleep_time = interval - elapsed_time # sleep for whatever leftover time

			if sleep_time > 0:
				time.sleep(sleep_time)

			# Update start_time for the next iteration
			start_time += interval


		# Add the data row to the DataFrame
		df.loc[len(df)] =  data_row

		# Check if it's time to save the data
		if time.time() >= next_save_time:
			# Save DataFrame to CSV
			df.to_csv(file_name, index=False)
			# Update next save time
			next_save_time += save_interval

except Exception as e:
	print(f"Error occurred: {e}")

	# Save DataFrame
	df.to_csv(file_name, index=False)
finally:
	# Close SPI and cleanup GPIO
	GPIO.cleanup()