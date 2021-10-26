"""
*
* Created by    -   Iman Sonji (MT 3217)
* Creation date -   02 July 2021
* Description   -   The purpose of this module is to generate synthetic data for a long period of time based 
* on previously accquired real datasets. 
"""

import helpers
from utils import *


def generate_synthetic_data():
    # generate activity data
    activity_data = generate_daily_activity_data()
    # generate sleep data
    activity_sleep_data = generate_sleep_activity(activity_data)
    helpers.export_data_to_csv(activity_sleep_data, generated_data_url + 'fitbit_activity_data.csv')
    # generate environmental data
    generate_outdoor_environment()
    environmental_data = generate_indoor_environment()
    helpers.export_data_to_csv(environmental_data, generated_data_url + 'environmental.csv')
    # generate vitals & medication
    vitals_data = generate_vitals()
    vitals_data = vitals_data.astype({'heart_rate': 'int32', 'blood_pressure': 'int32', 'respiration_rate': 'int32', 
        'skin_temprature': 'int32'}, copy=False)
    helpers.export_data_to_csv(vitals_data, generated_data_url + 'vitals.csv')
    # generate medication times
    medication_times = generate_medication_times()
    medication_times.to_csv(generated_data_url + 'medication_times.csv')
    return activity_sleep_data, environmental_data, vitals_data


def plot_generated_data(activity_sleep_data, env_data, vitals_data):
    # plots
    plot_data(activity_sleep_data)
    plot_data(env_data)
    plot_data(vitals_data)


if __name__ == "__main__":
    activity_sleep_data, env_data, vitals_data = generate_synthetic_data()
    plot_generated_data(activity_sleep_data, env_data, vitals_data)