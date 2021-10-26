import pandas as pd
import numpy as np


def create_timerange_five(start_time, end_time):
    return pd.date_range(start=start_time, end=end_time, freq='5Min')

def create_timerange_fifteen(start_time, end_time):
    return pd.date_range(start=start_time, end=end_time, freq='15Min')

def create_timerange_thirty(start_time, end_time):
    return pd.date_range(start=start_time, end=end_time, freq='30Min')

def create_timerange_sixty(start_time, end_time):
    return pd.date_range(start=start_time, end=end_time, freq='60Min')

def create_timerange_three_hours(start_time, end_time):
    return pd.date_range(start=start_time, end=end_time, freq='3H')

def create_timerange_day(start_time, end_time):
    return pd.date_range(start=start_time, end=end_time, freq='1D')

def generate_random_data(mu, sigma, date_range):
    return (np.random.normal(loc=mu, scale=sigma, size=len(date_range))).tolist()

def export_data_to_csv(df, output_file):
    df.to_csv(output_file, index=False)

def read_file(input_file):
    return pd.read_csv(input_file)
    