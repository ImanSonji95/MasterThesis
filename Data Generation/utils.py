from datetime import datetime, timedelta
import random
import pandas as pd
import matplotlib.pyplot as plt
import helpers
import numpy as np
import re
import plotly.graph_objects as go


# CONSTANTS
VITALS_SAMPLES_DAY = 288 # 1 sample every 5 minutes
ENV_SAMPLES_DAY = 24 # 1 sample every one hour
VITALS_ENV_RATIO = 12  # VITALS_SAMPLES_DAY/VITALS_ENV_RATIO

HR_RR_RATIO = 4 # heart rate to respiration rate ratio


environmental_df = pd.DataFrame()
medication_df = pd.DataFrame()
vitals_df = pd.DataFrame()

heart_rate = []
blood_pressure = []
opened_times = []

real_data_url = 'Data Generation/real_datasets/'
generated_data_url = 'Data Generation/generated_datasets/'

# medication lists
medication_dict = {'Levothyroxin': [], 'Pantoprazol': [], 'Hydrochlorothiazid': [], 'Calcium': [],
                   'Ramipril': [], 'Risperidon': [], 'Tamsulosin': [], 'Zopiclon': []
                   }

colors = [[0,  '#FFD43B'],  # discrete colorscale to map boolean
        [0.5,  '#FFD43B'],
        [0.5, '#4B8BBE '],
        [1, '#4B8BBE ']]


# generate Fitbit data
def generate_daily_activity_data():
    daily_activity = []
    activity_start_time = []
    location = []
    activity_df = pd.read_csv(real_data_url + 'fitbit_activity_data.csv') # contains daily data for one day
    for index, row in activity_df.iterrows():
        # generate random start time and activity
        start_time = datetime.strptime('14:30', '%H:%M') + timedelta(minutes=random.randint(10, 25))
        start_time = start_time.strftime('%H:%M')
        if row['minutesVeryActive'] > 8:
            daily_activity.append('running')
            activity_start_time.append(start_time)
            location.append(1)
        elif row['minutesLightlyActive'] > 100:
            daily_activity.append('walking')
            activity_start_time.append(start_time)
            location.append(1)
        else:
            daily_activity.append('resting')
            activity_start_time.append('08:00')
            location.append(0)
    activity_df['daily_activity'] = daily_activity
    activity_df['activity_start_time'] = activity_start_time
    activity_df['location'] = location
    return activity_df


# generate sleep data
def generate_sleep_activity(df):
    minutes_asleep = []
    minutes_awake = []
    sleep_start_time = []
    time_in_bed = []
    efficiency = []
    for i in range(0, len(df)):
        # generate random sleep times starting from 23:00
        mins_asleep = int(np.random.normal(7.5, 0.5) * 60)
        mins_awake = int(np.random.uniform(0.1, 0.5) * 60)
        total_time_in_bed = int(np.add(mins_asleep, mins_awake))
        sleep_time = datetime.strptime('23:00', '%H:%M') + timedelta(minutes=random.randint(5, 55))
        minutes_asleep.append(mins_asleep)
        minutes_awake.append(mins_awake)    
        time_in_bed.append(total_time_in_bed)
        efficiency.append(np.random.normal(85, 2.1))
        sleep_start_time.append(sleep_time.strftime("%H:%M:%S"))
    df['minutesAsleep'] = minutes_asleep
    df['minutesAwake'] = minutes_awake
    df['time_in_bed'] = total_time_in_bed
    df['efficiency'] = efficiency
    df['sleep_start_time'] = sleep_start_time
    return df


# environmental data generation
def generate_outdoor_environment():
    time_range = pd.Series(helpers.create_timerange_sixty('07/12/2021', '12/12/2021 23:00:00'))
    df =  helpers.read_file(real_data_url + 'outdoor_data.csv')
    outdoor_temps = []
    # generate outdoor temprarture and humidity
    for _ in np.split(time_range[:2232], len(time_range[:2232])/24):
        n = random.randint(8, 28)
        rand_temp = df['temperature'][:24].apply(lambda x: x + n)
        outdoor_temps.extend(rand_temp)
    for _ in np.split(time_range[2232:], len(time_range[2232:])/24):
        n = random.randint(-5, 4)
        rand_temp = df['temperature'][116:].apply(lambda x: x - n)
        outdoor_temps.extend(rand_temp)
    outdoor_hum = np.random.uniform(low=44, high=70, size=3696)
    outdoor_hum = pd.Series(outdoor_hum)
    outdoor_hum = outdoor_hum.apply(lambda x: round(x, 2))
    environmental_df['date'] = time_range
    environmental_df['outdoor_temperature'] = outdoor_temps
    environmental_df['outdoor_humidity'] = outdoor_hum


def generate_indoor_environment():
    df =  helpers.read_file(real_data_url + 'indoor_data.csv')  # contains data for one week sampled per 1 hour
    indoor_temps = []
    indoor_hums = []
    indoor_temps_ = []
    indoor_hums_ = []
    indoor_data = zip(df['time, indoor_temperature'], df['time, indoor_humidity'])
    for v1, v2 in indoor_data:
        result_1 = re.search(r'\"[0-9]{2}\.[0-9]{1,2}\"', v1)
        result_2 = re.search(r'\"[0-9]{2}\"', v2)
        indoor_temps_.append(float(result_1[0][1:-1]))
        indoor_hums_.append(int(result_2[0][1:-1]))
    for _ in range(0, 22):
        n = random.uniform(0.8, 1.2)
        indoor_temps.extend(n * l for l in indoor_temps_)
        indoor_hums.extend(n * l for l in indoor_hums_)
    environmental_df['indoor_temperature'] = [round(x, 3) for x in indoor_temps]
    environmental_df['indoor_humidity'] = [round(x, 3) for x in indoor_hums]
    return environmental_df


def generate_normal_vitals(night_range, morning_range, noon_range,evening_range):
    # heart rate
    heart_rate.extend(helpers.generate_random_data(63.5, 5.6, night_range))
    heart_rate.extend(helpers.generate_random_data(76.1, 9.6, morning_range))
    heart_rate.extend(helpers.generate_random_data(63.5, 5.6, noon_range))
    heart_rate.extend(helpers.generate_random_data(63.5, 5.6, evening_range))
    # blood pressure
    blood_pressure.extend(helpers.generate_random_data(75.1, 9.8, night_range))
    blood_pressure.extend(helpers.generate_random_data(89.9, 10.4, morning_range))
    blood_pressure.extend(helpers.generate_random_data(89.9, 10.4, noon_range))
    blood_pressure.extend(helpers.generate_random_data(75.1, 9.8, evening_range)) 
    return heart_rate, blood_pressure


def generate_abnormal_vitals(night_range, morning_range, noon_range,evening_range):
    # heart rate
    heart_rate.extend(helpers.generate_random_data(63.5, 5.6, night_range))
    heart_rate.extend(helpers.generate_random_data(100.5, 5.6, morning_range))
    heart_rate.extend(helpers.generate_random_data(120.1, 5.2, noon_range))
    heart_rate.extend(helpers.generate_random_data(63.5, 5.6, evening_range))
    # blood pressure
    blood_pressure.extend(helpers.generate_random_data(80.1, 9.8, night_range))
    blood_pressure.extend(helpers.generate_random_data(90.1, 9.8, morning_range))
    blood_pressure.extend(helpers.generate_random_data(92.9, 10.4, noon_range))
    blood_pressure.extend(helpers.generate_random_data(80.1, 9.8, evening_range)) 
    return heart_rate, blood_pressure


# medication adherence generation
def generate_normal_adherence():  
    for key in medication_dict.keys():
        medication_dict[key].append(1)


def generate_abnormal_adherence():
    rand_values = []
    for i in range(4):
        rand_values.extend([random.randint(0, 1)] * 2)
    for index, key in enumerate(medication_dict):
        medication_dict[key].append(rand_values[index])


def bool_to_string(A):
    #convert a binary array into an array of strings 'True' and 'False'
    S = np.empty(A.shape,  dtype=object)
    S[np.where(A)] = 'True'
    S[np.where(A==0)] = 'False'
    return S


def write_medication_adherence():
    # medicaiton adherence add dates
    dates = helpers.create_timerange_day('07/12/2021', '12/12/2021')
    medication_df['date'] = dates
    for key, value in medication_dict.items():
        medication_df[key] = value
    
    helpers.export_data_to_csv(medication_df, generated_data_url + 'medication.csv')
    # plot medication adherence
    plot_medication_adherence()


def plot_medication_adherence():
    A = medication_df[['Levothyroxin', 'Pantoprazol', 'Hydrochlorothiazid', 'Calcium', 'Ramipril',
                       'Risperidon', 'Tamsulosin', 'Zopiclon']]
    A =  A.values.T
    groups = list(medication_df.columns.difference(['date']))
    fig = go.Figure(data=go.Heatmap(z=A, y=groups, x=medication_df['date'], coloraxis='coloraxis',
                                    xgap=1, ygap=1, customdata=bool_to_string(A)))
    fig.update_layout(title_text='Medication Adherence', title_x=0.5,
                      width=799, height=400,
                      coloraxis=dict(colorscale=colors, showscale=False))
    fig.show()
    

# vitals data generation
def generate_vitals():
    date_range = helpers.create_timerange_five('07/12/2021', '12/12/2021 23:59:00')
    num_days = len(date_range)/VITALS_SAMPLES_DAY
    heart_rate_activity = []
    vitals_df['date'] = list(date_range)
    physical_activities = pd.read_csv(real_data_url + 'fitbit_activity_data.csv')
    environmental_data = pd.read_csv(generated_data_url + 'environmental.csv')
    # correlate vitals and medication
    for range_ in np.split(date_range, int(num_days)): # for each day 
        # day = list(set(range_.strftime('%Y-%m-%d')))
        night_range = range_[:96]  # adjust ranges based on number of samples per day
        morning_range = range_[96:144]
        noon_range = range_[144:240]
        evening_range = range_[240:]
        if np.random.uniform(0, 1) <= 0.6:
            heart_rate, blood_pressure = generate_normal_vitals(night_range, morning_range, noon_range, evening_range)
            generate_normal_adherence()
            # generate_normal_medication_times(day)
        else:
            heart_rate, blood_pressure = generate_abnormal_vitals(night_range, morning_range, noon_range, evening_range)
            if np.random.uniform(0, 1) <= 0.1:             
                generate_normal_adherence()
                # generate_normal_medication_times(day)
            else: 
                generate_abnormal_adherence()
                # generate_abnormal_medication_times()
    # correlate vitals and activity
    for index, row in physical_activities.iterrows():
        n = index * VITALS_SAMPLES_DAY
        daily_heart_rate = heart_rate[n:n + VITALS_SAMPLES_DAY]
        if row['daily_activity'] in ('running'):
            start_time = int(datetime.strptime(row['activity_start_time'], '%H:%M').strftime("%H"))
            m = start_time * VITALS_ENV_RATIO
            for i in range(m - 20, m + 20):
                daily_heart_rate[i] = 130 * np.random.uniform(1, 1.15) # highest is 150 bpm
        elif row['daily_activity'] in ('walking'):
            start_time = int(datetime.strptime(row['activity_start_time'], '%H:%M').strftime("%H"))
            m = start_time * VITALS_ENV_RATIO
            for i in range(m - 20, m + 20):
                daily_heart_rate[i] = 100 * np.random.uniform(1, 1.15) # highest is 150 bpm
        heart_rate_activity.extend(daily_heart_rate)
    vitals_df['heart_rate'] = heart_rate_activity
    vitals_df['blood_pressure'] = blood_pressure
    vitals_df['respiration_rate'] = vitals_df['heart_rate'].div(HR_RR_RATIO)
    # correlate skin temperature and indoor temperature
    skin_temperature = np.random.normal(loc=33, scale=0.5, size=len(date_range))
    for index, row in environmental_data.iterrows():
        if row['indoor_temperature'] > 27:
            skin_temperature[index * VITALS_ENV_RATIO:index * VITALS_ENV_RATIO + 8] *= 1.07
    vitals_df['skin_temprature'] = skin_temperature
    # write medication adherence
    write_medication_adherence()
    # write_medication_times()
    return vitals_df



def generate_normal_medication_times(time, day, n):
    opened_times.append(datetime.strptime(
        day + ' ' + time, '%Y-%m-%d %H:%M:%S') + timedelta(minutes=np.random.uniform(1, n)))
    # opened_times.append(datetime.strptime(
    #     day[0] + ' 13:00:00', '%Y-%m-%d %H:%M:%S') + timedelta(minutes=np.random.uniform(1, 59)))
    # opened_times.append(datetime.strptime(
    #     day[0] + ' 17:00:00', '%Y-%m-%d %H:%M:%S') + timedelta(minutes=np.random.uniform(1, 180)))
    # opened_times.append(datetime.strptime(
    #     day[0] + ' 21:00:00', '%Y-%m-%d %H:%M:%S') + timedelta(minutes=np.random.uniform(1, 59)))


def generate_abnormal_medication_times():
    opened_times.append('not_opened')


def generate_medication_times():
    medication_dates = helpers.create_timerange_day('07/12/2021', '12/12/2021')
    index = pd.MultiIndex.from_product([medication_dates,['morning', 'noon', 'evening', 'night']])
    index.set_names(['date', 'daytime'], inplace=True)
    medication_times_df = pd.DataFrame(index=index)
    # correlate medication taken and medication times
    df = pd.read_csv(generated_data_url + 'medication.csv')
    for index, row in df.iterrows():
        medication = row.tolist()
        if (medication[2] == 1):
            generate_normal_medication_times('08:00:00', medication[0], n=59)
        else:
            generate_abnormal_medication_times()
        if (medication[4] == 1):
            generate_normal_medication_times('13:00:00', medication[0], n=180)
        else:
            generate_abnormal_medication_times()
        if (medication[6] == 1):
            generate_normal_medication_times('17:00:00', medication[0], n=180)
        else:
            generate_abnormal_medication_times()
        if (medication[8] == 1):
            generate_normal_medication_times('21:00:00', medication[0], n=59)
        else:
            generate_abnormal_medication_times()
    medication_times_df['triggered_times'] = ['08:00', '13:00', '17:00', '21:00'] * 154
    medication_times_df['opened_times'] = opened_times
    return medication_times_df


def plot_data(df):
    df.plot(x='date', figsize=(12, 4), subplots=True)   
    plt.show()
    