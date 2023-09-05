from flask import Flask, render_template
from pandas._libs.tslibs import timestamps
import plotly.graph_objects as go
from _plotly_utils.utils import PlotlyJSONEncoder
import pandas as pd
import plotly.express as px
from itertools import cycle
from statistics import mean
import numpy as np
from pycaret.anomaly import *
import json
from models import *
import datetime


data_url = 'Data Generation/generated_datasets/'

# Configure api
app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='template')
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False


colors = [[0,  '#FFD43B'],  # discrete colorscale to map boolean
        [0.5,  '#FFD43B'],
        [0.5, '#4B8BBE '],
        [1, '#4B8BBE ']]



def get_normal_vitals_activity():
    vital_records = get_vitals_nodes()
    vital_nodes = [record['n.name'] for record in vital_records]
    acitivity_records = get_activities_nodes()
    activity_nodes = list(set([record['m.name'] for record in acitivity_records]))
    activity_thresholds = {}

    # get thresholds
    for vital in vital_nodes:
        activity_thresholds[vital] = {}
        for activity in activity_nodes:
            # get threshold
            threshold_result = find_vital_threshold('activity: ' + str(activity), str(vital))
            thresholds_list = [record['m.name'] for record in threshold_result]
            if len(thresholds_list) != 0:
                # take only first value
                match = re.search("(?<=\[).+?(?=\])", thresholds_list[0])
                if match:
                    thresholds = match.group()
                    min_max_values = list(thresholds.split(","))
                    activity_thresholds[vital][activity] = min_max_values

    fig = go.Figure(data=[go.Table(columnorder = [1, 2], columnwidth = [125, 300],
                    header=dict(values=['Vitals', 'Normal Ranges'],
                                fill_color='paleturquoise',
                                align='left',
                                height=33),
                    cells=dict(values=[list(activity_thresholds.keys()),
                               [json.dumps(s) for s in activity_thresholds.values()]],
                    fill_color='lavender',
                    align='left',
                    height = 33
                    ))
                ])
    fig.update_layout(title_text='Normal Vitals Range', title_x=0.5, width=800)
    normal_vitals = json.dumps(fig, cls=PlotlyJSONEncoder)
    return normal_vitals, activity_thresholds


def get_medicines():
    f = open('Context Learning/medication_plan.json')
    medicaiton_plan = json.load(f)
    medicines = []
    dosages = []
    doses_per_day = []
    data = medicaiton_plan['medication_plan']['medicines']
    for i in range(1, len(data)):
        medicines.append(data['medicine_' + str(i)]['active_ingredient_str'])
        dosages.append(data['medicine_' + str(i)]['dosage'])
        doses_per_day.append(data['medicine_' + str(i)]['dose_per_day'])

    doses = [str(x) + ' mg ' + str(y) + ' times per day' for x, y in zip(dosages, doses_per_day)]

    # calculate medication adherence
    df = pd.read_csv(data_url + 'medication.csv')
    A = df.columns.difference(['date'])
    adherence_rates = []
    for column in A:
        adherence_rates.append(round((df[column] == 1).sum()/len(df[column]), 2))
    adherence_percentages = [str(x*100) + ' %' for x in adherence_rates]

    fig = go.Figure(data=[go.Table(columnorder = [1,2, 3], columnwidth = [100, 100, 80],
                    header=dict(values=['Medication', 'Dose', 'Adherence'],
                    fill_color='paleturquoise',
                    height = 20,
                    align='left'),
                    cells=dict(values=[medicines,
                               doses, adherence_percentages],
                    fill_color='lavender',
                    height = 20,
                    align='left'
                    ))
                ])
    fig.update_layout(title_text='Medication List', title_x=0.5, width=450)
    med_table = json.dumps(fig, cls=PlotlyJSONEncoder)
    return med_table


def bool_to_string(A):
    #convert a binary array into an array of strings 'True' and 'False'
    S = np.empty(A.shape,  dtype=object)
    S[np.where(A)] = 'True'
    S[np.where(A==0)] = 'False'
    return S


def get_medication_adherence():
    med_df = pd.read_csv(data_url + 'medication.csv')

    # get medicines list
    f = open('Context Learning/medication_plan.json')
    medicaiton_plan = json.load(f)
    data = medicaiton_plan['medication_plan']['medicines']
    medicines_list = []
    for i in range(1, len(data)):
        medicines_list.append(data['medicine_' + str(i)]['active_ingredient_str'])

    A = med_df[medicines_list]
    A =  A.values.T
    # groups = list(med_df.columns.difference(['date']))
    fig = go.Figure(data=go.Heatmap(z=A, y=medicines_list, x=med_df['date'], coloraxis='coloraxis',
                                    xgap=1, ygap=1, customdata=bool_to_string(A)))
    fig.update_layout(title_text='Medication Adherence', title_x=0.5,
                      width=800, height=400,
                      coloraxis=dict(colorscale=colors))
    fig = fig.update_xaxes(rangeslider_visible=True)
    med_plot_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return med_plot_json


def get_vitals_analysis():
    # plot heart rate
    vitals_df = pd.read_csv(data_url + 'vitals.csv')
    resulted_df = detect_anomalies(vitals_df)
    resulted_df.to_csv('anomalies.csv')

    fig = px.line(resulted_df, x=resulted_df.index, y=['heart_rate'])
    var_labels = cycle(['HR (bpm)'])
    fig.for_each_trace  (lambda t:  t.update(name = next(var_labels)))
    fig = fig.update_xaxes(rangeslider_visible=True)

    # create list of outlier_dates
    outlier_dates = resulted_df[resulted_df['Anomaly'] == 1].index
    y_values = [resulted_df.loc[i]['heart_rate'] for i in outlier_dates]
    fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers',
                name = 'Anomaly',
                marker=dict(color='red',size=10)))
    fig.update_layout(title_text='User Vitals & Activity', title_x=0.5,
                      width=850, height=400, autosize=True)

    vitals_plot_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    return vitals_plot_json


def detect_anomalies(vitals_df):
    # dataframe containing heart rate anomalies
    data = pd.DataFrame()

    vitals_df.drop(['blood_pressure', 'respiration_rate', 'skin_temprature'], axis=1, inplace=True)
    vitals_df['date'] = pd.to_datetime(vitals_df['date'])
    timestamps = pd.date_range(start='07/12/2021', end='12/12/2021 23:00:00', freq='60Min')

    data['timestamps'] = timestamps
    data['heart_rate'] = [mean(vitals_df['heart_rate'][i*12:i*12+12]) for i in range(len(timestamps))]

    # create moving-average
    data['MA48'] = data['heart_rate'].rolling(48).mean()
    data['MA336'] = data['heart_rate'].rolling(336).mean()

    # drop moving-average columns
    data.drop(['MA48', 'MA336'], axis=1, inplace=True)

    # set timestamp to index
    data.set_index('timestamps', drop=True, inplace=True)

    # resample timeseries to hourly
    data = data.resample('H').sum()

    # creature features from date
    data['day'] = [i.day for i in data.index]
    data['day_name'] = [i.day_name() for i in data.index]
    data['day_of_year'] = [i.dayofyear for i in data.index]
    data['week_of_year'] = [i.weekofyear for i in data.index]
    data['hour'] = [i.hour for i in data.index]
    data['is_weekday'] = [i.isoweekday() for i in data.index]

    # init setup
    s = setup(data, session_id = 123)

    # train model
    iforest = create_model('lof')
    iforest_results = assign_model(iforest)
    iforest_results.head()

    # check anomalies
    iforest_results[iforest_results['Anomaly'] == 1].head()

    return iforest_results


def get_heart_rate_analysis():
    anomalies_df = pd.read_csv('anomalies.csv')

    # get timestamps and heart rate values where anomaly occurs
    timestamps = anomalies_df.loc[anomalies_df['Anomaly'] == 1, 'timestamps']
    heart_rate_values = anomalies_df.loc[anomalies_df['Anomaly'] == 1, 'heart_rate']
    heart_rate_values = list(map(lambda x: int(x), heart_rate_values))
    possible_causes = []
    related_context = []
    quality_list = []
    for ts, value in zip(timestamps, heart_rate_values):
        # find co-occurring nodes
        current_date = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').strftime('_%Y_%m_%dT%H')
        previous_date = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours=2)
        previous_date = previous_date.strftime('_%Y_%m_%dT%H')
        # get co-occuring and previous occuring nodes
        cooccuring_related_nodes = get_related_cooccurring_nodes('heart_rate', current_date)
        previous_related_nodes = get_related_cooccurring_nodes('heart_rate', previous_date)
       
        # compare current and previous nodes
        common_nodes = [x for x in cooccuring_related_nodes if x not in previous_related_nodes]
        related_nodes = [x for x in cooccuring_related_nodes if x not in common_nodes]
        possible_causes.append(', '.join(common_nodes))
        related_context.append(', '.join(related_nodes))

        # check if heart rate higher or lower than normal range
        current_activity = get_activity(current_date)
        threshold = get_threshold('heart_rate', current_activity[0])
        min = float(threshold[0][0])
        max = float(threshold[0][1])
        if value <= min:
            quality_list.append('Low')
        elif value >= max:
            quality_list.append('High')
        else:
            quality_list.append('Normal')

    # draw table
    fig = go.Figure(data=[go.Table(columnorder = [1,2, 3, 4, 5], columnwidth = [150, 80, 500, 500, 100],
                    header=dict(values=['Heart rate Anomalies', 'Value','Possible causes', 'Related Context', 'Range'],
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[timestamps, heart_rate_values, possible_causes, related_context, quality_list],
                    fill_color='lavender',
                    align=['left', 'center', 'left', 'left'],
                    height = 50
                    ))
                ])
    fig.update_layout(title_text='Heart Rate Analysis', title_x=0.5, width=1330)
    heart_rate_analysis = json.dumps(fig, cls=PlotlyJSONEncoder)
    return heart_rate_analysis


def detect_medication_anomalies():
    df_medication = pd.read_csv('Context Learning/context_states.csv')
    # get medication rates and timestamps
    timestamps = df_medication['day'] + ' ' + df_medication['time']
    medication_rates = df_medication['medication_rate']
    dates = []
    medication_times = []
    vitals_effects = []
    # get anomalies
    for ts, rate in zip(timestamps, medication_rates):
        day = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M').strftime('_%Y_%m_%dT%H')
        if (rate != 'null' and rate < 0.75):
            dates.append(day)
            coocurring_nodes = get_related_cooccurring_nodes('medication_rate', day)
            effects = [x for x in coocurring_nodes if
                       any(y in x for y in ['heart_rate', 'blood_pressure', 'respiration_rate'])]
            if (len(effects) != 0):
                vitals_effects.append(effects)
            else:
                vitals_effects.append('No side effects found')
            if (rate == 0.5):
                medication_times.append('Deviation by 1 to 2 hours')
            elif (rate == 0.25):
                medication_times.append('Deviation by 2 hours')
            elif (rate == 0.0):
                medication_times.append('Medication not taken')

     # draw table
    fig = go.Figure(data=[go.Table(columnorder = [1, 2, 3], columnwidth = [150, 300, 800],
                    header=dict(values=['Medication Anomalies', 'Medication Time','Vitals Side Effects'],
                                fill_color='paleturquoise',
                                height = 20,
                                align='left'),
                    cells=dict(values=[timestamps, medication_times, vitals_effects],
                    fill_color='lavender',
                    height = 20,
                    align='left'
                    ))
                ])
    fig.update_layout(title_text='Medication Rate Analaysis', title_x=0.5, width=1250)
    medication_rate_analysis = json.dumps(fig, cls=PlotlyJSONEncoder)
    return medication_rate_analysis


@app.route('/main_menu')
def main_menu():
    return render_template('main_menu.html')


@app.route('/context_model')
def context_model():
    return render_template('context_model_visualization.html')


@app.route('/data_monitoring')
def data_monitoring():
    user = get_user_information()
    med_table = get_medicines()
    med_plot_json = get_medication_adherence()
    normal_vitals_json = get_normal_vitals_activity()
    vitals_plot_json = get_vitals_analysis()
    vitals_table_json = get_heart_rate_analysis()
    medication_table_json = detect_medication_anomalies()
    return render_template('data_monitoring.html', name=user[0]['name'], age=int(user[0]['age']), 
                           gender=user[0]['gender'], weight=user[0]['weight'], med_table=med_table, med_plot_json=med_plot_json,
                           vitals_plot_json=vitals_plot_json, vitals_table_json=vitals_table_json,
                           medication_table_json=medication_table_json,
                           normal_vitals_json=normal_vitals_json)


if __name__ == '__main__':
    app.run()