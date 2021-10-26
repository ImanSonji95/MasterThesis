"""
*
* Created by    -   Iman Sonji (MT 3217)
* Creation date -   06 August 2021
* Description   -   The purpose of this module is to retrieve graphs fron Neo4j and display them in a web 
* application
"""

from _plotly_utils.utils import PlotlyJSONEncoder
from flask.globals import request
from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle
import json
import numpy as np
from models import *


data_url = 'Data Generation/generated_datasets/'

colors = [[0,  '#FFD43B'],  # discrete colorscale to map boolean
        [0.5,  '#FFD43B'],
        [0.5, '#4B8BBE '],
        [1, '#4B8BBE ']]


# Configure api
app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='template')
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False

   
def get_context(context):
    df = pd.read_csv(data_url + 'vitals.csv')
    data = list(df[context[0]].values)
    time = list(df['date'].values)
    return time, data


def get_onload_data():
    # user information
    user_info = get_user_information()
    user = [record for record in user_info]
    return user[0]


def get_activity():
    steps = []
    calories = []
    df_activtiy = pd.read_csv(data_url + 'fitbit_activity_data.csv')
    for i, row in df_activtiy.iterrows():
        steps.extend([row['steps']] * 288)
        calories.extend([row['calories']] * 288)
    return steps, calories


def plotly_global_timeseries():
    df = pd.DataFrame()
    vitals_dict = get_vitals_data()
    df['date'] = vitals_dict['heart_rate']['timestamps']
    for key in vitals_dict.keys():
        df[key] = vitals_dict[key]['data']

    # get activity data
    steps, calories = get_activity()
    df['steps'] = [x/1000 for x in steps]
    df['calories'] = [x/1000 for x in calories]
    fig = px.line(df, x='date', y=['heart_rate', 'blood_pressure', 'respiration_rate', 'skin_temprature', 
                'steps', 'calories'])
    var_labels = cycle(['HR (bpm)', 'BP (mmHg)', 'RR (bpm)', 'ST (°C)', 'steps per 1000', 'calories per 1000'])
    fig.for_each_trace(lambda t:  t.update(name = next(var_labels)))
    fig = fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(title_text='User Vitals & Activity', title_x=0.5, 
                      width=850, height=400, autosize=True)
    plot_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return plot_json


def bool_to_string(A):
    #convert a binary array into an array of strings 'True' and 'False'
    S = np.empty(A.shape,  dtype=object)
    S[np.where(A)] = 'True'
    S[np.where(A==0)] = 'False'
    return S


def plot_medication():
    medicines_taken_list = []
    dates = []
    medicines, medicines_taken = get_medicines_taken()
    medicines_list = medicines[0]['n.medicines']
    for i in range(len(medicines_taken)):
        medicines_taken_list.append(medicines_taken[i]['n.medicines_taken'])
        dates.append(medicines_taken[i]['n.Graph_Type'])
    result = map(lambda record: [1 if x in record else 0 for x in medicines_list], medicines_taken_list)
    med_df = pd.DataFrame(result, columns=medicines_list)
    med_df['date'] = dates
    
    # med_df = pd.read_csv(data_url + 'medication.csv')
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


def calculate_adherence_per_medicine():
    df = pd.read_csv(data_url + 'medication.csv')
    A = df.columns.difference(['date'])
    adherence_rates = []
    for column in A:
        adherence_rates.append(round((df[column] == 1).sum()/len(df[column]), 2))
    adherence_percentages = [str(x*100) + ' %' for x in adherence_rates]
    return adherence_percentages


def get_medication():
    medication_plan = get_medical_profile_data()
    result = [record for record in medication_plan]
    medicines = result[0]['n.medicines']
    dosages = []
    for i, j in zip(result[0]['n.dosages'], result[0]['n.doses_per_day']):
        dosages.append(str(i) + ' mg ' + str(j) + ' times per day')
    adherence_rates = calculate_adherence_per_medicine()
    return medicines, dosages, adherence_rates


def create_medication_table():
    medicines, dosages, adherence_rates = get_medication()
    fig = go.Figure(data=[go.Table(columnorder = [1,2, 3], columnwidth = [125, 120, 80],
                    header=dict(values=['Medication', 'Dose', 'Adherence'],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='center',
                    height = 33),
                    cells=dict(values=[medicines,
                               dosages, adherence_rates],
                    line_color='darkslategray',
                    fill_color='lightcyan',
                    align='center',
                    height = 33
                    ))
                ])
    fig.update_layout(title_text='Medication List', title_x=0.5, width=450, height=400)
    med_table = json.dumps(fig, cls=PlotlyJSONEncoder)
    return med_table
    

@app.route("/", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
       if request.form['action'] == 'Submit':
           return "hello"
                                   

@app.route('/main_menu')
def main_menu():
    return render_template('main_menu.html')


@app.route('/context_model')
def context_model():
    return render_template('context_model_visualization.html')


@app.route('/data_monitoring')
def data_monitoring():
    user = get_onload_data() 
    plot_json = plotly_global_timeseries()
    med_plot_json = plot_medication()
    med_table = create_medication_table()
    return render_template('data_monitoring.html', name=user["name"], age=user["age"],
                                   gender=user["gender"].capitalize(), weight='85 kg', birthdate=user["birthdate"], plot_json=plot_json,
                                   med_plot_json=med_plot_json, med_table=med_table)


if __name__ == '__main__':
    app.run()
    