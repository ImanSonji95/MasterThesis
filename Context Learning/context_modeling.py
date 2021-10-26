"""
*
* Created by    -   Iman Sonji (MT 3217)
* Creation date -   07 July 2021
* Description   -   The purpose of this module is to populate the context model in Neo4j with the newly
* generated data. This models the whole raw data.
"""


from datetime import datetime
import json
import time
from neo4j import GraphDatabase
import re
import numpy as np
import pandas as pd
from itertools import compress
from db_communication import *


user = 'neo4j'
# local connection
uri = "neo4j://localhost:7687"
password = "middleware"
# remote connection neo4j Aura
uri_aura = 'neo4j+s://9355a9d4.databases.neo4j.io'
password_aura = 'pdtjOyCoIHd2UyynGkzbQHozE46jmClHCTGrQsNKxT0'


data_url = 'Data Generation/generated_datasets/'
# sampling rates of data
SAMPLES_NUM_vitals = 288  # vitals sampled every 5 minutes
SAMPLES_NUM_env = 24  # environmental data sampled every one hour


class App:
    def __init__(self, uri, user, key):
        self.driver = GraphDatabase.driver(uri, auth=(user, key))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def create_vital_node(self, name, graph_type, time, value, unit):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_vital_node, name, graph_type, time, value, unit)

    @staticmethod
    def _create_vital_node(tx, name, graph_type, time, value, unit):
        relation = '_' + re.sub('-', '_', graph_type)
        query = (
            "CREATE (n:Context:data_source:vitals_parameter:end_parameter {name: $name ,Graph_Type: $graph_type, timestamps: $time, values: $value, unit: $unit}) "
            "WITH n "
            "Match (m:Context {name: 'vitals'}) "
            "CREATE (n)-[:" + relation + "]->(m) "
        )
        tx.run(query, name=name, graph_type=graph_type, time=time, value=value, unit=unit)

    
    def create_medicines(self, name, graph_type, medicines, dosages, doses_per_day):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_medicines, name, graph_type, medicines, dosages, doses_per_day)

    
    @staticmethod
    def _create_medicines(tx, name, graph_type, medicines, dosages, doses_per_day):
        relation = '_' + re.sub('-', '_', graph_type)
        query = (
            "CREATE (n:Context:data_source:medical_plan_parameter:medical_profile_parameter:end_parameter \
            {name: $name ,Graph_Type: $graph_type, medicines: $medicines, dosages: $dosages, doses_per_day: $doses_per_day}) "
            "WITH n "
            "Match (m:Context {name: 'medication_plan'}) "
            "CREATE (n)-[:" + relation + "]->(m) "
        )
        tx.run(query, name=name, graph_type=graph_type, medicines=medicines, dosages=dosages, doses_per_day=doses_per_day)


    def create_env_node(self, name, graph_type, time, value, main_node):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_env_node, name, graph_type, time, value, main_node)


    @staticmethod
    def _create_env_node(tx, name, graph_type, time, value, main_node):
        relation = '_' + re.sub('-', '_', graph_type)
        query = (
            "CREATE (n:Context:environment_parameter:end_parameter {name: $name ,Graph_Type: $graph_type, timestamps: $time, values: $value}) "
            "WITH n "
            "Match (m:Context {name: $main_node}) "
            "CREATE (n)-[:" + relation + "]->(m) "
        )
        tx.run(query, name=name, graph_type=graph_type,
               time=time, value=value, main_node=main_node)

    
    def create_medication_intake_node(self, name, graph_type, medicines_taken, triggered_times, opened_times):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_medication_intake_node, name, graph_type, medicines_taken, triggered_times, opened_times)


    @staticmethod
    def _create_medication_intake_node(tx, name, graph_type, medicines_taken, triggered_times, opened_times):
        relation = '_' + re.sub('-', '_', graph_type)
        query = (
            "CREATE (n:Context:medicine_intake_behavior_parameter:end_parameter {name: $name ,Graph_Type: $graph_type, medicines_taken: $medicines_taken, triggered_times: $triggered_times, opened_times: $opened_times }) "
            "WITH n "
            "Match (m:Context {name: 'medicine_intake_behavior'}) "
            "CREATE (n)-[:" + relation + "]->(m) "
        )
        tx.run(query, name=name, graph_type=graph_type, medicines_taken=medicines_taken, 
               triggered_times=triggered_times, opened_times=opened_times)

        
    def create_activity_node(self, name, graph_type, value, main_node):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_activity_node, name, graph_type, value, main_node)


    @staticmethod
    def _create_activity_node(tx, name, graph_type, value, main_node):
        relation = '_' + re.sub('-', '_', graph_type)
        query = (
            "CREATE (n:Context:activities_parameter:end_parameter {name: $name ,Graph_Type: $graph_type, value: $value}) "
            "WITH n "
            "Match (m:Context {name: $main_node}) "
            "CREATE (n)-[:" + relation + "]->(m) "
        )
        tx.run(query, name=name, graph_type=graph_type,
               value=value, main_node=main_node)

    
    def create_daily_activity(self, name, graph_type, activity, activity_start_time):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_daily_activity, name, graph_type, activity, activity_start_time)


    @staticmethod
    def _create_daily_activity(tx, name, graph_type, activity, activity_start_time):
        relation = '_' + re.sub('-', '_', graph_type)
        query = (
            "CREATE (n:Context:activities_parameter:end_parameter {name: $name ,Graph_Type: $graph_type, activity: $activity, start_time: $activity_start_time}) "
            "WITH n "
            "Match (m:Context {name: 'activities'}) "
            "CREATE (n)-[:" + relation + "]->(m) "
        )
        tx.run(query, name=name, graph_type=graph_type,
               activity=activity, activity_start_time=activity_start_time)


def model_vitals(app_instance):
    df = pd.read_csv(data_url + 'vitals.csv')
    vitals_list = list(df.columns.difference(['date']))
    units = {'heart_rate': 'bpm', 'blood_pressure': 'mmHg', 'respiration_rate': 'bpm', 'skin_temprature': '°C'}
    # write vitals for each day
    for i in range(0, int(len(df) / SAMPLES_NUM_vitals)):
        daily_vitals = df[i * SAMPLES_NUM_vitals:(i + 1) * SAMPLES_NUM_vitals]
        # get relationship and graph type
        current_date = datetime.strptime(daily_vitals['date'][i * SAMPLES_NUM_vitals], "%Y-%m-%d %H:%M:%S").strftime(
            "%Y-%m-%d")
        time = daily_vitals['date'].tolist()
        for vital in vitals_list:
            value = daily_vitals[vital].tolist()
            unit = units[vital]
            # write to graph database
            app_instance.create_vital_node(vital, current_date, time, value, unit)


def model_medical_profile(app_instance):
    # read medication_plan json file
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
    # the node is only added for the first day
    app_instance.create_medicines('medicines', '2021-07-12', medicines, dosages, doses_per_day)


def model_environmental_data(app_instance):
    df = pd.read_csv(data_url + 'environmental.csv')
    env_list = list(df.columns.difference(['date']))
    # write environment for each day
    for i in range(0, int(len(df)/SAMPLES_NUM_env)):
        daily_env = df[i * SAMPLES_NUM_env:(i + 1) * SAMPLES_NUM_env]
        # get relationship and graph type
        current_date = datetime.strptime(daily_env['date'][i * SAMPLES_NUM_env], "%Y-%m-%d %H:%M:%S").strftime(
            "%Y-%m-%d")
        for item in env_list:
            time = daily_env['date'].tolist()
            value = daily_env[item].tolist()
            if item in ('indoor_temperature', 'indoor_humidity'):
                app_instance.create_env_node(item, current_date, time, value, 'indoor_environment')
            else:
                app_instance.create_env_node(item, current_date, time, value, 'outdoor_environment')


def model_medication_intake(app_instance):
    df_medication = pd.read_csv(data_url + 'medication.csv')
    df_medication_times = pd.read_csv(data_url + 'medication_times.csv')
    medicines = list(df_medication.columns.difference(['date']))
    df_medication_values = df_medication.loc[:, df_medication.columns != 'date']
    # write medication intake behavoir for each day
    for i in range(0, int(len(df_medication))):
        current_date = datetime.strptime(
            df_medication.iloc[i]['date'], "%Y-%m-%d").strftime("%Y-%m-%d")
        medicines_taken = list(compress(medicines, df_medication_values.iloc[i].tolist()))
        triggered_times = df_medication_times['triggered_times'].iloc[i*4:i*4+4]
        opened_times = df_medication_times['opened_times'].iloc[i*4:i*4+4]
        app_instance.create_medication_intake_node('intake_time', current_date, medicines_taken, 
                                                    triggered_times.tolist(), opened_times.tolist())


def model_activity_data(app_instance):
    df = pd.read_csv(data_url + 'fitbit_activity_data.csv')
    # write activity for each day
    for i in range(0, int(len(df))):
        daily_activity = df.iloc[i]
        current_date = datetime.strptime(
            daily_activity['date'], "%d/%m/%Y").strftime("%Y-%m-%d")
        for item in daily_activity.index.T[1:].tolist():
            value = daily_activity[item]
            if isinstance(value, np.integer):
                value = int(daily_activity[item])
            if item in ('minutesVeryActive', 'minutesSedentary', 'minutesLightlyActive', 'minutesFairlyActive'):
                app_instance.create_activity_node(item, current_date, value, 'minutes_active')
            elif item in ('minutesAsleep', 'minutesAwake', 'time_in_bed', 'efficiency', 'sleep_start_time'):
                app_instance.create_activity_node(item, current_date, value, 'sleep')
            elif item in ('calories', 'steps', 'distance'):
                app_instance.create_activity_node(item, current_date, value, 'activities')


def model_daily_activity(app_instance):
    df = pd.read_csv(data_url + 'fitbit_activity_data.csv')
    df_daily_act = df[['date', 'daily_activity', 'activity_start_time']]
    for i, row in df_daily_act.iterrows():
        if row['daily_activity'] != 'resting':
            current_date = datetime.strptime(
                           row['date'], "%d/%m/%Y").strftime("%Y-%m-%d")
            daily_activity = row['daily_activity']
            act_start_time = row['activity_start_time']
            app_instance.create_daily_activity('daily_activity', current_date, daily_activity, act_start_time)


if __name__ == "__main__":
    app = App(uri, user, password)
    start_time = time.time()
    # Model data
    model_vitals(app)
    model_medical_profile(app)
    model_environmental_data(app)
    model_medication_intake(app)
    model_activity_data(app)
    model_daily_activity(app)
    print("Context is Modeled --- %s seconds ---" % (time.time() - start_time))
    # Close connection
    app.close()
    