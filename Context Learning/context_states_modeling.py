"""
*
* Created by    -   Iman Sonji (MT 3217)
* Creation date -   01 September 2021
* Description   -   The purpose of this module is to populate the context model in Neo4j with the context states.
* This modeles the preprocessed data.
"""

import json
import re
import pandas as pd
from db_communication import *


# similar to meta model
def create_static_nodes():
    query = (
        "CREATE (n:user:static:user_content_view:user_parameter {name: 'user'}) "
        "CREATE (m:user_parameter:static:user_content_view:user_parameter {name: 'vitals'}) "
        "CREATE (p:user_parameter:static:user_content_view:user_parameter {name: 'environment'}) "
        "CREATE (s:user_parameter:static:user_content_view:user_parameter {name: 'medication_behavoir'}) "
        "CREATE (q:user_parameter:static:user_content_view:user_parameter:end_parameter {name: 'location'}) "
        "CREATE (r:user_parameter:static:user_content_view:user_parameter:end_parameter {name: 'activity'}) "
        "CREATE (t:vitals_parameter:static:user_content_view:end_parameter {name: 'heart_rate'}) "
        "CREATE (u:vitals_parameter:static:user_content_view:end_parameter {name: 'blood_pressure'}) "
        "CREATE (v:vitals_parameter:static:user_content_view:end_parameter {name: 'respiration_rate'}) "
        "CREATE (w:vitals_parameter:static:user_content_view:end_parameter {name: 'skin_temprature'}) "
        "CREATE (x:environmental_parameter:static:user_content_view:end_parameter  {name: 'temperature'}) "
        "CREATE (y:environmental_parameter:static:user_content_view:end_parameter  {name: 'humidity'}) "
        "CREATE (z:medication_parameter:end_parameter:static:user_content_view {name: 'medication_rate'}) "
        "CREATE (a:medication_parameter:end_parameter:static:user_content_view {name: 'medicines'})"

        "CREATE (b:Overview {name: 'System'})"
        "CREATE (c:Overview:system_parameter {name: 'Fitbit'})"
        "CREATE (d:Overview:system_parameter {name: 'Sensory_Network'})"
        "CREATE (e:Overview:system_parameter {name: 'Pill_Dispenser'})"
        "CREATE (f:Overview:system_parameter {name: 'Human_Body_Simulator'})"
        "CREATE (g:Overview:pill_dispenser_parameter {name: 'Stationary_Pill_Dispenser'})"
        "CREATE (h:Overview:pill_dispenser_parameter {name: 'Mobile_Pill_Dispenser'})"
        
    )
    graph.run(query)


def create_static_relationships():
    query_user = (
        "MATCH (n:user), (m:user_parameter) "
        "CREATE (n)-[r:HAS_A]->(m) "
    )

    query_vitals = (
        "MATCH (n:user_parameter {name: 'vitals'}), (m:vitals_parameter) "
        "CREATE (n)<-[r:BELONGS_TO]-(m) "
    )

    query_env = (
        "MATCH (n:user_parameter {name: 'environment'}), (m:environmental_parameter) "
        "CREATE (n)<-[r:BELONGS_TO]-(m) "
    )

    query_med = (
        "MATCH (n:user_parameter {name: 'medication_behavoir'}), (m:medication_parameter) "
        "CREATE (n)<-[r:BELONGS_TO]-(m) "
    )

    query_sys = (
        "MATCH (n:Overview {name: 'System'}), (m:system_parameter) "
        "CREATE (n)-[r:HAS_A]->(m)"
    )

    query_pill_dispenser = (
        "MATCH (n:Overview {name: 'Pill_Dispenser'}), (m:pill_dispenser_parameter) "
        "CREATE (n)-[r:HAS_A]->(m)"
    )
    graph.run(query_user)
    graph.run(query_vitals)
    graph.run(query_env)
    graph.run(query_med)
    graph.run(query_sys)
    graph.run(query_pill_dispenser)
    

# read data from the context_states.csv and create context nodes. 
def create_model(context_states):
    days = context_states['day']
    times = context_states['time']
    df = context_states[context_states.columns.difference(['day', 'time'])]
    nodes_names = df.columns.values.tolist()
    for node in nodes_names:
        data = df[node].tolist()
        for value, day, time in zip(data, days, times):
            if pd.notnull(value):
                relation = '_' + re.sub('-', '_', day) + 'T' + re.match('.+?(?=:)', time).group(0)
                query = (
                    "CREATE (n:Context:Dynamic:end_parameter:" + str(node) + "{name: $value}) "
                    "WITH n "
                    "MATCH (m:static:user_content_view {name: $node}) "
                    "CREATE (n)-[:" + f"{relation}" + "]->(m)"
                )
                graph.run(query, value=str(value), node=node)


# Add user static context
def add_user_info_context():
    query = (
        "MATCH (n:user:static) "
        "SET n.username = 'Mark', n.age = 70,  n.weight = '85 kg' "        
    )
    graph.run(query)


def add_user_medical_context():
    # read medication_plan json file
    f = open('medication_plan.json')
    medicaiton_plan = json.load(f)
    medicines = []
    dosages = []
    doses_per_day = []
    data = medicaiton_plan['medication_plan']['medicines']
    for i in range(1, len(data)):
        medicines.append(data['medicine_' + str(i)]['active_ingredient_str'])
        dosages.append(data['medicine_' + str(i)]['dosage'])
        doses_per_day.append(data['medicine_' + str(i)]['dose_per_day'])
    return ""


def model_data():
    context_states = pd.read_csv('Context Learning/context_states.csv')
    create_static_nodes()
    create_static_relationships()
    create_model(context_states)
    add_user_info_context()
 


if __name__ == '__main__':
    model_data()
    