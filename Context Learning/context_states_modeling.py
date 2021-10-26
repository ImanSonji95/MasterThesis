"""
*
* Created by    -   Iman Sonji (MT 3217)
* Creation date -   01 September 2021
* Description   -   The purpose of this module is to populate the context model in Neo4j with the context states.
* This modeles the preprocessed data.
"""

import re
import pandas as pd
from db_communication import *


# similar to meta model
def create_static_nodes():
    query = (
        "CREATE (n:user {name: 'user'}) "
        "CREATE (m:user_parameter {name: 'vitals'}) "
        "CREATE (p:user_parameter {name: 'environment'}) "
        "CREATE (s:user_parameter {name: 'medication_behavoir'}) "
        "CREATE (q:user_parameter:end_parameter {name: 'location'}) "
        "CREATE (r:user_parameter:end_parameter {name: 'activity'}) "
        "CREATE (t:vitals_parameter:end_parameter {name: 'heart_rate'}) "
        "CREATE (u:vitals_parameter:end_parameter {name: 'blood_pressure'}) "
        "CREATE (v:vitals_parameter:end_parameter {name: 'respiration_rate'}) "
        "CREATE (w:vitals_parameter:end_parameter {name: 'skin_temprature'}) "
        "CREATE (x:environmental_parameter:end_parameter  {name: 'temperature'}) "
        "CREATE (y:environmental_parameter:end_parameter  {name: 'humidity'}) "
        "CREATE (z:medication_parameter:end_parameter {name: 'medication_rate'}) "
        "CREATE (a:medication_parameter:end_parameter {name: 'medicines'})"
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
    graph.run(query_user)
    graph.run(query_vitals)
    graph.run(query_env)
    graph.run(query_med)

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
                    "CREATE (n:Context {name: $value}) "
                    "WITH n "
                    "MATCH (m:end_parameter {name: $node}) "
                    "CREATE (n)-[:" + f"{relation}" + "]->(m)"
                )
                graph.run(query, value=value, node=node)


def model_data():
    context_states = pd.read_csv('Context Learning/context_states.csv')
    create_static_nodes()
    create_static_relationships()
    create_model(context_states)


if __name__ == '__main__':
    model_data()
    
    