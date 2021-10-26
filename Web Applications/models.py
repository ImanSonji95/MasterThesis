"""
*
* Created by    -   Iman Sonji (MT 3217)
* Creation date -   07 July 2021
* Description   -   The purpose of this module is to populate the context model in Neo4j with the newly
* generated data.
"""

from datetime import time
from itertools import chain
from py2neo import Graph

# local connection
uri = "neo4j://localhost:7687"
username = "neo4j"
password = "middleware"

# initialize neo4j conenction
graph = Graph(uri=uri, user=username, password=password)

# graph_context_states = Graph(uri=uri_context_states, user=username, password=password)


def get_user_information():
    query = '''
    MATCH (n:user:Init_Graph) RETURN n.user_name as name, n.gender as gender, 
    n.birthdate as birthdate, n.age as age
    '''
    return graph.run(query)


def get_medical_profile_data():
    query = '''
    MATCH (n:Context:medical_profile_parameter {name: "medicines", Graph_Type: "2021-07-12"})
    RETURN n.medicines, n.dosages, n.doses_per_day
    '''
    return graph.run(query)


def get_medicines_taken():
    # intake_times
    medicines_query = '''
    MATCH (n:Context:medical_profile_parameter {name: "medicines", Graph_Type: "2021-07-12"})
    RETURN n.medicines
    '''
    medicines_result = graph.run(medicines_query)
    medicines = [record for record in medicines_result]

    medicines_taken_query = '''
    MATCH (n {name: "intake_time"}) RETURN n.Graph_Type, n.medicines_taken
    '''
    medicines_taken_result = graph.run(medicines_taken_query)
    medicines_taken = [record for record in medicines_taken_result]
    
    return medicines, medicines_taken


def get_vitals_data():
    vitals_dict = {}
    # vitals_nodes_query = '''
    # MATCH (n:Context:vitals_parameter) RETURN n.name
    # '''
    # vitals_nodes_result = graph.run(vitals_nodes_query).data()
    # vitals_nodes = [list(set([record['n.name'] for record in vitals_nodes_result]))]

    vitals_nodes = ['skin_temprature', 'blood_pressure', 'heart_rate', 'respiration_rate']

    for vital in vitals_nodes:
        vitals_data_query = (
            "MATCH (n:Context {name: $vital}) RETURN n.timestamps, n.values"
        )
        vitals_data_result = graph.run(vitals_data_query, vital=vital).data()
        timestamps = [record['n.timestamps'] for record in vitals_data_result]
        timestamps = list(filter(None, timestamps))
        timestamps = list(chain.from_iterable(timestamps))
        data = [record['n.values'] for record in vitals_data_result]
        data = list(filter(None, data))
        data = list(chain.from_iterable(data))
        vitals_dict[vital] = {}
        vitals_dict[vital]['timestamps'] = timestamps
        vitals_dict[vital]['data'] = data

    return vitals_dict


def get_activity_data():
    activity_dict = {}

    activity_nodes = ['steps', 'calories']
    for activity in activity_nodes:
        activity_data_query = (
            "MATCH (n:Context {name: $activity}) RETURN n.value"
        )
        activity_data_result = graph.run(activity_data_query, activity=activity).data()
        data = [record['n.value'] for record in activity_data_result]
        data = list(filter(None, data))
        activity_dict[activity] = {}
        activity_dict[activity]['data'] = data

    return activity_dict