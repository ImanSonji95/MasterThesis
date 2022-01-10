"""
*
* Created by    -   Iman Sonji (MT 3217)
* Creation date -   07 July 2021
* Description   -   The purpose of this module is to populate the context model in Neo4j with the newly
* generated data.
"""

from itertools import chain
import re
from py2neo import Graph

# local connection
uri = "neo4j://localhost:7687"
username = "neo4j"
password = "middleware"

# initialize neo4j conenction
graph = Graph(uri=uri, auth=(username, password))
# graph_2 = Graph(uri=uri, auth=(username, password), name='middleware')

# graph_context_states = Graph(uri=uri_context_states, user=username, password=password)


def get_user_information():
    query = '''
    MATCH (n:user:static) RETURN n.username as name, n.age as age,
    n.gender as gender, n.weight as weight
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

    vital_records = get_vitals_nodes()
    vital_nodes = [record['n.name'] for record in vital_records]

    for vital in vital_nodes:
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


def get_vitals_nodes():
    query = (
        "MATCH (n:end_parameter:vitals_parameter) RETURN n.name"
    )
    return graph.run(query)


def get_activities_nodes():
    query = (
        "MATCH (n:user_parameter {name: 'activity'})-[r]-(m:Context) RETURN m.name"
    )
    return graph.run(query)


def find_vital_threshold(activity_node, vital_node):
    query = (
        "MATCH (n:Correlation {name: $activity_node})-[r:RELATED_TO]-(m:Correlation) "
        "WHERE m.name STARTS WITH $vital_node "
        "RETURN m.name ORDER BY r.weight DESC"
    )
    return graph.run(query, activity_node=activity_node, vital_node=vital_node)


def find_anomalous_heart_rate():
    high_value = 0
    query = (
        "MATCH (n:Context)-[r]-(m {name: 'heart_rate'}) "
        "RETURN n.name ORDER BY n.name ASC"
    )
    values = graph.run(query)
    values = list(set([record['n.name'] for record in values]))
    for value in values:
        match = re.search("(?<=\[).+?(?=\])", value)
        if match:
            range = match.group()
            min_max_values = list(map(float, list(range.split(","))))
            if (min_max_values[0] > high_value):
                high_value = min_max_values[0]
                high_value_list = str(min_max_values)

    query = (
        "MATCH (n {name: $high_value_list})-[r]-(m {name: 'heart_rate'}) "
        "RETURN type(r)"
    )
    timestamps = graph.run(query, high_value_list=high_value_list).data()
    timestamps = [record["type(r)"] for record in timestamps]
    return high_value_list, timestamps


def get_cooccurring_nodes(node, timestamp):
    cooccurring_nodes = []
    query = (
        "MATCH (n {name: $node})-[r:" + timestamp + "]-(m)-[a:RELATED_TO]-(b)-[s:" + timestamp + "]-(d)"
        "RETURN b.name, d.name"
    )
    
    result = graph.run(query, node=node).data()
    entities = [record["d.name"] for record in result]
    context = [record["b.name"] for record in result]
    for x, y in zip(entities, context):
        cooccurring_nodes.append(str(x) + ':' + ' ' + str(y))

    return cooccurring_nodes


def get_correlations_heart_rate(high_value_list):
    query = (
        "MATCH (n:Correlation {name: 'heart_rate: " + high_value_list + "'})-[r]-(m:Correlation) "
        "WHERE m.name <> 'activity: resting' and m.name <> 'location: indoors'"
        "RETURN m.name, r.weight"
    )
    result = graph.run(query).data()
    correlations = [record["m.name"] for record in result]
    weights = [record["r.weight"] for record in result]

    return correlations, weights


def get_medication_nonadherence():
    query = (
        "MATCH (n:Correlation)-[r]-(m:Correlation) "
        "WHERE n.name= 'medication_rate: 1.0' "
        "RETURN m.name"
    )
    result = graph.run(query).data()
    correlations = [record["m.name"] for record in result]
    print(correlations)