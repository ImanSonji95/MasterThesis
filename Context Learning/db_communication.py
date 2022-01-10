"""
*
* Created by    -   Iman Sonji (MT 3217)
* Creation date -   30 Aug 2021
* Description   -   The purpose of this module is to communicate with neo4j graph database.
"""

from py2neo import Graph

# local connection
uri = "neo4j://localhost:7687"
username = "neo4j"
password = "middleware"

# initialize neo4j conenction
graph = Graph(uri=uri, auth=(username, password))
# graph_2 = Graph(uri=uri, auth=(username, password), name='middleware')


def get_meta_nodes():
    query = """
    MATCH (n:META) RETURN n.name
    """
    return graph.run(query)


def create_node(context_1, context_2, weight):
    query = (
        "MERGE (n:Correlation {name: $context_1}) "
        "MERGE (m:Correlation {name: $context_2}) "
        "CREATE (n)-[r:RELATED_TO {weight: $weight}]->(m)"       
    )

    graph.run(query, context_1=context_1, context_2=context_2, weight=weight)
    # graph_2.run(query, context_1=context_1, context_2=context_2, weight=weight)
    

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


def write_threshold_meta_model(activity, vital, min_max_dict):
    query = (
        "MATCH (n:vitals_parameter {name: $vital}) "
        "WITH n " 
        "SET n." + str(activity) + "= $min_max_dict"
    )

    graph.run(query, vital=vital, min_max_dict=min_max_dict)


# get threshold for specific vital and activity
def get_threshold(activity, vital):
    query = (
        "MATCH (n:vitals_parameter {name: $vital}) "
        "RETURN n." + str(activity)
    )
    
    return graph.run(query, vital=vital)


def get_timestamps_context_nodes(main_node_name, context_node_name):
    query = (
        "MATCH (n:Context {name: $context_node_name})-[r]-(m:end_parameter {name: $main_node_name}) "
        "RETURN type(r)"
    )

    return graph.run(query, main_node_name=main_node_name, context_node_name=context_node_name)


def get_timestamps_anomalous_vitals(main_node_name, context_node_name):
    query = (
        "MATCH (n:vitals_parameter {name: $main_node_name})-[r]-(m:Context) "
        "WHERE m.name <> $context_node_name "
        "RETURN type(r), m.name"
    )

    return graph.run(query, main_node_name=main_node_name, context_node_name=context_node_name)


def set_vital_anomalous(node_name, timestamp):
    query = (
        "MATCH (n {name: $node_name})-[r:" + str(timestamp) + "]-(m) "
        "WITH n "
        "SET n:anomalous"
    )

    graph.run(query, node_name=node_name)


def get_correlations():
    query = (
        "MATCH (n:Correlation)-[r]->(m:Correlation) "
        "RETURN n.name, m.name, r.weight"
    )
    return graph.run(query)


def get_timed_relations(category, value):
    query = (
        "MATCH (n {name: $category})-[r]-(m:Context {name: $value}) RETURN type(r)"
    )
    return graph.run(query, category=category, value=value)


def create_relationship_context_model(source_category, source_value, destiny_category, destiny_value, relation_weight, ts):
    query = (
        "MATCH (a {name: $source_category})-[r:" + ts + "]-(b:Context {name: $source_value}) "
        "MATCH (c {name: $destiny_category})-[s:" + ts + "]-(d:Context {name: $destiny_value}) "
        "WITH b, d "
        "CREATE (b)-[t:RELATED_TO {weight: $relation_weight}]->(d)"
    )

    graph.run(query, source_category=source_category, source_value=source_value, destiny_category=destiny_category,
            destiny_value=destiny_value, relation_weight=relation_weight, ts=ts)

