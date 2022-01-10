import itertools
from tslearn.clustering import TimeSeriesKMeans
from datetime import datetime, timedelta
import pandas as pd
from itertools import groupby
from random import randint
from sklearn.mixture import GaussianMixture
import math
from db_communication import *
import re


"""
Functions for finding association rules.
"""

def get_itemset_list():
    tempItemSet = {}
    main_nodes_query = 'MATCH (n:end_parameter) RETURN n.name'
    main_nodes_result = graph.run(main_nodes_query).data()
    for item in main_nodes_result:
        node = item['n.name']
        tempItemSet[node] = []
        value_nodes_query = (
            'MATCH (n:end_parameter {name: $node})-[r]-(m:Context) RETURN m.name'
        )
        value_nodes_result = graph.run(value_nodes_query, node=node).data()
        value_nodes_list = [item['m.name'] for item in value_nodes_result]
        tempItemSet[node] = set(value_nodes_list)
    # df = context_states[context_states.columns.difference(['days'])]
    # for column in df:
    #     tempItemSet[column] = list(set(df[column]))
    return tempItemSet


def get_freq_one_itemset(oneItemSet, min_support):
    freqOneItemSet = {}
    keyTotalCount = {}

    for key, itemSet in oneItemSet.items():
        total_count_query = (
           'MATCH (n:end_parameter {name: $key})-[r]-(m:Context) RETURN count(m)' 
        )
        total_count = graph.run(total_count_query, key=key).data()
        total_count = total_count[0]['count(m)']
        keyTotalCount[key] = total_count
        freqOneItemSet[key] = []
        for item in itemSet:
            item_count_query = (
                'MATCH (n:Context {name: $item}) RETURN count(n)'
            )
            occ_count = graph.run(item_count_query, item=item).data()
            occ_count = occ_count[0]['count(n)']
            # occ_count = len(df[df[key] == item])
            support = round(float(occ_count/ total_count), 4)
            freqOneItemSet[key].append((item, support))

    for key, itemSet in freqOneItemSet.items():
        freqOneItemSet[key] = list(filter(lambda x: x[1] > min_support, itemSet))

    return freqOneItemSet, keyTotalCount


def prune_combs(item):
    key_1 = next(iter(item[0].keys()))
    key_2 = next(iter(item[1].keys()))
    # remove combinations that have no correlation between items
    list_1 = ['activity', 'humidity', 'temperature', 'location']
    list_2 = ['medication_rate', 'medicines']
    if key_1 == key_2:
        return False
    elif (key_1 in list_1 and key_2 in list_2) or (key_1 in list_2 and key_2 in list_1):
        return False
    else:
        return True


def get_union(freqOneItemSet, k):
    items = []
    for key, itemSet in freqOneItemSet.items():
        for item in itemSet:
            items.append({key: item[0]})
    # combinations replaced by permuations
    combs = list(itertools.permutations(items, k))
    reduced_combs = list(filter(prune_combs, combs))  # reduced by 648 combination
    return reduced_combs

        
def get_freq_itemset(combs, keyTotalCount, minSup):
    freq_itemsets = []
    support_count = []
    for item in combs:
        # occu_count = 0
        current_keys = [next(iter(item[x].keys())) for x in range(2)]
        current_itemSet = [next(iter(item[x].values())) for x in range(2)]
        # find the occurent count of each combination from neo4j graph
        # TODO
        # replace by get_timed_relations()
        timestamps_first_item_query = (
            'MATCH (n:Context {name: $current_itemSet})-[r]-(m:end_parameter {name: $current_keys}) RETURN type(r)'
        )
        timestamps_second_item_query = (
            'MATCH (n:Context {name: $current_itemSet})-[r]-(m:end_parameter {name: $current_keys}) RETURN type(r)'
        )
        timestamps_first_item = graph.run(timestamps_first_item_query, 
                               current_itemSet=current_itemSet[0], current_keys=current_keys[0]).data()
        timestamps_second_item = graph.run(timestamps_second_item_query, 
                               current_itemSet=current_itemSet[1], current_keys=current_keys[1]).data()
        timestamps_first_item_list = [d['type(r)'] for d in timestamps_first_item]
        timestamps_second_item_list = [d['type(r)'] for d in timestamps_second_item]
        occ_count = len([element for element in timestamps_first_item_list if element in timestamps_second_item_list])
        # for index, row in context_state.iterrows():
        #     itemSet = set(row.tolist())
        #     if current_itemSet.issubset(itemSet):
        #         occu_count += 1
        # support = round(float(occu_count/len(context_state)), 4)
        total = min(keyTotalCount[current_keys[0]], keyTotalCount[current_keys[1]])
        support = round(float(occ_count/total), 4)
        if (support >= minSup):
            # add current keys and current itemset
            freq_itemsets.append(item)
            support_count.append(support)
    return freq_itemsets, support_count
    

def get_association_rules(freq_itemsets, itemSet_support, freqOneItemSets, min_confidence):
    rules = []
    oneItemSets_list = []
    for item in freqOneItemSets.values():
        oneItemSets_list.extend(item)

    for itemSet, itemSupport in zip(freq_itemsets, itemSet_support):
        current_itemSet = [next(iter(itemSet[x].values())) for x in range(2)]
        item_sup = itemSupport
        x_sup = next((v[1] for v in oneItemSets_list if v[0] == current_itemSet[0]), None)
        # y_sup = next((v[1] for v in oneItemSets_list if v[0] == current_itemSet[1]), None)
        confidence = float(item_sup/x_sup)
        if (confidence > min_confidence and confidence <= 1):
            rules.append([itemSet, confidence])
            
    return rules


def create_df_rules(pruned_rules):
    df_rules = pd.DataFrame(columns=['context_1', 'context_2', 'weight'])
    context_1 = []
    context_2 = []
    weights = []
    for rule in pruned_rules:
        context_1.append(rule[0][0])
        context_2.append(rule[0][1])
        weights.append(rule[1])
    df_rules['context_1'] = context_1
    df_rules['context_2'] = context_2
    df_rules['weight'] = [round(x, 2) for x in weights]
    return df_rules


def remove_duplicated_rules(df_rules):
    items_dict = {}
    for index, row in df_rules.iterrows():
        # form key
        item_1 = str(next(iter(row['context_1'].items())))
        item_2 = str(next(iter(row['context_2'].items())))
        key_list = sorted([item_1, item_2])
        key = key_list[0] + ' ' + key_list[1]
        if key not in items_dict:
            items_dict[key] = row.tolist()
        else:
            # compare weights
            if row['weight'] > items_dict[key][2]:
                items_dict[key] = row.tolist()
    
    dict_values = list(items_dict.values())
    final_df = pd.DataFrame(dict_values)
    final_df.columns = ['context_1', 'context_2', 'weight']
    return final_df, dict_values
    

def prune_rules(reduced_rules):
    groups = []
    pruned_rules = []
    # sorted_rules = sorted(reduced_rules, key=lambda x:(next(iter(x[0].keys())), next(iter(x[1].keys()))))

    for k, g in groupby(reduced_rules, key=lambda x:(next(iter(x[0].keys())), next(iter(x[1].keys())))):
        groups.append(list(g))

    for g in groups:
        max_weight = max([x[2] for x in g])
        rule = list(filter(lambda x: x[2] >= max_weight, g))
        pruned_rules.extend(rule)

    return pruned_rules


def build_weighted_graph(df_rules):
    def rule(row):
        key_1, value_1 = next(iter(row['context_1'].items()))
        key_2, value_2 = next(iter(row['context_2'].items()))
        create_node(f"{key_1}: {value_1}", f"{key_2}: {value_2}", row['weight'])
    # for i, row in df_rules.iterrows():
    #     create_node(row['context_1'], row['context_2'], row['weight'])
    df_rules.apply(rule, axis=1)


def update_meta_model(df_rules):
    # update thresholds in vitals
    # find nodes 
    vital_records = get_vitals_nodes()
    vital_nodes = [record['n.name'] for record in vital_records]
    acitivity_records = get_activities_nodes()
    activity_nodes = list(set([record['m.name'] for record in acitivity_records]))

    # get thresholds
    for vital in vital_nodes:
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
                    # write threshold to meta model
                    write_threshold_meta_model(activity, vital, min_max_values)


# This function to create weighted relationships in the context model based on the learned correlations
def reflect_learning_in_context_model():
    # get correlations from weighted graph
    correlations = get_correlations().data()
    source_nodes = [result['n.name'] for result in correlations]
    destiny_nodes = [result['m.name'] for result in correlations]
    weights = [result['r.weight'] for result in correlations]
    combined = zip(source_nodes, destiny_nodes, weights)
    # for each correlation, find the nodes in the context model
    for source, destiny, weight in combined:
        relation_weight = float(weight)
        if relation_weight >= 0.4:
            source_category, source_value = source.split(':')[0], source.split(':')[1][1:]
            destiny_category, destiny_value = destiny.split(':')[0], destiny.split(':')[1][1:]
            # get common relationships (co-occuring nodes)
            source_ts = get_timed_relations(source_category, source_value).data()
            destiny_ts = get_timed_relations(destiny_category, destiny_value).data()
            source_ts_list = [d['type(r)'] for d in source_ts]
            destiny_ts_list = [d['type(r)'] for d in destiny_ts]
            common_ts = [element for element in source_ts_list if element in destiny_ts_list]
            for ts in common_ts:
                # create relationship
                create_relationship_context_model(source_category, source_value,
                                                destiny_category, destiny_value, relation_weight, ts)


"""
Functions for clustering vitals and environmental data
"""

# clustering data
def cluster_data(column, n_clusters, y):
    model = TimeSeriesKMeans(n_clusters, metric="dtw", max_iter=10)
    X = column.values.reshape(-1, 1)
    model.fit(X, y)
    # plt.scatter(y, X[:, 0], c=model.labels_, cmap='viridis')
    # plt.show()
    return model.labels_


def cluster_vitals(df):
    df_columns = df[df.columns.difference(['date'])]
    vitals_list = []
    for column in df_columns:
        model_labels = cluster_data(df_columns[column], 4, df['date'])
        df[column + '_labels'] = model_labels
        vitals_list.append(column)
    df_vitals = df
    return vitals_list, df_vitals


def cluster_env(df):
    df_columns = df[df.columns.difference(['date'])]
    environment_list = []
    for column in df_columns:
        model_labels = cluster_data(df_columns[column], 4, df['date'])
        df[column + '_labels'] = model_labels
        environment_list.append(column)
    df_env = df
    return environment_list, df_env


def gmm_clustering(column, n):
    model = GaussianMixture(n_components=n, covariance_type='spherical')
    X = column.values.reshape(-1, 1)
    labels = model.fit_predict(X)
    return labels, model.means_, model.covariances_


def gmm_cluster_vitals(df_vitals, n_clusters):
    vitals_means_std = {}
    df_columns = df_vitals[df_vitals.columns.difference(['date'])]
    vitals_list = []
    for column in df_columns:
        labels, means, covariances = gmm_clustering(df_columns[column], n=n_clusters)
        df_vitals[column + '_labels'] = labels
        vitals_means_std[column] = {}
        vitals_means_std[column]['mean'] = [item for sublist in means.tolist() for item in sublist]
        vitals_means_std[column]['std'] = list(map(lambda x: math.sqrt(x), covariances))
        vitals_list.append(column)
    return df_vitals, vitals_means_std, vitals_list


def gmm_cluster_env(df_env, n_clusters):
    env_means_std = {}
    df_columns = df_env[df_env.columns.difference(['date'])]
    env_list = []
    for column in df_columns:
        labels, means, covariances = gmm_clustering(df_columns[column], n=n_clusters)
        df_env[column + '_labels'] = labels
        env_means_std[column] = {}
        env_means_std[column]['mean'] = [item for sublist in means.tolist() for item in sublist]
        env_means_std[column]['std'] = list(map(lambda x: math.sqrt(x), covariances))
        env_list.append(column)
    return df_env, env_means_std, env_list


def find_dominant_labels(clustered_vitals, clustered_env, vitals_list, env_list):
    vitals_dominant_label = {}
    env_dominant_label = {}
    for item_1, item_2 in zip(vitals_list, env_list):
        vitals_dominant_label[item_1] = []
        env_dominant_label[item_2] = []
        for i in range(0, len(clustered_vitals), 12): # 12 is number of vitals per hour
            current_list = list(clustered_vitals[item_1 + '_labels'][i:i + 12])
            dominant_label = max(current_list, key=current_list.count)
            vitals_dominant_label[item_1].append(dominant_label)
        for i in range(0, len(clustered_env), 1): # 1 is number of env per hour 
            current_list = list(clustered_env[item_2 + '_labels'][i:i + 1])
            dominant_label = max(current_list, key=current_list.count)
            env_dominant_label[item_2].append(dominant_label)
    return vitals_dominant_label, env_dominant_label
    

def write_vitals_env(vitals_labels, env_labels, clustered_df):
    merged_labels = {**vitals_labels, **env_labels}
    for key, value in merged_labels.items():
        clustered_df[key] = value
    return clustered_df  


def find_thresholds(vitals_mean_std, env_mean_std, n_clusters, vitals_list, env_list):
    vitals_thresholds = {}
    env_thresholds = {}
    for vital, env in zip(vitals_list, env_list):
        vitals_thresholds[vital] = {}
        env_thresholds[env] = {}
        for i in range(n_clusters):
            mean_vital = vitals_mean_std[vital]['mean'][i]
            std_vital = vitals_mean_std[vital]['std'][i]
            mean_env = env_mean_std[env]['mean'][i]
            std_env = env_mean_std[env]['std'][i]
            vitals_thresholds[vital][i] = [round(mean_vital - std_vital, 2), round(mean_vital + std_vital, 2)]
            env_thresholds[env][i] = [round(mean_env - std_env, 2), round(mean_env + std_env, 2)]
    return vitals_thresholds, env_thresholds


def replace_labels_with_thresholds(vitals_thresholds, env_thresholds, clustered_df):
    for key, value in vitals_thresholds.items():
        clustered_df[key] = clustered_df[key].map(value)
    for key, value in env_thresholds.items():
        clustered_df[key] = clustered_df[key].map(value)
    return clustered_df


"""
Functions for integrating context
"""

def get_activity_dates(fitbit_df):
    activity_start_times = []
    activity_end_times = []
    # activities = []
    for index, row in fitbit_df.iterrows():
        # TODO
        # can be moved to data generation instead 
        if row['location'] == 1:
            start_date = row['date'] + ' ' + row['activity_start_time']
            start_date = datetime.strptime(start_date, '%d/%m/%Y %H:%M')
            end_date =  start_date + timedelta(minutes=randint(20, 35))
            activity_start_times.append(start_date)
            activity_end_times.append(end_date)
            # activities.append(row['daily_activity'])
    return activity_start_times, activity_end_times


def add_location(start_times, end_times, context_df):
    location = [0] * len(context_df)
    count = -1
    for date in context_df['time']:
        count += 1
        date = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
        for start, end in zip(start_times, end_times):
            if start <= date <= end:
                index = count
                location[index] = 1
    context_df['location'] = location
    return context_df


def add_activities_to_context(context_df): 
    fitbit_df = pd.read_csv('Data Generation/generated_datasets/fitbit_activity_data.csv')
    fitbit_df = fitbit_df.set_index('date')
    # 8 samples in total - need to be changed based on delta t
    activities = (['sleeping'] * 8 + ['resting'] * 16) * 154  # 8 hours sleeping, 16 resting, walking or running
    for index, row in context_df.iterrows():
        if row['location'] == 1:
            date = datetime.strptime(str(row['time']), '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y')
            activity = fitbit_df['daily_activity'][date]
            activities[index] = activity
    context_df['activity'] = activities
    return context_df
    

def adjust_time_location(context_df):
    temps = []
    hums = []
    for index, row in context_df.iterrows():
        if row['location'] == 0:
            temps.append(row['indoor_temperature'])
            hums.append(row['indoor_humidity'])
        else:
            temps.append(row['outdoor_temperature'])
            hums.append(row['outdoor_humidity'])
    context_df['temperature'] = temps
    context_df['humidity'] = hums
    context_df.drop(columns=['indoor_temperature', 'indoor_humidity', 'outdoor_temperature', 'outdoor_humidity'], inplace=True)
    return context_df


def split_time_day(context_df):
    dates = context_df['time']
    days = []
    times = []
    for date in dates:
        days.append(datetime.strptime(str(date),  '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'))
        times.append(datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S').strftime('%H:%M'))
    context_df.drop(columns='time', inplace=True)
    context_df.insert(0, 'day', days)
    context_df.insert(1, 'time', times)
    return context_df


"""
Functions for data analysis
"""

def calculte_medication_adherence(med_times_df):
    medication_rates = []
    for i in range(0, len(med_times_df), 4): # 4 is number of medications per day
        triggered_times = list(map(lambda x: datetime.strptime(x ,'%H:%M'), med_times_df['triggered_times'][i: i + 4]))
        opened_times = med_times_df['opened_times'][i: i + 4]
        # medication rate is valid for six hours
        for trigger_time, open_time in zip(triggered_times, opened_times):
            if open_time == 'not_opened':
                medication_rates.extend([0] * 4)
            else:
                open_time = datetime.strptime(str(open_time) ,'%Y-%m-%d %H:%M:%S.%f').strftime('%H:%M')
                open_time = datetime.strptime(str(open_time) ,'%H:%M')
                duration = (open_time - trigger_time).seconds
                if (duration <= 1800): # medication taken within 30 minutes
                    medication_rates.extend([1] * 4)
                elif (1800 < duration <= 3600):
                    medication_rates.extend([0.75] * 4)
                elif (3600 < duration < 7200):
                    medication_rates.extend([0.5] * 4)
                else:
                    medication_rates.extend([0.25] * 4)
    return medication_rates
    

def add_medication_rates(context_df, medication_rates):
    medication_rates_list = []
    for i in range(0, int(len(medication_rates)/16)):
        medication_rates_list.extend(['null'] * 8)
        medication_rates_list.extend(medication_rates[i*16:i*16 + 16])
    context_df['medication_rate'] = medication_rates_list
    return context_df


def get_medication_taken(medication_df):
    medicines_taken = []
    medicines = []
    A = medication_df[medication_df.columns.difference(['date'])]
    for col in A:
        medicines.append(col)
    for index, row in A.iterrows():
        values = row.tolist()
        for i in range(0, len(values), 2):
            daily_medicines = list(itertools.compress(medicines[i:i+2], values[i:i+2]))
            if not daily_medicines:
                daily_medicines = ['medicines not taken']
                medicines_taken.extend([daily_medicines] * 4)
            else:
                medicines_taken.extend([daily_medicines] * 4) # medicine for every 4 hours
    return medicines_taken


def add_medication_taken(context_df, medicines_taken):
    medicines_taken_list = []
    for i in range(0, int(len(medicines_taken)/16)):
        medicines_taken_list.extend(['null'] * 8)
        medicines_taken_list.extend(medicines_taken[i*16:i*16 + 16])
    context_df['medicines'] = medicines_taken_list
    return context_df


# Anomaly Detection

def detect_anomalies():
    acitivity_records = get_activities_nodes()
    activity_nodes = list(set([record['m.name'] for record in acitivity_records]))
    vital_records = get_vitals_nodes()
    vital_nodes = [record['n.name'] for record in vital_records]
    # for activity in activity_nodes:
    for activity in activity_nodes:
        for vital in vital_nodes:
            # get threshold
            threshold_result = get_threshold(activity, vital).data()
            if threshold_result:
                threshold = threshold_result[0]['n.' + str(activity)]
                if threshold:
                    vital_node_name = '[' + ','.join(threshold) + ']'
                    # get timestamps and the common between them
                    timestamps_vital_result = get_timestamps_anomalous_vitals(vital, vital_node_name).data()
                    timestamps_activites_result = get_timestamps_context_nodes('activity', activity).data()
                    timestamps_vital_list = [d['type(r)'] for d in timestamps_vital_result]
                    anomalous_vitals_name = list(set([d['m.name'] for d in timestamps_vital_result]))
                    timestamps_activites_list = [d['type(r)'] for d in timestamps_activites_result]
                    occ_timestamps = [element for element in timestamps_activites_list if element in 
                                     timestamps_vital_list]
                    print(vital_node_name)
                    print(anomalous_vitals_name)
                    if activity == 'sleeping' and vital =='heart_rate':
                        print(occ_timestamps)
                    # set vital with anomalous name and timestamp
                    # for anomaly in anomalous_vitals_name:
                    #     for timestamp in occ_timestamps:
                    #         set_vital_anomalous(anomaly, timestamp)
                            
