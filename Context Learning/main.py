"""
*
* Created by    -   Iman Sonji (MT 3217)
* Creation date -   27 July 2021
* Description   -   The purpose of this module is to for correlations learning and associaton rule mining 
* between the contextual attributes
"""

import pandas as pd
import time
import helpers
from utils import *
import context_states_modeling


data_url = 'Data Generation/generated_datasets/'


def context_clustering(df_vitals, df_env, n_clusters):
    clustered_df = pd.DataFrame()
    
    # cluster data and find thresholds
    clustered_vitals, vitals_mean_std, vitals = gmm_cluster_vitals(df_vitals, n_clusters)
    clustered_env, env_mean_std, env = gmm_cluster_env(df_env, n_clusters)

    # find dominant label
    vitals_dom_labels, env_dom_labels = find_dominant_labels(clustered_vitals, clustered_env, vitals, env)
    # replace labels with thresholds
    clustered_df = write_vitals_env(vitals_dom_labels, env_dom_labels, clustered_df)
    vitals_thresholds, env_thresholds = find_thresholds(vitals_mean_std, env_mean_std, n_clusters, vitals, env)
    clustered_df = replace_labels_with_thresholds(vitals_thresholds, env_thresholds, clustered_df)
    return clustered_df


def analyze_activity_data(df_fitbit):
    start_times, end_times = get_activity_dates(df_fitbit)
    return start_times, end_times


def aggregate_vitals_env(vitals_env_clustered_df):
    # add vitals, environmental data
    context_states = vitals_env_clustered_df
    dates = helpers.create_timerange_sixty('07/12/2021', '12/12/2021 23:00:00')
    context_states.insert(loc=0, column='time', value=dates)
    return context_states


def aggregate_activity(context_states, start_times, end_times):
    # add location, activity 
    context_states = add_location(start_times, end_times, context_states)
    context_states = adjust_time_location(context_states)
    context_states = add_activities_to_context(context_states)
    context_states["location"].replace({1: "outdoors", 0: "indoors"}, inplace=True)
    return context_states


def learning(min_support, min_confidence, k):
    oneItemSet = get_itemset_list()
    freqOneItemSet, keyTotalCount = get_freq_one_itemset(oneItemSet, min_support)

    combinations = get_union(freqOneItemSet, k)

    # count support for each itemset
    freqSets, itemSetSupport = get_freq_itemset(combinations, keyTotalCount, min_support)
    # get rules
    rules = get_association_rules(freqSets, itemSetSupport, freqOneItemSet, min_confidence)
    # prune association rules
    # TODO  
    # prune afer removing duplicates
    df_rules = create_df_rules(rules)
    df_reduced_rules, reduced_rules = remove_duplicated_rules(df_rules)
    prune_rules(reduced_rules)
    return df_reduced_rules


if __name__ == "__main__":
    start_time = time.time()

    # Read raw data from CSV files
    df_vitals = pd.read_csv(data_url + 'vitals.csv', parse_dates=True)
    df_environmental = pd.read_csv(data_url + 'environmental.csv', parse_dates=True)
    df_fitbit = pd.read_csv(data_url + 'fitbit_activity_data.csv', parse_dates=True)
    df_medication_times = pd.read_csv(data_url + 'medication_times.csv', parse_dates=True)
    df_medication = pd.read_csv(data_url + 'medication.csv', parse_dates=True)

    # Preprocess data
    vitals_env_clustered_df = context_clustering(df_vitals, df_environmental, n_clusters=4)
    start_times, end_times = analyze_activity_data(df_fitbit)
    medication_rates = calculte_medication_adherence(df_medication_times)
    medication_taken = get_medication_taken(df_medication)

    # Integrate context 
    context_states = aggregate_vitals_env(vitals_env_clustered_df)
    context_states = aggregate_activity(context_states, start_times, end_times)
    context_states = add_medication_rates(context_states, medication_rates)
    context_states = add_medication_taken(context_states, medication_taken)
    context_states = split_time_day(context_states)
    
    # Create Context states 
    helpers.export_data_to_csv(context_states, 'Context Learning/context_states.csv')

    # Model Data
    context_states_modeling.model_data()
    print("Context States are Modeled --- %s seconds ---" % (time.time() - start_time))

    # Mining association rules
    df_rules = learning(min_support=0.0045, min_confidence=0.5, k=2)
    helpers.export_data_to_csv(df_rules, 'Context Learning/rules.csv')
    print("Context learned --- %s seconds ---" % (time.time() - start_time))

    # update meta model and build weighted graph
    build_weighted_graph(df_rules)
    update_meta_model(df_rules)
    print("Meta model updated --- %s seconds ---" % (time.time() - start_time))

    # detect anomalies and set them in graph
    # This function needs to be optimized.
    # detect_anomalies()
    print("Anomalies detected --- %s seconds ---" % (time.time() - start_time))
    