"""
ASR Hypothesis Statistics Module

This module provides functionality for calculating and managing statistics about
ASR system hypotheses. It includes a Prefect flow that processes datasets through 
various ASR systems and collects statistics about hypothesis coverage and availability.

The module handles caching of statistics to avoid redundant calculations and provides
detailed reporting of hypothesis coverage across different datasets.
"""

from prefect import flow
from prefect_flows.tasks import load_hf_dataset_split, check_cached_hyps_size_and_coverage, cached_hyps_stats_to_df
from asr_systems import initialize_asr_system
from datetime import datetime as dt
import pandas as pd
import os

@flow(name="ASR Hypothesis Statistics Calculation Flow")
def asr_hyp_stats(config_user, config_common, config_runtime):
    """
    Calculate statistics about ASR system hypotheses for specified datasets.
    
    This flow processes the specified datasets through various ASR systems,
    collects statistics about hypothesis availability and coverage, and saves
    these statistics to a CSV file. If statistics have already been calculated
    for the day, they will be loaded from cache instead of recalculated.
    
    Args:
        config_user (dict): User configuration containing ASR system settings
        config_common (dict): Common configuration parameters
        config_runtime (dict): Runtime configuration specifying datasets, subsets,
                              splits, systems, and other runtime parameters
    
    Returns:
        pd.DataFrame or None: DataFrame containing hypothesis statistics if calculated,
                              otherwise None if loaded from cache
    """
    config_runtime_name = config_runtime["name"]

    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Set up cache directories
    cache_dir = os.path.join(script_dir, "../../../data/asr_hyps_cache")
    cache_stats_dir = os.path.join(cache_dir, "stats")
    os.makedirs(cache_dir, exist_ok=True)

    cached_hyps_stats={}
    today = dt.now().strftime("%Y%m%d")
    cached_hyps_stats_file = os.path.join(cache_stats_dir, "cached_hyps_stats-{}-{}.csv".format(config_runtime_name, today))

    # Extract runtime configuration parameters
    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    max_samples_per_subset = config_runtime["max_samples_per_subset"]

    # Check if statistics file exists and load it if available
    if(os.path.exists(cached_hyps_stats_file) and os.path.isfile(cached_hyps_stats_file) and os.path.getsize(cached_hyps_stats_file) > 0):
        print("Loading cached hypotheses statistics from file: ", cached_hyps_stats_file)
        cached_hyps_stats = pd.read_csv(cached_hyps_stats_file)
        print(cached_hyps_stats)
        return
    else:
        print("\n\nNo cached hypotheses statistics file found. Calculating and saving to file: {}\n\n\n".format(cached_hyps_stats_file))
        # Iterate through each configured ASR system
        for system in systems:
            for model in config_runtime["systems"][system]["models"]:
                # Initialize the ASR system with the specified model
                asr_system = initialize_asr_system(system, model, config_user)
                print("ASR system initialized")
                asr_system_codename=asr_system.get_codename()
                cached_hyps_stats[asr_system_codename]={}
                
                # Process each dataset, subset and split combination
                for dataset_name in datasets:
                    for subset in subsets:
                        for split in splits:
                            dataset_codename=dataset_name+"_"+subset+"_"+split
                            cached_hyps_stats[asr_system_codename][dataset_codename]={}                        

                            # Load the dataset split and prepare audio paths
                            hf_dataset = load_hf_dataset_split(dataset_name, subset, split)
                            audio_paths = hf_dataset["audiopath_local"]
                            # Limit the number of audio paths according to configuration
                            audio_paths = audio_paths[:max_samples_per_subset]
                            
                            # Check how many hypotheses are already cached for this system and dataset
                            nr_of_cached_hyps, nr_of_common_audio_paths, nr_of_missing_audio_paths = check_cached_hyps_size_and_coverage(asr_system, audio_paths)
                            hyps_coverage = round(nr_of_common_audio_paths/len(audio_paths) * 100, 2)
                            
                            # Print detailed statistics about hypotheses coverage
                            print("Cached hypotheses status for ASR system: {}".format(asr_system_codename))
                            print("Cached hypotheses total: ", nr_of_cached_hyps)
                            print("Cached hypotheses for the Dataset: {}\nSubset: {}\nSplit: {} ".format(dataset_name, subset, split))
                            print("Common audio paths with the dataset: ", nr_of_common_audio_paths)
                            print("Missing audio paths: ", nr_of_missing_audio_paths)
                            print("Hypotheses coverage [%]: ", hyps_coverage)

                            # Store the statistics in our data structure
                            cached_hyps_stats[asr_system_codename][dataset_codename]["nr_of_cached_hyps"]=nr_of_cached_hyps
                            cached_hyps_stats[asr_system_codename][dataset_codename]["nr_of_common_audio_paths"]=nr_of_common_audio_paths
                            cached_hyps_stats[asr_system_codename][dataset_codename]["nr_of_missing_audio_paths"]=nr_of_missing_audio_paths
                            cached_hyps_stats[asr_system_codename][dataset_codename]["hyps_coverage"]=hyps_coverage


        # Convert the nested dictionary to a DataFrame for easier handling and storage
        df_cached_hyps_stats = cached_hyps_stats_to_df(cached_hyps_stats)
        # Save the statistics DataFrame to a CSV file for future use
        print(df_cached_hyps_stats)  # Displaying only the first few rows for brevity
        df_cached_hyps_stats.to_csv(cached_hyps_stats_file, index=False)
        print("Cached hypotheses statistics saved to file: ", cached_hyps_stats_file)
        return df_cached_hyps_stats
