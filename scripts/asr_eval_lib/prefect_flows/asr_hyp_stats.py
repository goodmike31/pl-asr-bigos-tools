from prefect import flow
from prefect_flows.tasks import load_hf_dataset_split, check_cached_hyps_size_and_coverage, cached_hyps_stats_to_df
from asr_systems import initialize_asr_system
from datetime import datetime as dt
import pandas as pd
import os

@flow(name="ASR Hypothesis Statistics Calculation Flow")
def asr_hyp_stats(config_user, config_common, config_runtime):
    config_runtime_name = config_runtime["name"]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(script_dir, "../../../data/asr_hyps_cache")
    cache_stats_dir = os.path.join(cache_dir, "stats")
    os.makedirs(cache_dir, exist_ok=True)

    cached_hyps_stats={}
    today = dt.now().strftime("%Y%m%d")
    cached_hyps_stats_file = os.path.join(cache_stats_dir, "cached_hyps_stats-{}-{}.csv".format(config_runtime_name, today))

    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    max_samples_per_subset = config_runtime["max_samples_per_subset"]

    # check if file exists
    if(os.path.exists(cached_hyps_stats_file) and os.path.isfile(cached_hyps_stats_file) and os.path.getsize(cached_hyps_stats_file) > 0):
        print("Loading cached hypotheses statistics from file: ", cached_hyps_stats_file)
        cached_hyps_stats = pd.read_csv(cached_hyps_stats_file)
        print(cached_hyps_stats)
        return
    else:
        print("\n\nNo cached hypotheses statistics file found. Calculating and saving to file: {}\n\n\n".format(cached_hyps_stats_file))
        for system in systems:
            for model in config_runtime["systems"][system]["models"]:
                asr_system = initialize_asr_system(system, model, config_user)  # Assume this is defined
                print("ASR system initialized")
                asr_system_codename=asr_system.get_codename()
                cached_hyps_stats[asr_system_codename]={}
                for dataset_name in datasets:
                    for subset in subsets:
                        for split in splits:
                            dataset_codename=dataset_name+"_"+subset+"_"+split
                            cached_hyps_stats[asr_system_codename][dataset_codename]={}                        

                            hf_dataset = load_hf_dataset_split(dataset_name, subset, split)
                            audio_paths = hf_dataset["audiopath_local"]
                            # limit the number of audio paths for testing
                            audio_paths = audio_paths[:max_samples_per_subset]
                            
                            nr_of_cached_hyps, nr_of_common_audio_paths, nr_of_missing_audio_paths = check_cached_hyps_size_and_coverage(asr_system, audio_paths)
                            hyps_coverage = round(nr_of_common_audio_paths/len(audio_paths) * 100, 2)
                            
                            print("Cached hypotheses status for ASR system: {}".format(asr_system_codename))
                            print("Cached hypotheses total: ", nr_of_cached_hyps)
                            print("Cached hypotheses for the Dataset: {}\nSubset: {}\nSplit: {} ".format(dataset_name, subset, split))
                            print("Common audio paths with the dataset: ", nr_of_common_audio_paths)
                            print("Missing audio paths: ", nr_of_missing_audio_paths)
                            print("Hypotheses coverage [%]: ", hyps_coverage)

                            cached_hyps_stats[asr_system_codename][dataset_codename]["nr_of_cached_hyps"]=nr_of_cached_hyps
                            cached_hyps_stats[asr_system_codename][dataset_codename]["nr_of_common_audio_paths"]=nr_of_common_audio_paths
                            cached_hyps_stats[asr_system_codename][dataset_codename]["nr_of_missing_audio_paths"]=nr_of_missing_audio_paths
                            cached_hyps_stats[asr_system_codename][dataset_codename]["hyps_coverage"]=hyps_coverage


        # convert cached_hyps_stats to a dataframe
        df_cached_hyps_stats = cached_hyps_stats_to_df(cached_hyps_stats)
        # save the dataframe to a file
        print(df_cached_hyps_stats)  # Displaying only the first few rows for brevity
        df_cached_hyps_stats.to_csv(cached_hyps_stats_file, index=False)
        print("Cached hypotheses statistics saved to file: ", cached_hyps_stats_file)    
