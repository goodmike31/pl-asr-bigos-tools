from prefect import flow
from prefect_flows.tasks import load_hf_dataset_split, check_cached_hyps_size_and_coverage, cached_hyps_stats_to_df
from asr_systems import initialize_asr_system

@flow(name="ASR Hypothesis Statistics Calculation Flow")
def asr_hyp_stats(config_user, config_common, config_runtime):

    cached_hyps_stats={}
    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    max_samples_per_subset = config_runtime["max_samples_per_subset"]

    for system in systems:
        for model in config_runtime["systems"][system]["models"]:
            asr_system = initialize_asr_system(system, model, config_user)  # Assume this is defined
            print("ASR system initialized")
            asr_system_codename=asr_system.get_codename()
            for dataset_name in datasets:
                for subset in subsets:
                    for split in splits:
                        dataset_codenamed=dataset_name+"_"+subset+"_"+split
                        cached_hyps_stats[dataset_codenamed]={}
                        cached_hyps_stats[dataset_codenamed][asr_system_codename]={}
                        

                        hf_dataset = load_hf_dataset_split(dataset_name, split, subset)
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

                        cached_hyps_stats[dataset_codenamed][asr_system_codename]["nr_of_cached_hyps"]=nr_of_cached_hyps
                        cached_hyps_stats[dataset_codenamed][asr_system_codename]["nr_of_common_audio_paths"]=nr_of_common_audio_paths
                        cached_hyps_stats[dataset_codenamed][asr_system_codename]["nr_of_missing_audio_paths"]=nr_of_missing_audio_paths
                        cached_hyps_stats[dataset_codenamed][asr_system_codename]["hyps_coverage"]=hyps_coverage


    # convert cached_hyps_stats to a dataframe
    df_cached_hyps_stats = cached_hyps_stats_to_df(cached_hyps_stats)
    # save the dataframe to a file
    #df_cached_hyps_stats.to_csv("cached_hyps_stats.csv")
    
