"""
ASR Evaluation Flow Module

This module defines Prefect flows for running ASR evaluation, including functions for calculating
metrics on both per-dataset and per-sample levels. It orchestrates the evaluation pipeline, handles
data loading, metrics calculation, and results persistence.
"""
from prefect import flow
from prefect_flows.tasks import calculate_eval_metrics_per_dataset, calculate_eval_metrics_per_sample, save_metrics_tsv, save_metrics_json, load_hf_dataset_split
from config_utils import get_config_run 
import pandas as pd
from datetime import datetime
import os
from datasets import get_dataset_config_names

today = datetime.now().strftime("%Y%m%d")


def get_pretty_column_names(dataset, split):
    """
    Create a mapping of dataset config names to shortened display names.
    
    This function generates a dictionary that maps full dataset configuration names
    (including dataset name, config name, and split) to shortened, more readable names
    suitable for display in tables or charts.
    
    Args:
        dataset (str): The dataset name (e.g., "amu-cai/pl-asr-bigos-v2-secret")
        split (str): The dataset split (e.g., "test")
        
    Returns:
        dict: A dictionary mapping full dataset config paths to their shortened names
    """
    config_names = get_dataset_config_names(dataset)[:-1]
    # add split into to config_names e.g. "amu-cai/pl-asr-bigos-v2-secret-> amu-cai/pl-asr-bigos-v2-secret-test"
    # add split into to config_names e.g. "amu-cai/pl-asr-bigos-diagnostic-> amu-cai/pl-asr-bigos-diagnostic-test"
    config_names_new = [dataset + "-" + config_name + "-" + split for config_name in config_names]
    config_names_short = [config_name.split("-")[1].upper() for config_name in config_names]
    #print(config_names_new)
    #print(config_names_short)
    keys = config_names_new
    values = config_names_short
    column_names = dict(zip(keys, values))
    return(column_names)

def generate_sample_eval_metrics_subsets(config_user, config_common, config_runtime, force):
    """
    Generate and save sample-level evaluation metrics for specified ASR systems and datasets.
    
    This function processes evaluation data on a per-sample level by:
    1. Loading evaluation input data for each system/dataset combination
    2. Calculating per-sample evaluation metrics (WER, CER, etc.)
    3. Merging metrics with dataset metadata
    4. Saving results to TSV and JSON files for further analysis
    5. Saving aggregated results to the leaderboard input directory
    
    Args:
        config_user (dict): User-specific configuration containing paths
        config_common (dict): Common configuration settings
        config_runtime (dict): Runtime configuration specifying datasets, systems, and metrics settings
        force (bool): If True, recalculate metrics even if output files already exist
    """
    
    datasets, subsets, splits, systems, eval_run_codename = get_config_run(config_runtime)
    norm_types=config_runtime["norm_types"]
    ref_types=config_runtime["ref_types"]
    print("ref_types", ref_types)

    # read user specific config to retrieve path to repo to store evaluation data
    bigos_eval_data_dir = config_user["PATHS"]["BIGOS_EVAL_DATA_REPO_PATH"]
    print("bigos_eval_data_dir", bigos_eval_data_dir)

    eval_in_dir = os.path.join(bigos_eval_data_dir, "eval_input")
    eval_out_dir_common = os.path.join(bigos_eval_data_dir, "eval_output/per_sample/", eval_run_codename)
    leaderboard_in_dir = os.path.join(bigos_eval_data_dir, "leaderboard_input")
    os.makedirs(leaderboard_in_dir, exist_ok=True)
    os.makedirs(eval_out_dir_common, exist_ok=True)

    # initialize empty dataframe for storing all evaluation metrics
    df_eval_results_all = pd.DataFrame([])
    # for specified eval configuration, calculate evaluation metrics for each dataset, subset, split, system and model
    #TODO - decide on which level aggregate metrics should be calculated (e.g. for each dataset, subset, split, system, model, version, postnorm, evalnorm, ref_type, eval_type, etc.)
    
    for dataset in datasets:
        eval_out_dir_dataset = os.path.join(eval_out_dir_common, dataset, eval_run_codename)
        os.makedirs(eval_out_dir_dataset, exist_ok=True)
        for split in splits:
            for subset in subsets:
                hf_dataset = load_hf_dataset_split(dataset, subset, split)
                
                # convert HF dataset to pandas dataframe
                df_hf_dataset = pd.DataFrame(hf_dataset)
                # round all values to 2 decimal places
                df_hf_dataset = df_hf_dataset.round(2)

                print("HF dataset shape: ", df_hf_dataset.shape)
                print("HF dataset columns: ", df_hf_dataset.columns)
                print("HF dataset sample: ", df_hf_dataset.head(1))

                dataset_codename = str.join("-", [dataset, subset, split])
                eval_out_dir = os.path.join(eval_out_dir_common, dataset_codename, eval_run_codename)
                os.makedirs(eval_out_dir, exist_ok=True)
                print("Calculating evaluation metrics for ", dataset_codename, " dataset.")
                print("Output directory: ", eval_out_dir)
                # load dataset subset
                for system in systems:
                    for model in config_runtime["systems"][system]["models"]:
                        for version in config_runtime["systems"][system]["versions"]:
                            # TODO add flag to skipping initializing model, if asr system is needed just to read cache or name (read_model=False)
                            #TODO move to utils
                            system_codename = str.join("_", [system, model])
                            # TODO move eval input dir root to config
                            eval_input_dir = os.path.join(eval_in_dir, system_codename, version, dataset_codename, eval_run_codename)
                            eval_input_path = os.path.join(eval_input_dir, "eval_input.tsv")
                            print("eval_input_path", eval_input_path)
                            df_eval_input = pd.read_csv(eval_input_path, sep="\t")
                            fn_eval_results_system = os.path.join(eval_out_dir, "eval_results-per_sample-" + system_codename + ".tsv")
                            if not os.path.exists(fn_eval_results_system) or force:
                                #asr_system = initialize_asr_system(system, model, config_user)
                                df_eval_result_no_meta = calculate_eval_metrics_per_sample(df_eval_input, dataset, subset, split, system_codename, ref_types, norm_types)
                                # get columns names not available in df_eval_results but available in hf_dataset_column_names
                                # extend df_eval_results with metadata for specific dataset sample based on the content of hf_dataset
                                # join on column "audiopath_bigos"
                                df_eval_result = pd.merge(df_eval_result_no_meta, df_hf_dataset, how="left", left_on="id", right_on="audiopath_bigos")
                                # drop columns that are not needed for evaluation metrics - split_y	dataset_y audio audiopath_bigos audiopath_local
                                df_eval_result = df_eval_result.drop(columns=["split_y", "dataset_y", "audio", "audiopath_bigos", "audiopath_local"])
                                # rename columns with _x suffix to remove it
                                df_eval_result.columns = df_eval_result.columns.str.replace("_x", "")
                                # rename columns with _y suffix to remove it
                                df_eval_result.columns = df_eval_result.columns.str.replace("_y", "")
                                
                                print("df_eval_results_with_meta shape: ", df_eval_result.shape)
                                print("df_eval_results_with_meta columns: ", df_eval_result.columns)
                                print("df_eval_results_with_meta sample: ", df_eval_result.head(1))

                                save_metrics_tsv(df_eval_result, fn_eval_results_system)
                                save_metrics_json(df_eval_result, fn_eval_results_system.replace(".tsv", ".json"))
                            else:
                                print("Skipping calculation of evaluation metrics for ", system_codename, " and ", dataset_codename)
                                df_eval_result = pd.read_csv(fn_eval_results_system, sep="\t")
                            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_result])

            # Save aggregated evaluation metrics for all datasets, subsets, splits, systems and models
            fn_eval_results_agg = os.path.join(eval_out_dir_dataset, "eval_results-per_sample-all_systems_and_subsets" + datetime.now().strftime("%Y%m%d"))
            save_metrics_tsv(df_eval_results_all, fn_eval_results_agg + ".tsv")

            results_tsv_fn_leaderboard_repo = os.path.join(leaderboard_in_dir, dataset, split, eval_run_codename, "eval_results-per_sample-" + datetime.now().strftime("%Y%m%d") + ".tsv")
            results_tsv_fn_leaderboard_latest = os.path.join(leaderboard_in_dir, dataset, split , "eval_results-per_sample-latest.tsv")
            os.makedirs(os.path.dirname(results_tsv_fn_leaderboard_repo), exist_ok=True)
            
            save_metrics_tsv(df_eval_results_all, results_tsv_fn_leaderboard_repo)
            save_metrics_tsv(df_eval_results_all, results_tsv_fn_leaderboard_latest)


        #TODO - calculate WER per audio duration bucket and plot results

def generate_agg_eval_metrics_subsets(config_user, config_common, config_runtime, force):
    """
    Generate and save aggregated evaluation metrics for datasets.
    
    This function processes evaluation data on a per-dataset level by:
    1. Loading evaluation input data for each system/dataset combination
    2. Calculating aggregated metrics across the entire dataset
    3. Applying normalization lexicons if available
    4. Saving results to TSV and JSON files
    5. Saving aggregated results to the leaderboard input directory
    
    Args:
        config_user (dict): User-specific configuration containing paths
        config_common (dict): Common configuration settings
        config_runtime (dict): Runtime configuration specifying datasets, systems, and metrics settings
        force (bool): If True, recalculate metrics even if output files already exist
    """
    
    datasets, subsets, splits, systems, eval_run_codename = get_config_run(config_runtime)
    print("datasets", datasets)
    print("subsets", subsets)
    print("splits", splits)
    print("systems", systems)

    norm_types=config_runtime["norm_types"]
    ref_types=config_runtime["ref_types"]

    # read user specific config to retrieve path to repo to store evaluation data
    bigos_eval_data_dir = config_user["PATHS"]["BIGOS_EVAL_DATA_REPO_PATH"]
    print("bigos_eval_data_dir", bigos_eval_data_dir)

    eval_in_dir = os.path.join(bigos_eval_data_dir, "eval_input")
    eval_out_dir_common = os.path.join(bigos_eval_data_dir, "eval_output/per_sample/", eval_run_codename)
    leaderboard_in_dir = os.path.join(bigos_eval_data_dir, "leaderboard_input")
    os.makedirs(leaderboard_in_dir, exist_ok=True)
    os.makedirs(eval_out_dir_common, exist_ok=True)


    # initialize empty dataframe for storing all evaluation metrics
    df_eval_results_all = pd.DataFrame([])
    # for specified eval configuration, calculate evaluation metrics for each dataset, subset, split, system and model
    #TODO - decide on which level aggregate metrics should be calculated (e.g. for each dataset, subset, split, system, model, version, postnorm, evalnorm, ref_type, eval_type, etc.)
    
    for dataset in datasets:
        eval_out_dir_dataset = os.path.join(eval_out_dir_common, dataset, eval_run_codename)
        os.makedirs(eval_out_dir_dataset, exist_ok=True)
        norm_lexicon_path = os.path.join("./data/norm_lexicons", dataset + ".csv")
        print("norm_lexicon_path", norm_lexicon_path)
        if not os.path.exists(norm_lexicon_path):
            print("Norm lexicon file does not exist. Exiting.")
            norm_lexicon = None
        else:
            norm_lexicon_df = pd.read_csv(norm_lexicon_path, sep=",")
            norm_lexicon = dict(zip(norm_lexicon_df["orig"], norm_lexicon_df["norm"]))

        for split in splits:
            for subset in subsets:
                dataset_codename = str.join("-", [dataset, subset, split])
                eval_out_dir = os.path.join(eval_out_dir_common, dataset_codename, eval_run_codename)
                os.makedirs(eval_out_dir, exist_ok=True)
                print("Calculating evaluation metrics for ", dataset_codename, " dataset.")
                print("Output directory: ", eval_out_dir)
                for system in systems:
                    for model in config_runtime["systems"][system]["models"]:
                        for version in config_runtime["systems"][system]["versions"]:
                            # TODO add flag to skipping initializing model, if asr system is needed just to read cache or name (read_model=False)
                            #TODO move to utils
                            system_codename = str.join("_", [system, model])
                            # TODO move eval input dir root to config
                            eval_input_dir = os.path.join(eval_in_dir, system_codename, version, dataset_codename, eval_run_codename)
                            eval_input_path = os.path.join(eval_input_dir, "eval_input.tsv")    
                            df_eval_input = pd.read_csv(eval_input_path, sep="\t")
                            fn_eval_results_system = os.path.join(eval_out_dir, "eval_results-per_dataset-" + system_codename + ".tsv")
                            if not os.path.exists(fn_eval_results_system) or force:
                                #asr_system = initialize_asr_system(system, model, config_user)
                                df_eval_result = calculate_eval_metrics_per_dataset(df_eval_input, dataset, subset, split, system_codename, ref_types, norm_types, norm_lexicon)
                                save_metrics_tsv(df_eval_result, fn_eval_results_system)
                                save_metrics_json(df_eval_result, fn_eval_results_system.replace(".tsv", ".json"))
                            else:
                                print("Skipping calculation of evaluation metrics for ", system_codename, " and ", dataset_codename)
                                df_eval_result = pd.read_csv(fn_eval_results_system, sep="\t")
                            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_result])

            # Save aggregated evaluation metrics for all datasets, subsets, splits, systems and models
            fn_eval_results_agg = os.path.join(eval_out_dir_dataset, "eval_results-per_dataset-all_systems_and_subsets-" + datetime.now().strftime("%Y%m%d"))
            
            save_metrics_tsv(df_eval_results_all, fn_eval_results_agg + ".tsv")

            results_tsv_fn_leaderboard_repo = os.path.join(leaderboard_in_dir, dataset, split, eval_run_codename, "eval_results-per_dataset-" + datetime.now().strftime("%Y%m%d") + ".tsv")
            results_tsv_fn_leaderboard_latest = os.path.join(leaderboard_in_dir, dataset, split, "eval_results-per_dataset-latest.tsv")
            os.makedirs(os.path.dirname(results_tsv_fn_leaderboard_repo), exist_ok=True)
            
            save_metrics_tsv(df_eval_results_all, results_tsv_fn_leaderboard_repo)
            save_metrics_tsv(df_eval_results_all, results_tsv_fn_leaderboard_latest)

@flow(name="ASR Evaluation Execution Flow")
def asr_eval_run(config_user, config_common, config_runtime, force):
    """
    Main Prefect flow for executing ASR evaluation pipeline.
    
    This flow orchestrates the full ASR evaluation process by sequentially calling
    functions to generate both aggregated dataset-level metrics and detailed
    sample-level metrics for all specified systems and datasets.
    
    Args:
        config_user (dict): User-specific configuration containing paths
        config_common (dict): Common configuration settings 
        config_runtime (dict): Runtime configuration specifying datasets, systems, and metrics settings
        force (bool): If True, recalculate metrics even if output files already exist
    """
    generate_agg_eval_metrics_subsets(config_user, config_common, config_runtime, force)
    generate_sample_eval_metrics_subsets(config_user, config_common, config_runtime, force)