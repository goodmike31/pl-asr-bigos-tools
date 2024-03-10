from prefect import flow
from prefect_flows.tasks import calculate_eval_metrics_per_dataset, calculate_eval_metrics_per_sample, save_metrics_tsv, save_metrics_json
from config_utils import get_config_run 
import pandas as pd
from datetime import datetime
import os
from datasets import get_dataset_config_names

today = datetime.now().strftime("%Y%m%d")


def get_pretty_column_names(dataset, split):
    config_names = get_dataset_config_names(dataset)[:-1]
    #print(config_names)
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
    
    datasets, subsets, splits, systems, eval_run_codename = get_config_run(config_runtime)

    bigos_leaderboard_dir = os.path.join(config_user["PATHS"]["BIGOS_EVAL_LEADERBOARD_DIR"], "data")

    # TODO move to config
    script_dir = os.path.dirname(os.path.realpath(__file__))
    eval_in_dir = os.path.join(script_dir, "../../../data/eval_input")
    eval_out_dir_common = os.path.join(script_dir, "../../../data/eval_output/per_sample/", eval_run_codename)
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
                            print("eval_input_path", eval_input_path)
                            df_eval_input = pd.read_csv(eval_input_path, sep="\t")
                            fn_eval_results_system = os.path.join(eval_out_dir, "eval_results-per_sample" + system_codename + ".tsv")
                            if not os.path.exists(fn_eval_results_system) or force:
                                #asr_system = initialize_asr_system(system, model, config_user)
                                df_eval_result = calculate_eval_metrics_per_sample(df_eval_input, dataset, subset, split, system_codename)
                                save_metrics_tsv(df_eval_result, fn_eval_results_system)
                                save_metrics_json(df_eval_result, fn_eval_results_system.replace(".tsv", ".json"))
                            else:
                                print("Skipping calculation of evaluation metrics for ", system_codename, " and ", dataset_codename)
                                df_eval_result = pd.read_csv(fn_eval_results_system, sep="\t")
                            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_result])

            # Save aggregated evaluation metrics for all datasets, subsets, splits, systems and models
            fn_eval_results_agg = os.path.join(eval_out_dir_dataset, "eval_results-per_sample-all_systems_and_subsets" + datetime.now().strftime("%Y%m%d"))
            save_metrics_tsv(df_eval_results_all, fn_eval_results_agg + ".tsv")

            results_tsv_fn_leaderboard_repo = os.path.join(bigos_leaderboard_dir, dataset, split, eval_run_codename, "eval_results-per_sample-" + datetime.now().strftime("%Y%m%d") + ".tsv")
            results_tsv_fn_leaderboard_latest = os.path.join(bigos_leaderboard_dir, dataset, split , "eval_results-per_sample-latest.tsv")
            os.makedirs(os.path.dirname(results_tsv_fn_leaderboard_repo), exist_ok=True)
            
            save_metrics_tsv(df_eval_results_all, results_tsv_fn_leaderboard_repo)
            save_metrics_tsv(df_eval_results_all, results_tsv_fn_leaderboard_latest)


        #TODO - calculate WER per audio duration bucket and plot results

def generate_agg_eval_metrics_subsets(config_user, config_common, config_runtime, force):
    
    datasets, subsets, splits, systems, eval_run_codename = get_config_run(config_runtime)

    bigos_leaderboard_dir = os.path.join(config_user["PATHS"]["BIGOS_EVAL_LEADERBOARD_DIR"], "data")

    # TODO move to config
    script_dir = os.path.dirname(os.path.realpath(__file__))
    eval_in_dir = os.path.join(script_dir, "../../../data/eval_input")
    eval_out_dir_common = os.path.join(script_dir, "../../../data/eval_output/per_dataset/", eval_run_codename)
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
                            fn_eval_results_system = os.path.join(eval_out_dir, "eval_results-per_dataset" + system_codename + ".tsv")
                            if not os.path.exists(fn_eval_results_system) or force:
                                #asr_system = initialize_asr_system(system, model, config_user)
                                df_eval_result = calculate_eval_metrics_per_dataset(df_eval_input, dataset, subset, split, system_codename)
                                save_metrics_tsv(df_eval_result, fn_eval_results_system)
                                save_metrics_json(df_eval_result, fn_eval_results_system.replace(".tsv", ".json"))
                            else:
                                print("Skipping calculation of evaluation metrics for ", system_codename, " and ", dataset_codename)
                                df_eval_result = pd.read_csv(fn_eval_results_system, sep="\t")
                            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_result])

        # Save aggregated evaluation metrics for all datasets, subsets, splits, systems and models
        fn_eval_results_agg = os.path.join(eval_out_dir_dataset, "eval_results-per_dataset-all_systems_and_subsets-" + datetime.now().strftime("%Y%m%d"))
        
        save_metrics_tsv(df_eval_results_all, fn_eval_results_agg + ".tsv")

        results_tsv_fn_leaderboard_repo = os.path.join(bigos_leaderboard_dir, dataset, split, eval_run_codename, "eval_results-per_dataset-" + datetime.now().strftime("%Y%m%d") + ".tsv")
        results_tsv_fn_leaderboard_latest = os.path.join(bigos_leaderboard_dir, dataset, split, "eval_results-per_dataset-latest.tsv")
        os.makedirs(os.path.dirname(results_tsv_fn_leaderboard_repo), exist_ok=True)
        
        save_metrics_tsv(df_eval_results_all, results_tsv_fn_leaderboard_repo)
        save_metrics_tsv(df_eval_results_all, results_tsv_fn_leaderboard_latest)

@flow(name="ASR Evaluation Execution Flow")
def asr_eval_run(config_user, config_common, config_runtime, force):
    generate_agg_eval_metrics_subsets(config_user, config_common, config_runtime, force)
    generate_sample_eval_metrics_subsets(config_user, config_common, config_runtime, force)