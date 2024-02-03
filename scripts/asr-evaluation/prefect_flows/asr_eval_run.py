from prefect import flow
from prefect_flows.tasks import load_hf_dataset, select_split_of_dataset, prepare_eval_input_from_hyps_cache, calculate_eval_metrics, save_metrics_tsv, save_metrics_json 
import pandas as pd
from datetime import datetime
from asr_systems import initialize_asr_system
from pathlib import Path
import os

def get_config_run(config_runtime)->list:

    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    return datasets, subsets, splits, systems
                            
def generate_eval_metrics(config_user, config_common, config_runtime):
    datasets, subsets, splits, systems = get_config_run(config_runtime)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    df_eval_results_all = pd.DataFrame([])
    # for specified eval configuration, calculate evaluation metrics for each dataset, subset, split, system and model
    for dataset in datasets:
        for subset in subsets:
            for split in splits:
                dataset_codename = str.join("-", [dataset, subset, split])
                for system in systems:
                    for model in config_runtime["systems"][system]["models"]:
                        for version in config_runtime["systems"][system]["versions"]:
                            asr_system = initialize_asr_system(system, model, config_user) 
                            system_codename = asr_system.get_codename()
                            # TODO move eval input dir root to config
                            eval_input_dir = os.path.join(script_dir, "../../../data/eval_input", asr_system.get_codename(), version, dataset_codename )
                            eval_input_path = os.path.join(eval_input_dir, "eval_input.tsv")    
                            df_eval_input = pd.read_csv(eval_input_path, sep="\t")
                            df_eval_result = calculate_eval_metrics(df_eval_input, dataset_codename, system_codename)
                            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_result])
                            # save evaluation metrics
                            #save_metrics_tsv(eval_metrics)
                            #save_metrics_json(eval_metrics)
                            #print(final_result)
    print(df_eval_results_all)   


@flow(name="ASR Evaluation Execution Flow")
def asr_eval_run(config_user, config_common, config_runtime):
    generate_eval_metrics(config_user, config_common, config_runtime)
