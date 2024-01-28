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

def generate_eval_input(config_user, config_common, config_runtime):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets, subsets, splits, systems = get_config_run(config_runtime)

    # make sure that required hypothesis for specific systems, models, version and datasets are converted into eval input format
    for system in systems:
        for model in config_runtime["systems"][system]["models"]:
            for version in config_runtime["systems"][system]["versions"]:
                # TODO add version as input argument to control ASR version somehow
                asr_system = initialize_asr_system(system, model, config_user) 
                for dataset in datasets:
                    for subset in subsets:
                        hf_dataset = load_hf_dataset(dataset, subset)
                        for split in splits:
                            # generate eval input for specific dataset, subset, split, system, model and version
                            eval_input_dir = os.path.join(script_dir, "data/eval_input", dataset, subset, split, asr_system.get_codename())
                            os.makedirs(eval_input_dir, exist_ok=True)
                            eval_input_df_path = os.path.join(eval_input_dir, "eval_input.tsv")    
                            if os.path.exists(eval_input_df_path):
                                print("eval_input_df_path exists. Skipping")
                                print("eval_input_df_path", eval_input_df_path)
                                continue
                            else:
                                hf_dataset = select_split_of_dataset(hf_dataset, split)
                                # prepare eval input from hyps cache
                                eval_input_df = prepare_eval_input_from_hyps_cache(hf_dataset, asr_system)
                                # save eval input
                                eval_input_df.to_csv(eval_input_df_path, sep="\t", index=False)

                            
def generate_eval_metrics(config_user, config_common, config_runtime):
    datasets, subsets, splits, systems = get_config_run(config_runtime)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # for specified eval configuration, calculate evaluation metrics for each dataset, subset, split, system and model
    for dataset in datasets:
        for subset in subsets:
            for split in splits:
                dataset_codename = str.join("-", [dataset, subset, split])
                for system in systems:
                    for model in config_runtime["systems"][system]["models"]:
                        asr_system = initialize_asr_system(system, model, config_user) 
                        system_codename = asr_system.get_codename()
                        eval_input_dir = os.path.join(script_dir, "data/eval_input", dataset, subset, split, system_codename )
                        eval_input_df_path = os.path.join(eval_input_dir, "eval_input.tsv")    
                        eval_input_df = pd.read_csv(eval_input_df_path, sep="\t")
                        eval_metrics = calculate_eval_metrics(eval_input_df, dataset_codename, system_codename)
                        print(eval_metrics)
                        # save evaluation metrics
                        #save_metrics_tsv(eval_metrics)
                        #save_metrics_json(eval_metrics)
                        #print(final_result)        


@flow(name="ASR Evaluation Results Flow")
def asr_eval_results_flow(config_user, config_common, config_runtime):
    generate_eval_input(config_user, config_common, config_runtime)
    generate_eval_metrics(config_user, config_common, config_runtime)
