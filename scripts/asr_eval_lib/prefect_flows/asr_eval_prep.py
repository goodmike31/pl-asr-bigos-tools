from prefect import flow
from prefect_flows.tasks import load_hf_dataset, select_split_of_dataset, prepare_eval_input_from_hyps_cache
import pandas as pd
from datetime import datetime
from asr_systems import initialize_asr_system
from pathlib import Path
from config_utils import get_config_run 

import os

"""
Input: dataset refs + hyps.cache
Output: eval_input.tsv with all available ref types and hyps for each dataset, subset, split, system, model, version

"""
def generate_eval_input(config_user, config_common, config_runtime, force=False):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets, subsets, splits, systems, eval_run_codename = get_config_run(config_runtime)
    max_samples_per_subset = config_runtime["max_samples_per_subset"]
    
    bigos_eval_data_dir = config_user["PATHS"]["BIGOS_EVAL_DATA_REPO_PATH"]
    print("bigos_eval_data_dir", bigos_eval_data_dir)

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
                            dataset_codename = str.join("-", [dataset, subset, split])
                            # generate eval input for specific dataset, subset, split, system, model and version
                            # TODO move eval input dir root to config
                            
                            eval_input_dir = os.path.join(bigos_eval_data_dir, "eval_input", asr_system.get_codename(), version, dataset_codename, eval_run_codename)
                            os.makedirs(eval_input_dir, exist_ok=True)
                            eval_input_df_path = os.path.join(eval_input_dir, "eval_input.tsv")
                            if not os.path.exists(eval_input_df_path) or force:
                                print("Generating eval input DF for evaluation based on cached hypotheses.")
                                hf_dataset_split = select_split_of_dataset(hf_dataset, split)
                                # prepare eval input from hyps cache
                                eval_input_df = prepare_eval_input_from_hyps_cache(hf_dataset_split, asr_system, max_samples_per_subset)
                                # save eval input
                                eval_input_df.to_csv(eval_input_df_path, sep="\t", index=False)
                            else:
                                print("Input DF for evaluation already exists. Skipping generation.")
                                print("eval_input_df_path", eval_input_df_path)
                                continue


@flow(name="ASR Evaluation Preparation Flow")
def asr_eval_prep(config_user, config_common, config_runtime, force):
    generate_eval_input(config_user, config_common, config_runtime, force)
