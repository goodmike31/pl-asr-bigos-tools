from prefect_flows.asr_hyp_gen_flow import asr_hyp_gen_flow
from prefect_flows.asr_eval_results_flow import asr_eval_results_flow
import argparse
import os
import json
import configparser

from typing import List

def read_user_config(user_config_path):
    config = configparser.ConfigParser()
    config.read(user_config_path)
    return config

def read_common_config(common_config_path):
    with open(common_config_path, "r") as f:
        config = json.load(f)
    return config

# Example execution (you can also run this flow from CLI or Prefect UI)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating ASR hypotheses for a given set of datasets, asr systems and models.')

    # Default location of config files.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    common_config_path = os.path.join(script_dir, '../../config/common/config.json')
    print("common_config_path", common_config_path)
    
    user_config_path = os.path.join(script_dir, '../../config/user-specific/config.ini')
    print("user_config_path", user_config_path)

    # Optional CLI arguments. If not specified, the default values are used.
    # User can specify custom location of output files or cache.
    #parser.add_argument('--output_dir', type=str, help='Custom directory to save results to', default=os.path.join(bigos_repo_root_dir, "data/asr_hypotheses"))
    #parser.add_argument('--asr_hyps_cache_dir', type=str, help='Custom location cache files storing already generated ASR transcriptions',
    #                    default=os.path.join(bigos_repo_root_dir, "data/asr_hyps_cache"))

    # User can specify custom dataset to process. HF format required "account/dataset_name"
    # Add missing import statements here
    #parser.add_argument('--input_dataset', type=str, help='Custom dataset to read audio paths from', default='all')
    #parser.add_argument('--split', type=str, help='Split to convert. Default = all splits from the common config file',default='all')

    #args = parser.parse_args()

    user_config = read_user_config(user_config_path)
    common_config = read_common_config(common_config_path)

    datasets = ["amu-cai/pl-asr-bigos-v2"]
    #datasets = user_config["datasets_to_eval"]
    splits = ["test"]
    subsets = ["pwr-viu-unk", "pwr-maleset-unk"]

    # get for each dataset inside the loop
    #systems = ["google", "azure", "whisper_cloud", "whisper_local"]
    systems = ["whisper_local"]
    models = {  "google": ["default", "command_and_search", "latest_long", "latest_short"],
                "azure": ["latest"],
                "whisper_cloud": ["whisper-1"],
                "whisper_local": ["tiny", "base", "medium", "large", "large-v1", "large-v2"]
                }

    #asr_hyp_gen_flow(user_config, common_config, datasets, subsets, splits, systems, models)
    asr_eval_results_flow(user_config, common_config, datasets, subsets, splits, systems, models)