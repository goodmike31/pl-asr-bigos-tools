from prefect_flows.asr_hyp_gen_flow import asr_hyp_gen_flow
from prefect_flows.asr_eval_prep import asr_eval_prep
from prefect_flows.asr_eval_run import asr_eval_run
import argparse
import os
import json
import configparser

from typing import List

def read_config_user(config_user_path):
    config = configparser.ConfigParser()
    config.read(config_user_path)
    return config

def read_config_common(config_common_path):
    with open(config_common_path, "r") as f:
        config = json.load(f)
    return config

# Example execution (you can also run this flow from CLI or Prefect UI)
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='Script for generating ASR hypotheses for a given set of datasets, asr systems and models.')
    parser.add_argument('--config_runtime', type=str, help='Path to runtime config file', default=os.path.join(script_dir, "../../config/eval-run-specific/pl-asr-bigos-default.json"))
    parser.add_argument('--flow', type=str, help='Flow to execute: GEN, EVAL_PREP, EVAL_RUN or ALL', default="ALL")
    
    args = parser.parse_args()
                   
    # Default location of config files.
    config_common_path = os.path.join(script_dir, '../../config/common/config.json')
    print("config_common_path", config_common_path)
    
    config_user_path = os.path.join(script_dir, '../../config/user-specific/config.ini')
    print("config_user_path", config_user_path)

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

    config_user = read_config_user(config_user_path)
    config_common = read_config_common(config_common_path)

    #TODO - add support for "all" subset and split
    #"pwr-maleset-unk"
    #
    print(args.config_runtime)

    with open(args.config_runtime, "r") as f:
        config_runtime = json.load(f)

    if args.flow == "ALL":
        asr_hyp_gen_flow(config_user, config_common, config_runtime)
        asr_eval_prep(config_user, config_common, config_runtime)
        asr_eval_run(config_user, config_common, config_runtime)
    elif args.flow == "GEN":
        asr_hyp_gen_flow(config_user, config_common, config_runtime)
    elif args.flow == "EVAL_PREP":
        asr_eval_prep(config_user, config_common, config_runtime)
    elif args.flow == "EVAL_RUN":
        asr_eval_run(config_user, config_common, config_runtime)