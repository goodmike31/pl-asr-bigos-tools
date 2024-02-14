
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
    parser.add_argument('--eval_config', type=str, help='Name of the runtime config file', default="BIGOS")
    parser.add_argument('--flow', type=str, help='Flow to execute: GEN, EVAL_PREP, EVAL_RUN or ALL', default="ALL")
    
    args = parser.parse_args()

    if (args.eval_config == "BIGOS"):
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/bigos-default.json')
    elif (args.eval_config == "PELCRA"):
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/pelcra-default.json')
    elif (args.eval_config == "TEST"):
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/test.json')
    else:
        print("Unknown runtime name. Exiting.")
        exit(1)
    
    # Default location of config files.
    config_common_path = os.path.join(script_dir, '../../config/common/config.json')
    print("config_common_path", config_common_path)
    
    config_user_path = os.path.join(script_dir, '../../config/user-specific/config.ini')
    print("config_user_path", config_user_path)

    config_user = read_config_user(config_user_path)
    config_common = read_config_common(config_common_path)

    with open(config_runtime_file, "r") as f:
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