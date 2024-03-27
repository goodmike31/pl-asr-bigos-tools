
from prefect_flows.asr_hyp_gen import asr_hyp_gen
from prefect_flows.asr_eval_prep import asr_eval_prep
from prefect_flows.asr_eval_run import asr_eval_run
from prefect_flows.asr_hyp_stats import asr_hyp_stats
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
    parser.add_argument('--eval_config', type=str, help='Name of the runtime config file', default="TEST")
    parser.add_argument('--flow', type=str, help='Flow to execute: GEN, EVAL_PREP, EVAL_RUN or ALL', default="ALL")
    parser.add_argument('--force', type=bool, help='Force execution of the flow and generation of new results', default=False)
    
    args = parser.parse_args()

    if (args.eval_config == "BIGOS"):
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/bigos-default.json')
    elif (args.eval_config == "PELCRA"):
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/pelcra-default.json')
    elif (args.eval_config == "TEST"):
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/test.json')
    elif (args.eval_config == "AMU-MED"):
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/amumed-tts.json')
    elif (args.eval_config == "DIAGNOSTIC"):
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/bigos-diagnostic.json')
        
    else:
        print("Unknown runtime name. Exiting.")
        exit(1)
    
    force = args.force
    
    print("config_runtime_file", config_runtime_file)
    print("force", force)

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
        print("Executing all flows for the runtime config: ", args.eval_config) 
        asr_hyp_gen(config_user, config_common, config_runtime)
        asr_eval_prep(config_user, config_common, config_runtime)
        asr_eval_run(config_user, config_common, config_runtime, force)
    elif args.flow == "HYP_GEN":
        asr_hyp_gen(config_user, config_common, config_runtime)
    elif args.flow == "EVAL_PREP":
        asr_eval_prep(config_user, config_common, config_runtime)
    elif args.flow == "EVAL_RUN":
        asr_eval_run(config_user, config_common, config_runtime, force)
    elif args.flow == "HYP_STATS":
        asr_hyp_stats(config_user, config_common, config_runtime)
    else:
        print("Unknown flow name. Exiting.")
        exit(1)