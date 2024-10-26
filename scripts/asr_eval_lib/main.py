
from prefect_flows.asr_hyp_gen import asr_hyp_gen
from prefect_flows.asr_eval_prep import asr_eval_prep
from prefect_flows.asr_eval_run import asr_eval_run
from prefect_flows.asr_hyp_stats import asr_hyp_stats
from prefect_flows.asr_eval_man_inspect_prep import asr_eval_man_inspect_prep
from scripts.utils.utils import read_config_ini, read_config_json
from typing import List
import argparse
import os
import json
import sys

# Get the parent directory
repo_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
print("repo_root_dir", repo_root_dir)

# Add the parent directory to sys.path
sys.path.insert(0, repo_root_dir)

# Example execution (you can also run this flow from CLI or Prefect UI)
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='Script for generating ASR hypotheses for a given set of datasets, asr systems and models.')
    parser.add_argument('--eval_config', type=str, help='Name of the runtime config file', default="TEST")
    parser.add_argument('--flow', type=str, help='Flow to execute: GEN, EVAL_PREP, EVAL_RUN or ALL', default="ALL")
    parser.add_argument('--force', type=bool, help='Force execution of the eval results calculation flows (except hypothesis generation)', default=False)
    parser.add_argument('--force_hyps', type=bool, help='Force execution of the hypothesis generation flow', default=False)
    
    args = parser.parse_args()

    try:
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/{}.json'.format(args.eval_config))
    except:
        print("Error: Config file not found for the runtime")
        exit(1)

    force = args.force
    force_hyps = args.force_hyps
    
    print("config_runtime_file", config_runtime_file)
    print("force", force)

    # Default location of config files.
    config_common_path = os.path.join(script_dir, '../../config/common/config.json')
    print("config_common_path", config_common_path)
    
    config_user_path = os.path.join(script_dir, '../../config/user-specific/config.ini')
    print("config_user_path", config_user_path)

    config_user = read_config_ini(config_user_path)
    config_common = read_config_json(config_common_path)

    with open(config_runtime_file, "r") as f:
        config_runtime = json.load(f)

    if args.flow == "ALL":
        print("Executing all flows for the runtime config: ", args.eval_config) 
        asr_hyp_gen(config_user, config_common, config_runtime, force_hyps)
        asr_eval_prep(config_user, config_common, config_runtime, force)
        asr_eval_run(config_user, config_common, config_runtime, force)
    elif args.flow == "HYP_GEN":
        asr_hyp_gen(config_user, config_common, config_runtime, force_hyps)
    elif args.flow == "EVAL_PREP":
        asr_eval_prep(config_user, config_common, config_runtime, force)
    elif args.flow == "EVAL_RUN":
        asr_eval_run(config_user, config_common, config_runtime, force)
    elif args.flow == "HYP_STATS":
        asr_hyp_stats(config_user, config_common, config_runtime, force)
    elif args.flow == "PREP_EVAL_RESULTS_INSPECTION":
        asr_eval_man_inspect_prep(config_user, config_common, config_runtime, force)
    else:
        print("Unknown flow name. Exiting.")
        exit(1)