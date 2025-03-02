"""
BIGOS ASR Evaluation Framework - Main Entry Point

This script serves as the primary entry point for the BIGOS ASR evaluation framework.
It orchestrates the different flows of the evaluation process:

1. Hypothesis Generation (HYP_GEN): Generate ASR hypotheses for audio samples
2. Evaluation Preparation (EVAL_PREP): Prepare data for evaluation
3. Evaluation Execution (EVAL_RUN): Run evaluation metrics calculation
4. Hypothesis Statistics (HYP_STATS): Calculate statistics about cached hypotheses
5. Manual Inspection Preparation (PREP_EVAL_RESULTS_INSPECTION): Prepare data for manual inspection

Each flow can be run independently or together as part of a complete evaluation pipeline.
The script uses configuration files to determine which datasets, ASR systems, and evaluation
parameters to use.

Usage:
    python main.py --eval_config=<config_name> [--flow=<flow_name>] [--force=True] [--force_hyps=True]

Example:
    python main.py --eval_config=bigos --flow=HYP_GEN --force_hyps=True

Args:
    --eval_config: Name of the runtime configuration file (without .json extension)
    --flow: Name of the flow to execute (ALL, HYP_GEN, EVAL_PREP, EVAL_RUN, HYP_STATS, PREP_EVAL_RESULTS_INSPECTION)
    --force: Whether to force execution of evaluation flows
    --force_hyps: Whether to force regeneration of hypotheses
"""

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

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='BIGOS ASR Evaluation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the complete evaluation pipeline
  python main.py --eval_config=bigos

  # Generate ASR hypotheses only
  python main.py --flow=HYP_GEN --eval_config=bigos

  # Calculate statistics for cached hypotheses
  python main.py --flow=HYP_STATS --eval_config=bigos
        """
    )
    parser.add_argument('--eval_config', type=str, 
                        help='Name of the runtime config file', 
                        default="TEST")
    parser.add_argument('--flow', type=str, 
                        help='Flow to execute: ALL, HYP_GEN, EVAL_PREP, EVAL_RUN, HYP_STATS, PREP_EVAL_RESULTS_INSPECTION', 
                        default="ALL")
    parser.add_argument('--force', type=bool, 
                        help='Force execution of the eval results calculation flows (except hypothesis generation)', 
                        default=False)
    parser.add_argument('--force_hyps', type=bool, 
                        help='Force execution of the hypothesis generation flow', 
                        default=False)
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Validate and load the runtime configuration file
    try:
        config_runtime_file = os.path.join(script_dir, '../../config/eval-run-specific/{}.json'.format(args.eval_config))
        if not os.path.exists(config_runtime_file):
            raise FileNotFoundError(f"Config file not found: {config_runtime_file}")
    except Exception as e:
        print(f"Error loading runtime config: {e}")
        sys.exit(1)
    
    # Set force flags for execution
    force = args.force
    force_hyps = args.force_hyps
    
    print("config_runtime_file", config_runtime_file)
    print("force", force)

    # Load common and user-specific configuration files
    config_common_path = os.path.join(script_dir, '../../config/common/config.json')
    print("config_common_path", config_common_path)
    
    config_user_path = os.path.join(script_dir, '../../config/user-specific/config.ini')
    print("config_user_path", config_user_path)

    # Validate that config files exist
    if not os.path.exists(config_common_path):
        print(f"Common config file does not exist: {config_common_path}")
        sys.exit(1)
        
    if not os.path.exists(config_user_path):
        print(f"User config file does not exist: {config_user_path}")
        print("Please copy config/user-specific/template.ini to config/user-specific/config.ini and edit it")
        sys.exit(1)

    # Load configuration data
    config_user = read_config_ini(config_user_path)
    config_common = read_config_json(config_common_path)

    with open(config_runtime_file, "r") as f:
        config_runtime = json.load(f)

    # Execute the specified flow(s)
    if args.flow == "ALL":
        print("Executing all flows for the runtime config: ", args.eval_config) 
        asr_hyp_gen(config_user, config_common, config_runtime, force_hyps)
        asr_eval_prep(config_user, config_common, config_runtime, force)
        asr_eval_run(config_user, config_common, config_runtime, force)
    elif args.flow == "HYP_GEN":
        print(f"Executing hypothesis generation flow for config: {args.eval_config}")
        asr_hyp_gen(config_user, config_common, config_runtime, force_hyps)
    elif args.flow == "EVAL_PREP":
        print(f"Executing evaluation preparation flow for config: {args.eval_config}")
        asr_eval_prep(config_user, config_common, config_runtime, force)
    elif args.flow == "EVAL_RUN":
        print(f"Executing evaluation run flow for config: {args.eval_config}")
        asr_eval_run(config_user, config_common, config_runtime, force)
    elif args.flow == "HYP_STATS":
        print(f"Executing hypothesis statistics flow for config: {args.eval_config}")
        asr_hyp_stats(config_user, config_common, config_runtime, force)
    elif args.flow == "PREP_EVAL_RESULTS_INSPECTION":
        print(f"Executing manual inspection preparation flow for config: {args.eval_config}")
        asr_eval_man_inspect_prep(config_user, config_common, config_runtime, force)
    else:
        print(f"Unknown flow name: {args.flow}")
        print("Available flows: ALL, HYP_GEN, EVAL_PREP, EVAL_RUN, HYP_STATS, PREP_EVAL_RESULTS_INSPECTION")
        sys.exit(1)