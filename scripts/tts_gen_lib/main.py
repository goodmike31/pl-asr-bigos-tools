
from prefect_flows.tts_gen import tts_gen
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
    parser.add_argument('--tts_set_config', type=str, help='Name of the config file', default="TEST")
    parser.add_argument('--flow', type=str, help='Flow to execute: TTS_SET_GEN or ALL', default="ALL")
    parser.add_argument('--force', type=bool, help='Force generation of new audios for TTS', default=False)
    
    args = parser.parse_args()

    if (args.tts_set_config == "AMU-MED-ALL"):
        config_runtime_file = os.path.join(script_dir, '../../config/tts-set-specific/amu-med-all.json')
    elif (args.tts_set_config == "AMU-MED-ADOLESC"):
        config_runtime_file = os.path.join(script_dir, '../../config/tts-set-specific/amu-med-adolesc.json')
    else:
        print("Unknown TTS config name. Exiting.")
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
        tts_gen(config_user, config_common, config_runtime)
    elif args.flow == "TTS_SET_GEN":
        tts_gen(config_user, config_common, config_runtime)
    else:
        print("Unknown flow name. Exiting.")
        exit(1)