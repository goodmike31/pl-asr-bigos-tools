import json
import configparser

def read_config_ini(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def read_config_json(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config