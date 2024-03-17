

def get_config_run(config_runtime)->list:

    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    eval_run_codename = config_runtime["eval_run_codename"]
    return datasets, subsets, splits, systems, eval_run_codename