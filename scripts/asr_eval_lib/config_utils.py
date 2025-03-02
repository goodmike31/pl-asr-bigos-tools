#TODO - merge with bigos_utils_eval file and rename to "helpers.py"

def get_config_run(config_runtime)->list:
    """
    Extract runtime configuration parameters from the config_runtime dictionary.
    
    This function retrieves key runtime parameters used for ASR evaluation from
    the provided configuration dictionary.
    
    Args:
        config_runtime (dict): A dictionary containing runtime configuration parameters
                              with keys for datasets, subsets, splits, systems, and 
                              eval_run_codename.
    
    Returns:
        tuple: A tuple containing the following elements:
            - datasets (list): List of datasets to be evaluated
            - subsets (list): List of subsets within the datasets
            - splits (list): List of data splits (e.g., train, dev, test)
            - systems (list): List of ASR systems to evaluate
            - eval_run_codename (str): Identifier for the evaluation run
    """

    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    eval_run_codename = config_runtime["eval_run_codename"]
    return datasets, subsets, splits, systems, eval_run_codename