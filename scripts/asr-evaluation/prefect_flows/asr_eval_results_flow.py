from prefect import flow
from prefect_flows.tasks import prepare_eval_input_from_hyps_cache, calculate_eval_metrics, save_metrics_tsv, save_metrics_json 
import pandas as pd
from datetime import datetime

@flow(name="ASR Evaluation Results Flow")
def asr_eval_results_flow(config_user, config_common, datasets_to_process, subsets, splits, systems, models):
    # Assume the version is the latest one - current quarter
    # get current year and quarter
    # TODO consider making it more flexible, e.g. by allowing to specify the version in the config file
    year = datetime.now().year
    quarter = (datetime.now().month-1)//3 + 1
    version = "{}Q{}".format(year, quarter)

    # make sure that required hypothesis for specific systems, models, version and datasets are converted into eval input format
    for system in systems:
        for model in models[system]:
            # read cache and prepare input for evaluation
            # expected output - TODO
            pass
            #eval_input_df = prepare_eval_input_from_hyps_cache(system, model, version)

    # calculate evaluation metrics for all available hypotheses
    calculate_eval_metrics("eval-config.json")    
    
    # for specified eval configuration, calculate evaluation metrics for each dataset, subset, split, system and model
    for dataset in datasets_to_process:
        for subset in subsets:
            for split in splits:
                for system in systems:
                    for model in models[system]:
                        # calculate evaluation metrics for specific dataset, subset, split, system and model
                        # TODO consider adding version as input argument
                        # TODO consider adding postnorm as input argument
                        # TODO consider adding evalnorm as input argument
                        # TODO consider adding ref_type as input argument
                        # TODO consider adding eval_type as input argument
                        # TODO consider "all" for subsets as the input argument
                        pass
                        #eval_metrics = get_eval_metrics(eval_metrics_df, dataset, subset, split, system, model)
                        # save evaluation metrics
                        #save_metrics_tsv(eval_metrics)
                        #save_metrics_json(eval_metrics)
                        #print(final_result)        