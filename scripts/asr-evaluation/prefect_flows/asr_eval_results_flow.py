from prefect import flow
from prefect_flows.tasks import load_hf_dataset, select_split_of_dataset, prepare_eval_input_from_hyps_cache, calculate_eval_metrics, save_metrics_tsv, save_metrics_json 
import pandas as pd
from datetime import datetime
from asr_systems import initialize_asr_system

@flow(name="ASR Evaluation Results Flow")
def asr_eval_results_flow(config_user, config_common, config_runtime):

    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    
    # make sure that required hypothesis for specific systems, models, version and datasets are converted into eval input format
    for system in systems:
        for model in config_runtime["systems"][system]["models"]:
            for version in config_runtime["systems"][system]["versions"]:
                asr_system = initialize_asr_system(system, model, config_user) 
                for dataset_name in datasets:
                    for subset in subsets:
                        hf_dataset = load_hf_dataset(dataset_name, subset)
                        for split in splits:
                            hf_dataset = select_split_of_dataset(hf_dataset, split)
                            # prepare eval input from hyps cache
                            eval_input_df = prepare_eval_input_from_hyps_cache(hf_dataset, asr_system)
                            #eval_input_df.sample(10)

                            # save eval input
                            #save_eval_input(eval_input_df, eval_input_path)
                            
    # calculate evaluation metrics for all available hypotheses
    #calculate_eval_metrics("eval-config.json")    
    
    # for specified eval configuration, calculate evaluation metrics for each dataset, subset, split, system and model
    #for dataset in datasets_to_process:
    #    for subset in subsets:
    #        for split in splits:
    #            for system in systems:
    #                for model in models[system]:
                        # calculate evaluation metrics for specific dataset, subset, split, system and model
                        # TODO consider adding version as input argument
                        # TODO consider adding postnorm as input argument
                        # TODO consider adding evalnorm as input argument
                        # TODO consider adding ref_type as input argument
                        # TODO consider adding eval_type as input argument
                        # TODO consider "all" for subsets as the input argument
    #                    pass
                        #eval_metrics = get_eval_metrics(eval_metrics_df, dataset, subset, split, system, model)
                        # save evaluation metrics
                        #save_metrics_tsv(eval_metrics)
                        #save_metrics_json(eval_metrics)
                        #print(final_result)        