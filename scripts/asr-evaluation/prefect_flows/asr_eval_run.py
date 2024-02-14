from prefect import flow
from prefect_flows.tasks import calculate_eval_metrics_per_dataset, calculate_eval_metrics_per_sample, save_metrics_tsv, save_metrics_json, generate_plots
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
from datasets import get_dataset_config_names

def get_config_run(config_runtime)->list:

    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    return datasets, subsets, splits, systems

def get_pretty_column_names(dataset, split):
    config_names = get_dataset_config_names(dataset)[:-1]
    #print(config_names)
    # add split into to config_names e.g. "amu-cai/pl-asr-bigos-v2-secret-> amu-cai/pl-asr-bigos-v2-secret-test"
    # add split into to config_names e.g. "amu-cai/pl-asr-bigos-diagnostic-> amu-cai/pl-asr-bigos-diagnostic-test"
    config_names_new = [dataset + "-" + config_name + "-" + split for config_name in config_names]
    config_names_short = [config_name.split("-")[1].upper() for config_name in config_names]
    #print(config_names_new)
    #print(config_names_short)
    keys = config_names_new
    values = config_names_short
    column_names = dict(zip(keys, values))
    return(column_names)

def generate_sample_eval_metrics_subsets(config_user, config_common, config_runtime):
    
    # TODO move to config
    script_dir = os.path.dirname(os.path.realpath(__file__))
    eval_in_dir = os.path.join(script_dir, "../../../data/eval_input")
    eval_out_dir_common = os.path.join(script_dir, "../../../data/eval_output/per_sample")
    datasets, subsets, splits, systems = get_config_run(config_runtime)
    # initialize empty dataframe for storing all evaluation metrics
    df_eval_results_all = pd.DataFrame([])
    # for specified eval configuration, calculate evaluation metrics for each dataset, subset, split, system and model
    #TODO - decide on which level aggregate metrics should be calculated (e.g. for each dataset, subset, split, system, model, version, postnorm, evalnorm, ref_type, eval_type, etc.)
    
    for dataset in datasets:
        eval_out_dir_dataset = os.path.join(eval_out_dir_common, dataset)
        os.makedirs(eval_out_dir_dataset, exist_ok=True)
        for subset in subsets:
            for split in splits:
                dataset_codename = str.join("-", [dataset, subset, split])
                eval_out_dir = os.path.join(eval_out_dir_common, dataset_codename)
                os.makedirs(eval_out_dir, exist_ok=True)
                print("Calculating evaluation metrics for ", dataset_codename, " dataset.")
                print("Output directory: ", eval_out_dir)
                for system in systems:
                    for model in config_runtime["systems"][system]["models"]:
                        for version in config_runtime["systems"][system]["versions"]:
                            # TODO add flag to skipping initializing model, if asr system is needed just to read cache or name (read_model=False)
                            #TODO move to utils
                            system_codename = str.join("_", [system, model])
                            # TODO move eval input dir root to config
                            eval_input_dir = os.path.join(eval_in_dir, system_codename, version, dataset_codename )
                            eval_input_path = os.path.join(eval_input_dir, "eval_input.tsv")    
                            df_eval_input = pd.read_csv(eval_input_path, sep="\t")
                            fn_eval_results_system = os.path.join(eval_out_dir, "eval_results-" + system_codename + ".tsv")
                            if not os.path.exists(fn_eval_results_system):
                                #asr_system = initialize_asr_system(system, model, config_user)
                                df_eval_result = calculate_eval_metrics_per_sample(df_eval_input, dataset, subset, split, system_codename)
                                save_metrics_tsv(df_eval_result, fn_eval_results_system)
                                save_metrics_json(df_eval_result, fn_eval_results_system.replace(".tsv", ".json"))
                            else:
                                print("Skipping calculation of evaluation metrics for ", system_codename, " and ", dataset_codename)
                                df_eval_result = pd.read_csv(fn_eval_results_system, sep="\t")
                            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_result])

        # Save aggregated evaluation metrics for all datasets, subsets, splits, systems and models
        fn_eval_results_agg = os.path.join(eval_out_dir_dataset, "eval_results-all-" + datetime.now().strftime("%Y%m%d"))
        save_metrics_tsv(df_eval_results_all, fn_eval_results_agg + ".tsv")
        save_metrics_json(df_eval_results_all, fn_eval_results_agg + ".json")

        #TODO - calculate WER per audio duration bucket and plot results

def generate_agg_eval_metrics_subsets(config_user, config_common, config_runtime):
    
    # TODO move to config
    script_dir = os.path.dirname(os.path.realpath(__file__))
    eval_in_dir = os.path.join(script_dir, "../../../data/eval_input")
    eval_out_dir_common = os.path.join(script_dir, "../../../data/eval_output/per_dataset")
    datasets, subsets, splits, systems = get_config_run(config_runtime)
    # initialize empty dataframe for storing all evaluation metrics
    df_eval_results_all = pd.DataFrame([])
    # for specified eval configuration, calculate evaluation metrics for each dataset, subset, split, system and model
    #TODO - decide on which level aggregate metrics should be calculated (e.g. for each dataset, subset, split, system, model, version, postnorm, evalnorm, ref_type, eval_type, etc.)
    
    for dataset in datasets:
        eval_out_dir_dataset = os.path.join(eval_out_dir_common, dataset)
        os.makedirs(eval_out_dir_dataset, exist_ok=True)
        for subset in subsets:
            for split in splits:
                dataset_codename = str.join("-", [dataset, subset, split])
                eval_out_dir = os.path.join(eval_out_dir_common, dataset_codename)
                os.makedirs(eval_out_dir, exist_ok=True)
                print("Calculating evaluation metrics for ", dataset_codename, " dataset.")
                print("Output directory: ", eval_out_dir)
                for system in systems:
                    for model in config_runtime["systems"][system]["models"]:
                        for version in config_runtime["systems"][system]["versions"]:
                            # TODO add flag to skipping initializing model, if asr system is needed just to read cache or name (read_model=False)
                            #TODO move to utils
                            system_codename = str.join("_", [system, model])
                            # TODO move eval input dir root to config
                            eval_input_dir = os.path.join(eval_in_dir, system_codename, version, dataset_codename )
                            eval_input_path = os.path.join(eval_input_dir, "eval_input.tsv")    
                            df_eval_input = pd.read_csv(eval_input_path, sep="\t")
                            fn_eval_results_system = os.path.join(eval_out_dir, "eval_results-" + system_codename + ".tsv")
                            if not os.path.exists(fn_eval_results_system):
                                #asr_system = initialize_asr_system(system, model, config_user)
                                df_eval_result = calculate_eval_metrics_per_dataset(df_eval_input, dataset, subset, split, system_codename)
                                save_metrics_tsv(df_eval_result, fn_eval_results_system)
                                save_metrics_json(df_eval_result, fn_eval_results_system.replace(".tsv", ".json"))
                            else:
                                print("Skipping calculation of evaluation metrics for ", system_codename, " and ", dataset_codename)
                                df_eval_result = pd.read_csv(fn_eval_results_system, sep="\t")
                            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_result])

        # Save aggregated evaluation metrics for all datasets, subsets, splits, systems and models
        fn_eval_results_agg = os.path.join(eval_out_dir_dataset, "eval_results-all-" + datetime.now().strftime("%Y%m%d"))
        save_metrics_tsv(df_eval_results_all, fn_eval_results_agg + ".tsv")
        save_metrics_json(df_eval_results_all, fn_eval_results_agg + ".json")

        fn_hf_leaderboard_input = fn_eval_results_agg + "-hf_input.csv"
        calculate_wer_summary(df_eval_results_all, fn_hf_leaderboard_input)
        
        eval_plots_dir= os.path.join(eval_out_dir_dataset, "eval_plots")
        os.makedirs(eval_plots_dir, exist_ok=True)
        generate_plots(df_eval_results_all, eval_plots_dir)

def generate_eval_metrics_all():
    pass

def calculate_wer_summary(data, output_csv_path):
    # TODO - split system and model into separate columns?

    # Pivot the data to have one row per system/model with columns for each dataset's WER
    pivot_data = data.pivot_table(index='system', columns='dataset', values='WER', aggfunc='mean').reset_index()
    
    # change values in the dataset column, so that they are more human readable
    pivot_data.columns = pivot_data.columns.str.replace("test-", "")
    pivot_data.columns = pivot_data.columns.str.replace("valdation-", "")
    pivot_data.columns = pivot_data.columns.str.replace("train-", "")

    # Calculate average WER and weighted average WER with 2 decimal places precision

    data['weighted_wer'] = data['WER'] * data['samples']
    avg_wer = data.groupby('system')['WER'].mean().reset_index(name='avg. WER')
    wavg_wer = data.groupby('system')['weighted_wer'].sum() / data.groupby('system')['samples'].sum()
    wavg_wer = wavg_wer.reset_index(name='wavg. WER')
    # Round the WER values to 2 decimal places
    avg_wer['avg. WER'] = avg_wer['avg. WER'].round(2)
    wavg_wer['wavg. WER'] = wavg_wer['wavg. WER'].round(2)

    # Merge average and weighted average WER with pivot data
    summary_data = pd.merge(pivot_data, avg_wer, on='system')
    summary_data = pd.merge(summary_data, wavg_wer, on='system')
    
    # move columns with avg_wer and wavg_wer to the beginnging (after system and model columns)
    cols = summary_data.columns.tolist()
    cols = cols[:1] + cols[-2:] + cols[1:-2]
    summary_data = summary_data[cols]

    # Rename columns to match the provided CSV format, if necessary
    # This step would require specific column name mappings based on your example CSV
    
    # Save the summarized data to a CSV file
    summary_data.to_csv(output_csv_path, index=False)
    return summary_data

@flow(name="ASR Evaluation Execution Flow")
def asr_eval_run(config_user, config_common, config_runtime):
    generate_agg_eval_metrics_subsets(config_user, config_common, config_runtime)
    generate_sample_eval_metrics_subsets(config_user, config_common, config_runtime)
    
