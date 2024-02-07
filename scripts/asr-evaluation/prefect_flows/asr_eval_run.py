from prefect import flow
from prefect_flows.tasks import calculate_eval_metrics, save_metrics_tsv, save_metrics_json 
import pandas as pd
from datetime import datetime
from asr_systems import initialize_asr_system
from pathlib import Path
import os

def get_config_run(config_runtime)->list:

    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    return datasets, subsets, splits, systems
                            
def generate_eval_metrics_subsets(config_user, config_common, config_runtime):
    
    # TODO move to config
    script_dir = os.path.dirname(os.path.realpath(__file__))
    eval_in_dir = os.path.join(script_dir, "../../../data/eval_input")
    datasets, subsets, splits, systems = get_config_run(config_runtime)
    # initialize empty dataframe for storing all evaluation metrics
    df_eval_results_all = pd.DataFrame([])
    # for specified eval configuration, calculate evaluation metrics for each dataset, subset, split, system and model
    #TODO - decide on which level aggregate metrics should be calculated (e.g. for each dataset, subset, split, system, model, version, postnorm, evalnorm, ref_type, eval_type, etc.)
    
    for dataset in datasets:
        for subset in subsets:
            for split in splits:
                dataset_codename = str.join("-", [dataset, subset, split])
                eval_out_dir = os.path.join(script_dir, "../../../data/eval_output", dataset_codename)
                os.makedirs(eval_out_dir, exist_ok=True)
                print("Calculating evaluation metrics for ", dataset_codename, " dataset.")
                print("Output directory: ", eval_out_dir)
                for system in systems:
                    for model in config_runtime["systems"][system]["models"]:
                        for version in config_runtime["systems"][system]["versions"]:
                            # TODO add flag to skipping initializing model, if asr system is needed just to read cache or name (read_model=False)
                            asr_system = initialize_asr_system(system, model, config_user) 
                            system_codename = asr_system.get_codename()
                            # TODO move eval input dir root to config
                            eval_input_dir = os.path.join(eval_in_dir, system_codename, version, dataset_codename )
                            eval_input_path = os.path.join(eval_input_dir, "eval_input.tsv")    
                            df_eval_input = pd.read_csv(eval_input_path, sep="\t")
                            filename_out = os.path.join(eval_out_dir, "eval_results-" + system_codename + ".tsv")
                            if not os.path.exists(filename_out):
                                df_eval_result = calculate_eval_metrics(df_eval_input, dataset_codename, system_codename)
                                save_metrics_tsv(df_eval_result, filename_out)
                                save_metrics_json(df_eval_result, filename_out.replace(".tsv", ".json"))
                            else:
                                print("Skipping calculation of evaluation metrics for ", system_codename, " and ", dataset_codename)
                                df_eval_result = pd.read_csv(filename_out, sep="\t")
                            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_result])
        filename_out_agg = os.path.join(eval_out_dir, "eval_results-all-" + datetime.now().strftime("%Y%m%d"))
        save_metrics_tsv(df_eval_results_all, filename_out_agg + ".tsv")
        save_metrics_json(df_eval_results_all, filename_out_agg + ".json")
        calculate_wer_summary(df_eval_results_all, dataset, filename_out_agg + "-hf_input.csv")
        print(df_eval_results_all)

def generate_eval_metrics_all():
    pass

def calculate_wer_summary(data, dataset, output_csv_path):
    # TODO - split system and model into separate columns?
    
    # Pivot the data to have one row per system/model with columns for each dataset's WER
    pivot_data = data.pivot_table(index='system', columns='dataset', values='WER', aggfunc='mean').reset_index()
    
    # change values in the dataset column, so that they are more human readable
    pivot_data.columns = pivot_data.columns.str.replace("test-", "")
    pivot_data.columns = pivot_data.columns.str.replace("   valdation-", "")
    pivot_data.columns = pivot_data.columns.str.replace("train-", "")
    pivot_data.columns = pivot_data.columns.str.replace(dataset, "")

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

@flow(name="ASR Evaluation Execution Flow")
def asr_eval_run(config_user, config_common, config_runtime):
    generate_eval_metrics_subsets(config_user, config_common, config_runtime)
