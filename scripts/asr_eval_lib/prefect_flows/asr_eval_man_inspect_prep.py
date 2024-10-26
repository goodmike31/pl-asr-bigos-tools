from prefect import flow
from prefect_flows.tasks import load_hf_dataset_split
import pandas as pd
from datetime import datetime
from asr_systems import initialize_asr_system
from pathlib import Path
from config_utils import get_config_run
from eval_utils.manual_inspection_utils import init_rg_client, init_rg_dataset_settings, create_rg_dataset, prepare_subset_for_inspection_random, prepare_subset_for_inspection_sorted, prepare_rg_dataset_for_inspection, upload_rg_dataset_records

import os

"""
Input: Results from ASR evaluation - per sample. Contains references and hypotheses, metrics scores and metadata.
Output: Task in argilla system to annotate the evaluation results. If tasks was already created, it will be skipped, unless force=True.
Dependencies: Credetianls and annotation task configuration for Argilla system.
Parameters: Number of samples per system to be annotated. Default is 50.
"""

def generate_manual_inspection_tasks(config_user, config_common, config_runtime, force=False):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets, subsets, splits, systems, eval_run_codename = get_config_run(config_runtime)
    sampling_settings = config_runtime["sampling_settings_for_manual_inspection"]
    norm_types_for_manual_inspection = config_runtime["norm_types_for_manual_inspection"]
    
    audio_format_for_inspection = config_common["audio_format_argilla"]

    manual_inspect_task_version = config_user["ARGILLA_SETTINGS"]["ARGILLA_SETTINGS_MANUAL_INSPECT_VERSION"]
    rg_workspace = config_user["ARGILLA_SETTINGS"]["ARGILLA_WORKSPACE"]

    fp_rg_settings_manual_inspection = os.path.join(script_dir, "../../../config/common/argilla/manual-inspection-task-settings/{}.json".format(manual_inspect_task_version))
    print("fp_rg_settings", fp_rg_settings_manual_inspection)
    
    bigos_eval_repo_name = config_user["PATHS"]["BIGOS_EVAL_DATA_REPO_NAME"]
    bigos_eval_data_dir = config_user["PATHS"]["BIGOS_EVAL_DATA_REPO_PATH"]
    print("bigos_eval_data_dir", bigos_eval_data_dir)
    print("eval_run_codename", eval_run_codename)

    eval_out_dir_common = os.path.join(bigos_eval_data_dir, "eval_output/per_sample/", eval_run_codename)
    manual_inspection_input_dir = os.path.join(bigos_eval_data_dir, "inspection_input", eval_run_codename)
    os.makedirs(manual_inspection_input_dir, exist_ok=True)
    
    # we want to inspect the results of the evaluation for specific systems by taking a random and worst results samples
    # information how the sample was selected is stored in the task metadata
    # each task is a combination of specific system,model, dataset, subset amd split
    # such granularity allows to inspect the results for specific system separately and maintain indepedent versioning of the inspection tasks on HF hub
    df_eval_results_to_inspect = pd.DataFrame([])

    #initialize argilla client
    rg_client = init_rg_client(config_user["CREDENTIALS"]["HF_TOKEN"], config_user["CREDENTIALS"]["ARGILLA_API_KEY"], config_user["ARGILLA_SETTINGS"]["ARGILLA_URL"])
    print("Argilla client initialized!", rg_client)

    rg_man_inspection_settings = init_rg_dataset_settings(fp_rg_settings_manual_inspection)
    print("Argilla dataset settings read!", rg_man_inspection_settings)

    for dataset in datasets:
        eval_out_dir_dataset = os.path.join(eval_out_dir_common, dataset, eval_run_codename)
        os.makedirs(eval_out_dir_dataset, exist_ok=True)
        for split in splits:
            for subset in subsets:
                hf_dataset = load_hf_dataset_split(dataset, subset, split)
                # convert HF dataset to pandas dataframe
                df_hf_dataset = pd.DataFrame(hf_dataset)
                # round all values to 2 decimal places
                df_hf_dataset = df_hf_dataset.round(2)

                dataset_codename = str.join("-", [dataset, subset, split])
                eval_out_dir = os.path.join(eval_out_dir_common, dataset_codename, eval_run_codename)
                print("Preparing manual inspection sample for ", dataset_codename, " dataset.")
                print("Input directory: ", eval_out_dir)
                print("Output directory: ", manual_inspection_input_dir)
                
                for system in systems:
                    for model in config_runtime["systems"][system]["models"]:
                        for version in config_runtime["systems"][system]["versions"]:
                            #TODO - include versioning in the system codename when generating eval results  
                            system_codename = str.join("_", [system, model])
                            system_codename_with_version = str.join("_", [system, model, version])
                            fp_results_per_sample_per_system = os.path.join(eval_out_dir, "eval_results-per_sample-{}.tsv".format(system_codename))
                            df_results_per_sample_for_specific_system = pd.read_csv(fp_results_per_sample_per_system, sep="\t")
                            print("Read df_results_per_sample_for_specific_system: ", fp_results_per_sample_per_system )
                            print(df_results_per_sample_for_specific_system.sample(1))
                            
                            for norm_method in norm_types_for_manual_inspection:
                                for sampling_method in sampling_settings.keys():
                                    
                                    if sampling_method == "random":
                                        df_subset_to_inspect = prepare_subset_for_inspection_random(df_results_per_sample_for_specific_system, sampling_settings[sampling_method], norm_method)
                                    elif sampling_method == "worst":
                                        df_subset_to_inspect = prepare_subset_for_inspection_sorted(df_results_per_sample_for_specific_system, sampling_settings[sampling_method], norm_method)
                                    
                                    print("df_eval_results_to_inspect", df_subset_to_inspect.sample(2))

                                    # prepare argilla dataset for manual inspection
                                    # check if the argilla dataset for system_codename already exists
                                    dataset_codename_no_slash = dataset_codename.replace("/", "-")
                                    rg_dataset_name = "{}_{}_{}".format(system_codename_with_version, dataset_codename_no_slash, eval_run_codename, sampling_method, "norm_"+ norm_method)
                                    
                                    print("rg_dataset_name", rg_dataset_name)
                                    rg_dataset = rg_client.datasets(name=rg_dataset_name, workspace=rg_workspace)

                                    if rg_dataset is None:
                                        print("Creating argilla dataset for ", rg_dataset_name)
                                        rg_dataset = create_rg_dataset(rg_dataset_name, rg_workspace, rg_man_inspection_settings, rg_client)
                                    else:
                                        print("Argilla dataset for ", rg_dataset_name, " already exists!\n\n")
                                    
                                    manual_inspection_input_dir_audio = os.path.join(manual_inspection_input_dir, "audio", dataset_codename)
                                    os.makedirs(manual_inspection_input_dir_audio, exist_ok=True)

                                    # convert the subset of results to inspect into argilla records
                                    rg_records = prepare_rg_dataset_for_inspection(df_hf_dataset, df_subset_to_inspect, audio_format_for_inspection, manual_inspection_input_dir_audio, bigos_eval_repo_name)
                                    print("rg_records", rg_records)

                                    # upload the records to argilla dataset
                                    upload_rg_dataset_records(rg_dataset, rg_records)
                            





@flow(name="ASR Evaluation Results Inspection Preparation Flow")
def asr_eval_man_inspect_prep(config_user, config_common, config_runtime, force):
    generate_manual_inspection_tasks(config_user, config_common, config_runtime, force)
