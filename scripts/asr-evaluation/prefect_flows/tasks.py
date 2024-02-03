from prefect import task
from datasets import load_dataset
from eval_utils.lexical_metrics import get_lexical_metrics
import pandas as pd

@task
def load_config(config_path):
    # Implement your config loading logic
    pass
    print("Loading config from {}".format(config_path))

@task
def gen_hyps_from_audio_samples(audio_paths, asr_system):
    asr_hyps = []
    for audiopath in audio_paths[0:3]:
        print("Processing sample {}".format(audiopath))
        asr_hyp = asr_system.process_audio(audiopath)
        asr_hyps.append(asr_hyp)
    
    asr_system.save_cache()
    return(asr_hyps)

@task
def load_hf_dataset(dataset_name,subset="all"):
    hf_dataset = load_dataset(dataset_name, subset)
    return hf_dataset

def load_hf_dataset_split(dataset_name, split, subset="all"):
    hf_dataset = load_dataset(dataset_name, subset, split=split)
    return hf_dataset

@task
def select_split_of_dataset(dataset, split):
    dataset = dataset[split]
    return dataset

@task
def select_subset_of_dataset(dataset, subset):
    dataset = dataset[subset]
    return dataset

@task
def save_results(results):
    # Implement logic to save results
    pass

@task
def prepare_eval_input_from_hyps_cache(hf_dataset, asr_system) -> pd.DataFrame:
    
    # Implement logic to prepare eval input
    # use audio path columns as index
    audio_paths = hf_dataset["audiopath_local"]

    eval_input_df = pd.DataFrame(audio_paths, columns=["audiopath_local"])
    eval_input_df["system"] = asr_system.get_system()
    eval_input_df["model"] = asr_system.get_model()
    eval_input_df["version"] = asr_system.get_version()
    eval_input_df["codename"] = asr_system.get_codename()

    # TODO add more metadata columns - currently only ref* columns are supported
    # TODO make metdata columns configurable on global, user and runtime levels
    # extract names of columns with references from hf dataset (starting with "ref")
    ref_cols = [col for col in hf_dataset.column_names if col.startswith("ref")]
    print("Column with references found in the HF dataset: ", ref_cols)
    
    # extract references from hf dataset
    for ref_col in ref_cols:
        print("Adding reference column: ", ref_col)
        eval_input_df[ref_col] = hf_dataset[ref_col]
    
    # get hyps from cache
    eval_input_df["hyp_"+ asr_system.get_codename()] = eval_input_df["audiopath_local"].apply(lambda x: asr_system.get_hyp_from_cache(x, asr_system.get_version()))
    return(eval_input_df)

@task
def calculate_eval_metrics(eval_input_df, dataset_codename, system_codename):
    # Implement logic to calculate eval metrics

    # loop over all datasets, subsets, splits, systems, models, versions, postnorms, evalnorms, ref_types, eval_types
    df_eval_results = get_lexical_metrics(eval_input_df, dataset_codename, system_codename, "orig", "all")
    #def calculate_lexical_metrics(df_eval_input, test_set_name,  system_codename, ref_type, norm)->pd.DataFrame:
    return(df_eval_results)

@task
def save_metrics_tsv(df_eval_results):
    # Implement logic to save metrics as TSV
    pass

@task
def save_metrics_json(df_eval_results):
    # Implement logic to save metrics as JSON
    pass