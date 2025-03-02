from prefect import task
from datasets import load_dataset
from eval_utils.lexical_metrics import get_lexical_metrics_per_dataset, get_lexical_metrics_per_sample
import pandas as pd
import matplotlib.pyplot as plt
import os

@task
def load_config(config_path):
    """
    Load configuration from the specified path.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: Loaded configuration.
    """
    # Implement your config loading logic
    pass
    print("Loading config from {}".format(config_path))

@task
def gen_hyps_from_audio_samples(audio_paths, asr_system, force_hyps):
    """
    Generate ASR hypotheses from audio samples.
    
    Args:
        audio_paths (list): List of paths to audio files.
        asr_system (object): ASR system object with process_audio method.
        force_hyps (bool): Flag to force generation even if hypotheses exist in cache.
    
    Returns:
        list: Generated ASR hypotheses.
    """
    asr_hyps = []
    for audiopath in audio_paths:
        print("Processing sample {}".format(audiopath))
        asr_hyp = asr_system.process_audio(audiopath, force_hyps)
        asr_hyps.append(asr_hyp)
    
    return(asr_hyps)

@task
def load_hf_dataset(dataset_name, subset="all", force_download=False):
    """
    Load a Hugging Face dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        subset (str, optional): Subset of the dataset to load. Defaults to "all".
        force_download (bool, optional): Force download of the dataset. Defaults to False.
    
    Returns:
        Dataset: Loaded Hugging Face dataset.
    """
    hf_dataset = load_dataset(dataset_name, subset)
    return hf_dataset

def load_hf_dataset_split(dataset_name, subset="all", split='test', force_download=False):
    """
    Load a specific split of a Hugging Face dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        subset (str, optional): Subset of the dataset to load. Defaults to "all".
        split (str, optional): Split of the dataset to load. Defaults to 'test'.
        force_download (bool, optional): Force download of the dataset. Defaults to False.
    
    Returns:
        Dataset: Loaded split of the Hugging Face dataset.
    """
    if force_download:
        hf_dataset = load_dataset(dataset_name, subset, split=split, download_mode="force_redownload")
    else:
        hf_dataset = load_dataset(dataset_name, subset, split=split)
    return hf_dataset

@task
def select_split_of_dataset(dataset, split):
    """
    Select a specific split from a loaded dataset.
    
    Args:
        dataset (Dataset): Loaded dataset.
        split (str): Split to select (e.g., 'train', 'test', 'validation').
    
    Returns:
        Dataset: Selected split of the dataset.
    """
    dataset = dataset[split]
    return dataset

@task
def select_subset_of_dataset(dataset, subset):
    """
    Select a subset from a loaded dataset.
    
    Args:
        dataset (Dataset): Loaded dataset.
        subset (str): Subset to select.
    
    Returns:
        Dataset: Selected subset of the dataset.
    """
    dataset = dataset[subset]
    return dataset

@task
def save_results(results):
    """
    Save evaluation results.
    
    Args:
        results (object): Results to save.
    """
    # Implement logic to save results
    pass

@task
def prepare_eval_input_from_hyps_cache(hf_dataset, asr_system, max_samples_per_subset) -> pd.DataFrame:
    """
    Prepare evaluation input dataframe from hypotheses cache.
    
    Args:
        hf_dataset (Dataset): Hugging Face dataset.
        asr_system (object): ASR system object with methods for retrieving system metadata and hypotheses.
        max_samples_per_subset (int): Maximum number of samples to include per subset.
    
    Returns:
        pd.DataFrame: DataFrame containing audio paths, ASR system metadata, references, and hypotheses.
    """
    
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
    print("Eval input DF shape: ", eval_input_df.shape)
    # get only the first max_samples_per_subset samples
    eval_input_df = eval_input_df.head(max_samples_per_subset)
    print("Eval input DF shape after limiting to max_samples_per_subset: ", eval_input_df.shape)
    return(eval_input_df)

@task
def calculate_eval_metrics_per_dataset(eval_input_df, dataset, subset, split, system_codename, ref_types=["orig"], norm_types=["all"], norm_lexicon=None):
    """
    Calculate evaluation metrics for the entire dataset.
    
    Args:
        eval_input_df (pd.DataFrame): Input DataFrame with references and hypotheses.
        dataset (str): Dataset name.
        subset (str): Subset name.
        split (str): Split name.
        system_codename (str): ASR system codename.
        ref_types (list, optional): Types of references to use. Defaults to ["orig"].
        norm_types (list, optional): Types of normalization to apply. Defaults to ["all"].
        norm_lexicon (dict, optional): Normalization lexicon. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for the dataset.
    """
    
    df_eval_results_all= pd.DataFrame()

    for ref_type in ref_types:
        #iterate over normalization methods
        for norm_type in norm_types:
            df_eval_results = get_lexical_metrics_per_dataset(eval_input_df, dataset, subset, split, system_codename, ref_type, norm_type, norm_lexicon)
            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_results])
            
    return(df_eval_results_all)

@task
def calculate_eval_metrics_per_sample(eval_input_df, dataset, subset, split, system_codename, ref_types=["orig"], norm_types=["all"], norm_lexicon=None):
    """
    Calculate evaluation metrics for each sample individually.
    
    Args:
        eval_input_df (pd.DataFrame): Input DataFrame with references and hypotheses.
        dataset (str): Dataset name.
        subset (str): Subset name.
        split (str): Split name.
        system_codename (str): ASR system codename.
        ref_types (list, optional): Types of references to use. Defaults to ["orig"].
        norm_types (list, optional): Types of normalization to apply. Defaults to ["all"].
        norm_lexicon (dict, optional): Normalization lexicon. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each sample.
    """
    # TODO - add support for different normalization methods and ref_types - provide them as input arguments (list or specific value)?
    #iterate over reference types
    df_eval_results_all= pd.DataFrame()
    for ref_type in ref_types:
        #iterate over normalization methods
        for norm_type in norm_types:
            df_eval_results = get_lexical_metrics_per_sample(eval_input_df, dataset, subset, split, system_codename, ref_type, norm_type, norm_lexicon)
            df_eval_results_all = pd.concat([df_eval_results_all, df_eval_results])

    return(df_eval_results_all)


@task
def save_metrics_tsv(df_eval_results, abs_filepath):
    """
    Save evaluation metrics to a TSV file.
    
    Args:
        df_eval_results (pd.DataFrame): DataFrame containing evaluation metrics.
        abs_filepath (str): Absolute path to save the TSV file.
    """
    # Implement logic to save metrics as TSV
    print("Saving metrics to {}".format(abs_filepath))
    df_eval_results.to_csv(abs_filepath, sep="\t", index=False)
    # copy results to the HF repository


@task
def save_metrics_json(df_eval_results, abs_filepath):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        df_eval_results (pd.DataFrame): DataFrame containing evaluation metrics.
        abs_filepath (str): Absolute path to save the JSON file.
    """
    # Implement logic to save metrics as JSON
    print("Saving metrics to {}".format(abs_filepath))
    df_eval_results.to_json(abs_filepath, orient="records")

@task
def check_cached_hyps_size_and_coverage(asr_system, audio_paths):
    """
    Check the size and coverage of cached hypotheses for a set of audio paths.
    
    Args:
        asr_system (object): ASR system object with methods to get cached hypotheses.
        audio_paths (list): List of audio paths to check for in the cache.
    
    Returns:
        tuple: A tuple containing (number of cached hypotheses, 
               number of audio paths found in cache, 
               number of audio paths missing from cache).
    """
    # Implement logic to get number of cached hypotheses
    asr_system_codename = asr_system.get_codename()
    cached_hyps = asr_system.get_cached_hyps()
    # check if audio paths are in cache
    cached_audio_paths = list(cached_hyps.keys())
    nr_of_cached_hyps = len(cached_audio_paths)
    # check common part of cached_audio_paths and audio_paths
    common_audio_paths = list(set(cached_audio_paths) & set(audio_paths))
    nr_of_common_audio_paths = len(common_audio_paths)
    missing_audio_paths = list(set(audio_paths) - set(cached_audio_paths))
    nr_of_missing_audio_paths = len(missing_audio_paths)
    return(nr_of_cached_hyps, nr_of_common_audio_paths, nr_of_missing_audio_paths)


    print("Retrieved number of cached hypotheses for ASR system {} and dataset {} {} {}".format(asr_system_codename, dataset_name, subset, split))

@task
def cached_hyps_stats_to_df(cached_hyps_stats):
    """
    Convert cached hypotheses statistics to a DataFrame.
    
    Args:
        cached_hyps_stats (dict): Dictionary containing statistics of cached hypotheses
                                organized by system and dataset.
    
    Returns:
        pd.DataFrame: DataFrame with columns for System, Dataset, Target_Hypotheses,
                      Common_Hypotheses, Missing_Hypotheses, and Hypothesis_Coverage.
    """
    data_list_new_format = []

    for system, datasets in cached_hyps_stats.items():
        for dataset, metrics in datasets.items():
            target_hypotheses = metrics['nr_of_common_audio_paths'] + metrics['nr_of_missing_audio_paths']
            common_hypotheses = metrics['nr_of_common_audio_paths']
            missing_hypotheses = metrics['nr_of_missing_audio_paths']
            hypothesis_coverage = metrics['hyps_coverage']
            data_list_new_format.append([system, dataset, target_hypotheses, common_hypotheses, missing_hypotheses, hypothesis_coverage])

    df = pd.DataFrame(data_list_new_format, columns=['System', 'Dataset', 'Target_Hypotheses', 'Common_Hypotheses', 'Missing_Hypotheses', 'Hypothesis_Coverage'])

    return(df)
