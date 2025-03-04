import pandas as pd
import argilla as rg
import os
from pydub import AudioSegment
from huggingface_hub import HfApi


def init_rg_client(HF_TOKEN, ARGILLA_API_KEY, ARGILLA_URL):
    """Initialize and configure an Argilla client.
    
    Args:
        HF_TOKEN (str): Hugging Face API token.
        ARGILLA_API_KEY (str): Argilla API key.
        ARGILLA_URL (str): URL endpoint for Argilla service.
    
    Returns:
        rg.Argilla: Configured Argilla client instance.
    """
    print("Initializing Argilla client")
    print("HF_TOKEN", HF_TOKEN)
    print("ARGILLA_API_KEY", ARGILLA_API_KEY)
    print("ARGILLA_URL", ARGILLA_URL)

    rg_client = rg.Argilla(
        api_url=ARGILLA_URL,
        api_key=ARGILLA_API_KEY,
        headers={"Authorization": f"Bearer {HF_TOKEN}"}
    )
    return rg_client

def init_rg_dataset_settings(path):
    """Load Argilla dataset settings from a JSON file.
    
    Args:
        path (str): Path to the JSON file containing settings.
    
    Returns:
        rg.Settings: Argilla dataset settings.
    """
    settings = rg.Settings.from_json(path)
    return settings

def create_rg_dataset(name, workspace, settings, client):
    """Create a new Argilla dataset.
    
    Args:
        name (str): Name of the dataset.
        workspace (str): Workspace in which to create the dataset.
        settings (rg.Settings): Dataset configuration settings.
        client (rg.Argilla): Argilla client instance.
    
    Returns:
        rg.Dataset: The newly created dataset.
    """
    dataset = rg.Dataset(
        name=name,
        workspace=workspace,
        settings=settings,
        client=client,
    )
    dataset.create()
    return dataset

def prepare_subset_for_inspection_random(df_results_per_sample_for_specific_system, number_of_samples, norm_method):
    """Prepare a random subset of samples for inspection based on normalization method.
    
    Args:
        df_results_per_sample_for_specific_system (pd.DataFrame): DataFrame with ASR results.
        number_of_samples (int): Number of samples to select.
        norm_method (str): Normalization method to filter by.
    
    Returns:
        pd.DataFrame: Random subset of samples for inspection.
    """
    # filter results based on values in "norm_method" column
    df_results_per_sample_for_specific_system = df_results_per_sample_for_specific_system[df_results_per_sample_for_specific_system["norm_type"] == norm_method]

    # get random samples
    df_subset_to_inspect = df_results_per_sample_for_specific_system.sample(n=number_of_samples)
    return df_subset_to_inspect

def prepare_subset_for_inspection_sorted(df_results_per_sample_for_specific_system, number_of_samples, norm_method, worst=True, metric="wer"):
    """Prepare a subset of samples sorted by specified metric for inspection.
    
    Args:
        df_results_per_sample_for_specific_system (pd.DataFrame): DataFrame with ASR results.
        number_of_samples (int): Number of samples to select.
        norm_method (str): Normalization method to filter by.
        worst (bool, optional): Whether to select worst (True) or best (False) samples. Defaults to True.
        metric (str, optional): Metric to sort by. Defaults to "wer".
    
    Returns:
        pd.DataFrame: Sorted subset of samples for inspection.
    """
    df_results_per_sample_for_specific_system = df_results_per_sample_for_specific_system[df_results_per_sample_for_specific_system["norm_type"] == norm_method]

    if worst:
        ascending = False
    else:
        ascending = True
    
    df_subset_to_inspect = df_results_per_sample_for_specific_system.sort_values(by=metric, ascending=False).head(number_of_samples)
    return df_subset_to_inspect

def prepare_rg_dataset_for_inspection(df_hf_dataset, df_subset_to_inspect, audio_format_for_inspection, eval_results_audio_in, hf_repo_with_eval_results):
    """Prepare Argilla dataset records for inspection by converting audio files and uploading to HuggingFace.
    
    Args:
        df_hf_dataset (pd.DataFrame): DataFrame with information about audio files in the HuggingFace dataset.
        df_subset_to_inspect (pd.DataFrame): DataFrame with subset of samples to inspect.
        audio_format_for_inspection (str): Target audio format for web playback (e.g., "mp3").
        eval_results_audio_in (str): Directory to store converted audio files.
        hf_repo_with_eval_results (str): HuggingFace repository ID to upload audio files.
    
    Returns:
        list: List of rg.Record objects ready for upload to Argilla dataset.
    """
    rg_records = []
    
    # drop all columns with "NaN" values
    df_subset_to_inspect = df_subset_to_inspect.dropna(axis=1)
    
    for idx, row in df_subset_to_inspect.iterrows():
        audio_file_id = row['id']
        dataset_name = row['dataset']
        subset = row['subset']

        # find the audio file in the HF dataset
        # print the content of column "audiopath_local" from df_hf_dataset where value of 'audiopath_bigos' is equal to audio_file_id
        audio_file_path = df_hf_dataset[df_hf_dataset['audiopath_bigos'] == audio_file_id]['audiopath_local'].values[0]
        
        print(audio_file_path)

        # convert the audio file to mp3
        audio = AudioSegment.from_file(audio_file_path)
        # mp3 filename - replace wav extension with mp3
        audio_web_playback_filename = os.path.basename(audio_file_path).replace("wav", audio_format_for_inspection)
        audio_web_playback_target_dir = os.path.join(eval_results_audio_in, dataset_name, subset)
        os.makedirs(audio_web_playback_target_dir, exist_ok=True)
        audio_web_playback_path = os.path.join(audio_web_playback_target_dir, audio_web_playback_filename)

        audio.export(audio_web_playback_path, format=audio_format_for_inspection, bitrate="192k")
        path_in_repo = os.path.join("inspection_input/audio", dataset_name, subset, audio_web_playback_filename)
        download_link = upload_audio_to_hf(hf_repo_with_eval_results, audio_web_playback_path, path_in_repo)
        html_audio_playback = generate_audio_html(download_link)
        
        # select columns with metadata for the row using metadata_for_manual_inspection
        #row_meta = row[metadata_for_manual_inspection]
        #row_meta = row.filter(metadata_for_manual_inspection)
        row_meta = row.dropna()
        #row_meta = row_meta.reset_index(drop=True)
        print(row_meta)
        # create a dictionary from all columns in the row
        meta_dict = row_meta.to_dict()

        reference = row['ref']
        hypothesis = row['hyp']

        rg_record = rg.Record(
            fields={
                "audio": html_audio_playback,
                "ref": reference,
                "hyp": hypothesis,
            },
            metadata=meta_dict,
        )
        rg_records.append(rg_record)
    return rg_records


def upload_audio_to_hf(dataset_repo_id, audio_file_to_upload, path_in_repo):
    """Upload an audio file to a HuggingFace dataset repository.
    
    Args:
        dataset_repo_id (str): HuggingFace dataset repository ID.
        audio_file_to_upload (str): Local path to the audio file to upload.
        path_in_repo (str): Destination path within the repository.
    
    Returns:
        str: Download URL for the uploaded audio file.
    """
    api = HfApi()
    api.upload_file(
        path_or_fileobj=audio_file_to_upload,
        path_in_repo=path_in_repo,
        repo_id=dataset_repo_id,
        repo_type="dataset",
    )
    download_link = f"https://huggingface.co/datasets/{dataset_repo_id}/resolve/main/{path_in_repo}"
    return download_link

    """# Example usage:
    dataset_repo_id = "michaljunczyk/bigos-eval-results-secret"
    audio_file_to_upload = "/home/michal/Development/hugging-face/michaljunczyk/bigos-eval-results-secret/inspection_input/audio/amu-cai/pl-asr-bigos-v2-secret/pwr-viu-unk/pwr-viu-unk-test-0003-00283.mp3"
    path_in_repo = "inspection_input/audio/amu-cai/pl-asr-bigos-v2-secret/pwr-viu-unk/pwr-viu-unk-test-0003-00283.mp3"

    download_link = upload_audio_to_hf(dataset_repo_id, audio_file_to_upload, path_in_repo)
    print(download_link)"""

def generate_audio_html(audio_filepath):
    """Generate HTML code for embedding audio player.
    
    Args:
        audio_filepath (str): Path or URL to the audio file.
    
    Returns:
        str: HTML code for audio player.
    
    Raises:
        ValueError: If the file extension is not recognized.
    """
    file_extension = audio_filepath.split(".")[-1]
    extension_to_type = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav"
    }
    
    if file_extension not in extension_to_type:
        raise ValueError(f"Unrecognized file extension: {file_extension}")
    
    return f"""
    <audio controls>
        <source src="{audio_filepath}" type="{extension_to_type[file_extension]}">
    </audio>
    """

def upload_rg_dataset_to_hub(client, rg_dataset_name, rg_workspace, hf_dataset_repo_id):
    """Upload Argilla dataset to HuggingFace Hub.
    
    Args:
        client (rg.Argilla): Argilla client instance.
        rg_dataset_name (str): Name of the Argilla dataset.
        rg_workspace (str): Workspace containing the dataset.
        hf_dataset_repo_id (str): HuggingFace repository ID to upload the dataset.
    
    Returns:
        rg.Dataset: The retrieved dataset.
    """
    retrieved_dataset = client.datasets(name=rg_dataset_name, workspace=rg_workspace)
    # Retrieve the dataset from the specified workspace
    retrieved_dataset.to_hub(repo_id=hf_dataset_repo_id)
    return retrieved_dataset

def upload_rg_dataset_records(rg_dataset, rg_records):
    """Upload records to an Argilla dataset.
    
    Args:
        rg_dataset (rg.Dataset): Argilla dataset to upload records to.
        rg_records (list): List of rg.Record objects to upload.
    """
    rg_dataset.records.log(rg_records)