import pandas as pd
import argilla as rg
import os
from pydub import AudioSegment
from huggingface_hub import HfApi


def init_rg_client(HF_TOKEN, ARGILLA_API_KEY, ARGILLA_URL):
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
    settings = rg.Settings.from_json(path)
    return settings

def create_rg_dataset(name, workspace, settings, client):
    dataset = rg.Dataset(
        name=name,
        workspace=workspace,
        settings=settings,
        client=client,
    )
    dataset.create()
    return dataset

def prepare_subset_for_inspection_random(df_results_per_sample_for_specific_system, number_of_samples, norm_method):
    # filter results based on values in "norm_method" column
    df_results_per_sample_for_specific_system = df_results_per_sample_for_specific_system[df_results_per_sample_for_specific_system["norm_type"] == norm_method]

    # get random samples
    df_subset_to_inspect = df_results_per_sample_for_specific_system.sample(n=number_of_samples)
    return df_subset_to_inspect

def prepare_subset_for_inspection_sorted(df_results_per_sample_for_specific_system, number_of_samples, norm_method, worst=True, metric="wer"):
    df_results_per_sample_for_specific_system = df_results_per_sample_for_specific_system[df_results_per_sample_for_specific_system["norm_type"] == norm_method]

    if worst:
        ascending = False
    else:
        ascending = True
    
    df_subset_to_inspect = df_results_per_sample_for_specific_system.sort_values(by=metric, ascending=False).head(number_of_samples)
    return df_subset_to_inspect

def prepare_rg_dataset_for_inspection(df_hf_dataset, df_subset_to_inspect, audio_format_for_inspection, eval_results_audio_in, hf_repo_with_eval_results):
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
    retrieved_dataset = client.datasets(name=rg_dataset_name, workspace=rg_workspace)
    # Retrieve the dataset from the specified workspace
    retrieved_dataset.to_hub(repo_id=hf_dataset_repo_id)
    return retrieved_dataset

def upload_rg_dataset_records(rg_dataset, rg_records):
    rg_dataset.records.log(rg_records)