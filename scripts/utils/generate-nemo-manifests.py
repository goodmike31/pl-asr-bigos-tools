"""
Generate NeMo-compatible manifest files from HuggingFace datasets.

This script converts HuggingFace datasets into NeMo manifest files in JSONL format.
It can convert all datasets, subsets, and splits defined in the common configuration file
or specific ones provided as command-line arguments.
"""
import json
import os
import sys
from datasets import load_dataset
import argparse
import configparser

def hf_to_nemo_manifest(hf_dataset)->dict:  
    """
    Converts a Hugging Face dataset to a NeMo manifest.
    
    Parameters:
    -----------
    hf_dataset: Hugging Face dataset
        Single subset and split of HF dataset to be converted.
    
    Returns:
    --------
    list:
        List of dictionaries, where each dictionary represents a sample in the NeMo manifest format
        with keys: 'audio_filepath', 'duration', 'text', 'audioname', and 'speaker_id'.
    """
    manifest = []
    sampling_rate=hf_dataset.features['audio'].sampling_rate
          
    for sample in hf_dataset:
        # if duration is not provided, calculate it from the audio array
        # check if the key "duration" exists in "sample" dict
        if 'duration' not in sample.keys():
            duration = len(sample['audio']['array']) / sampling_rate
        else:
            duration = sample['duration']
        
        # get path to local audio file
        audio_path = sample['audiopath_local']

        # get filename
        audioname=sample['audioname']

        # get speaker_id
        speaker_id=sample['speaker_id']

        # append sample to manifest
        manifest.append({
            'audio_filepath': audio_path,
            'duration': duration,
            'text': sample['ref_orig'],
            'audioname': audioname,
            "speaker_id": speaker_id,
        })
    return manifest

def save_manifest_as_jsonl(manifest_dict, target_file):
    """
    Saves a manifest dictionary as a JSONL file.
    
    Parameters:
    -----------
    manifest_dict: list
        List of dictionaries representing the NeMo manifest.
    target_file: str
        Path where the JSONL manifest will be saved.
    """
    with open(target_file, 'w') as f:
        for sample in manifest_dict:
            json.dump(sample, f)
            f.write('\n')


def convert_hf_to_nemo(dataset_id, subset, split, target_dir=None):
    """
    Converts a Hugging Face dataset to a NeMo manifest and saves it as a JSONL file.
    
    Parameters:
    -----------
    dataset_id: str
        Identifier of the Hugging Face dataset to convert.
    subset: str
        Name of a subset taken from subsets.
    split: str
        Name of a split taken from BIGOS_SPLITS.
    target_dir: str, optional
        Directory where the manifest will be saved. If None, uses current directory.
    
    Returns:
    --------
    None
        The function saves the manifest to disk and doesn't return any value.
    """
    target_file = os.path.join(target_dir, subset + "-" + split + '.jsonl')
    
    if os.path.exists(target_file):
        print("Manifest already exists: ", target_file)
        return

    hf_dataset = load_dataset(dataset_id, subset, split=split, trust_remote_code=True)
    #print(dataset)  
    manifest = hf_to_nemo_manifest(hf_dataset)

    print("Saving manifest as: ", target_file)
    
    save_manifest_as_jsonl(manifest, target_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Converts Hugging Face datasets to NeMo manifests')
    parser.add_argument('--dataset', type=str, help='Split to convert. Default = all datasets from the common config file', default='all')
    parser.add_argument('--subset', type=str, help='Subset to convert. Default = all splits from the common config file',default='all')
    parser.add_argument('--split', type=str, help='Split to convert. Default = all splits from the common config file',default='all')
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    common_config_path_default=os.path.join(script_dir, '../config/common/config.json')
    parser.add_argument('--common_config_file', type=str, help='JSON file with common configuration', default=common_config_path_default)
    user_config_path_default = os.path.join(script_dir, '../config/user-specific/config.ini')
    parser.add_argument('--user_config_file', type=str, help='INI file with user specific configuration', default=user_config_path_default)

    args = parser.parse_args()


    if args.dataset and args.subset and args.split and args.common_config_file and args.user_config_file:
        # load common config
        if not os.path.exists(args.common_config_file):
            print("Common config file does not exist: ", args.common_config_file)
            sys.exit(1)
        else:
            config_common = json.load(open(args.common_config_file))
            
        # load user config
        if not os.path.exists(args.user_config_file):
            print("User config file does not exist: ", args.user_config_file)
            sys.exit(1)
        else:
            config_user = configparser.ConfigParser()
            config_user.read(args.user_config_file)
            
            BIGOS_REPO_DIR = config_user.get("PATHS", "BIGOS_REPO_DIR")
            print("BIGOS_REPO_DIR: ", BIGOS_REPO_DIR)
            
            NEMO_MANIFEST_DIR = config_user.get('PATHS','NEMO_MANIFEST_DIR')
            print("NEMO_MANIFEST_DIR: ", NEMO_MANIFEST_DIR)

        if args.dataset == 'all':
            datasets = config_common["BIGOS_CORPORA"]
            print("Converting all datasets: ", datasets)
            for dataset_id in datasets:
                print("Converting dataset: ", dataset_id)
                subsets = config_common[dataset_id]["SUBSETS"]
                splits = config_common[dataset_id]["SPLITS"]
                for subset in subsets:
                    for split in splits:
                        print("Converting subset: ", subset, " split: ", split)
                        convert_hf_to_nemo(dataset_id, subset, split, NEMO_MANIFEST_DIR)
    else:
        print("Please specify --input_file, --config_file, --asr_hyps_cache_file, --col_name_audiopath and --output_file arguments")
        sys.exit(1)