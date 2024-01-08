import json
import os
from datasets import load_dataset, Audio
from huggingface.const import BIGOS_SUBSETS, BIGOS_SPLITS
from nemo.const import NEMO_MANIFEST_DIR, HF_CACHE_DIR

def hf_to_nemo_manifest(hf_dataset)->dict:  
    """
    Converts a Hugging Face dataset to a NeMo manifest.
    Parameters:
    -----------
    hf_dataset: Hugging Face dataset
        Single subset and split of HF dataset to be converted.
    """
    manifest = []
    sampling_rate=hf_dataset.features['audio'].sampling_rate
    print("sampling_rate: ", sampling_rate)
          
    for sample in hf_dataset:
        # if duration is not provided, calculate it from the audio array
        print(sample)
        # check if the key "duration" exists in "sample" dict
        if 'duration' not in sample.keys():
            duration = len(sample['audio']['array']) / sampling_rate
        else:
            duration = sample['duration']
        
        # get path to local audio file
        audio_path = sample['audiopath_local']

        # append sample to manifest
        manifest.append({
            'audio_filepath': audio_path,
            'duration': duration,
            'text': sample['ref_orig'],
        })
    return manifest

def save_manifest_as_jsonl(manifest_dict, target_file):
    with open(target_file, 'w') as f:
        for sample in manifest_dict:
            json.dump(sample, f)
            f.write('\n')


def convert_hf_to_nemo(subset, split, target_dir=None):
    """
    Converts a Hugging Face dataset to a NeMo manifest.
    Parameters:
    -----------
    subset: str
        Name of a subset taken from BIGOS_SUBSETS.
    split: str
        Name of a split taken from BIGOS_SPLITS.
    """
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    dataset = load_dataset('amu-cai/pl-asr-bigos-v2', subset, split=split, cache_dir=HF_CACHE_DIR)
    #print(dataset)  
    manifest = hf_to_nemo_manifest(dataset)

    os.makedirs(os.path.join(target_dir, subset), exist_ok=True)
    target_file = os.path.join(target_dir, subset, split + '.jsonl')
    print("Saving manifest as: ", target_file)
    
    save_manifest_as_jsonl(manifest, target_file)

def main():
    #subsets = BIGOS_SUBSETS
    #splits = BIGOS_SPLITS
    splits=['validation']
    subsets=['pwr-viu-unk']
    for subset in subsets:
        for split in splits:
            print("Converting subset: ", subset, " split: ", split)
            convert_hf_to_nemo(subset, split, NEMO_MANIFEST_DIR)

main()