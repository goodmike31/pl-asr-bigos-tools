from prefect import task
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from utils import download_tsv_from_google_sheet, get_meta_header_tts
import glob
import os
import tarfile

@task
def load_config(config_path):
    # Implement your config loading logic
    pass
    print("Loading config from {}".format(config_path))

@task 
def read_prompts(promptset_source, promptset_type, sample_prompts = True, sample_size = 10, sample_type = "head"):
    if (promptset_type == "google_sheet"):
        # download prompts from google sheet
        df_prompts = download_tsv_from_google_sheet(promptset_source)

    # validate prompts 
    #TODO
    if (sample_prompts):
        print("Sampling prompts")
        if (sample_type == "random"):
            print("Sampling random prompts")
            df_prompts = df_prompts.sample(sample_size)
        elif (sample_type == "head"):
            print("Sampling head prompts")
            df_prompts = df_prompts.head(sample_size)
    else:
        print("Not sampling prompts")   
     
    return(df_prompts)

@task
def generate_speech_and_meta_for_tts_voice(df_prompts, tts_system, out_dir_meta, subset, split, speaker_id):
    print(f"Generating speech for tts: {tts_system}")

    tts_engine = tts_system.get_system()
    tts_voice = tts_system.get_voice()

    sampling_rate = tts_system.get_sampling_rate()

    df_header = get_meta_header_tts()
    out_df_spk = pd.DataFrame([], columns=df_header)

    # check if the prompt set is consistent and match the subset
    for index, row in df_prompts.iterrows():
        prompt_set = row['prompt_set_id']
        assert(prompt_set == subset)
    
    # if OK, process the prompts
    print("Processing prompt set: ", prompt_set)

    os.makedirs(out_dir_meta, exist_ok=True)

    out_dir_meta_spk = os.path.join(out_dir_meta, speaker_id)
    os.makedirs(out_dir_meta_spk, exist_ok=True)

    for index, row in df_prompts.iterrows():
        
        # prompt_id is concatenated from promptset_id and prompt_index making it unique across multiple promptsets
        prompt_id = row['prompt_id']
        
        # index is used as the audio file id
        prompt_index = row['prompt_index']
        
        # pad audio_file_id to 5 digits
        audio_file_id = str(prompt_index).zfill(5)

        prompt_text = row['prompt']

        # prepare output in BIGOS format
        audioname = str.join("-",[prompt_set, split, speaker_id, audio_file_id])

        audiopath_bigos = "{}.wav".format(audioname)

        out_fp = os.path.join(out_dir_meta_spk, audiopath_bigos)

        if (os.path.exists(out_fp)):
            print("File already exists, skipping!\n", out_fp)
        else:
            tts_system.gen_audio(out_fp, prompt_text)
        
        df_row = pd.DataFrame([[audioname, split, subset, speaker_id, sampling_rate, sampling_rate, prompt_text, audiopath_bigos, prompt_id, tts_engine, tts_voice]], columns=df_header)
        out_df_spk = pd.concat([out_df_spk, df_row], axis=0)

    # save the results for the speaker
    out_fp_spk = os.path.join(out_dir_meta, f"{speaker_id}.tsv")
    out_df_spk.to_csv(out_fp_spk, sep='\t', index=False)
    print("Saved meta for system {} and voice: {}".format(tts_system, tts_voice), out_fp_spk)
    print("Done!")
    return(out_df_spk)

@task
def prepare_hf_release(df_subset_meta, source_dir, target_dir):
    print("Preparing HF release")
    split = os.path.basename(source_dir) # split name is the name of the directory

     # create TSV file for all speakers
    subset_meta_fp = os.path.join(target_dir, f"{split}.tsv")
    df_subset_meta.to_csv(subset_meta_fp, sep='\t', index=False)
    print("Saved subset meta for split {} results to: ".format(split), subset_meta_fp)

    # generate archive_name based on audio_dir
    archive_name = os.path.join(target_dir, split + ".tar.gz")

    # Search for all audio files ending with ".wav"
    audio_files = glob.glob(os.path.join(source_dir,"*/*.wav"))

    with tarfile.open(archive_name, "w:gz") as tar:
        # Add each audio file to the archive without preserving the directory structure
        for audio_file in audio_files:
            tar.add(audio_file, arcname=os.path.basename(audio_file))

    print("Audio files added to", archive_name)
        

@task
def upload_subset_to_hf(subset, subset_dir_hf_release, bigos_hf_repo, hf_repo_url, overwrite=False, secret_repo=False):
    # current directory
    print("Current directory: ", os.getcwd())
    cur_dir = os.getcwd()
    if not subset:
        print("Please provide subset name as the first argument")
        return
    if not subset_dir_hf_release:
        print("Please provide subset directory as the second argument")
        return
    if not bigos_hf_repo:
        print("Please provide path to HF subset repository as the third argument")
        return
    if not hf_repo_url:
        print("Please provide HF subset repository URL as the fourth argument")
        return

    if not secret_repo:
        secret_repo = False
        print("Secret is not provided, using default value FALSE")

    # Copy files for the subset to be updated
    os.makedirs(os.path.join(bigos_hf_repo, 'data', subset), exist_ok=True)

    # If secret repo, copy only test tsv file
    # if already exists in the repo, skip
    if secret_repo:
        if overwrite == False & os.path.exists(os.path.join(bigos_hf_repo, 'data', subset, f'{subset}_test.tsv')):
            print(f"TSV file for {subset} already exists in the repository, skipping...")
        else:
            shutil.copy2(os.path.join(subset_dir_hf_release, f'{subset}_test.tsv'), os.path.join(bigos_hf_repo, 'data', subset), force=True)
        if overwrite == False & os.path.exists(os.path.join(bigos_hf_repo, 'data', subset, 'test.tar.gz')):
            print(f"Tar file for {subset} already exists in the repository, skipping...")
        else:
            shutil.copy2(os.path.join(subset_dir_hf_release, 'test.tar.gz'), os.path.join(bigos_hf_repo, 'data', subset), force=True)


    if not secret_repo:
        # If not secret repo, copy all files
        for file in os.listdir(subset_dir_hf_release):
            if file.endswith('.tsv'):
                shutil.copy2(os.path.join(subset_dir_hf_release, file), os.path.join(bigos_hf_repo, 'data', subset))
            elif file.endswith('.tar.gz'):
                shutil.copy2(os.path.join(subset_dir_hf_release, file), os.path.join(bigos_hf_repo, 'data', subset))

    # Change working directory to HF subset repository
    os.chdir(bigos_hf_repo)
    os.system('git add .')
    os.system('git commit -m "Update subset"') 
    os.system('git push')
    os.chdir(cur_dir)
