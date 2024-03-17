from prefect import flow
from prefect_flows.tasks import generate_speech_and_meta_for_tts_voice, read_prompts, prepare_hf_release, upload_subset_to_hf
from tts_systems import initialize_tts_system
from utils import get_meta_header_tts
import pandas as pd
import os

@flow(name="TTS Evaluation Data Generation Flow")
def tts_gen(config_user, config_common, config_runtime_tts):

    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    tts_data_dir = os.path.join(script_dir, "../../../data/tts_set")
    print("tts_data_dir", tts_data_dir)

    tts_hf_rel_dir = os.path.join(script_dir, "../../../data/tts_set_hf")
    print("tts_hf_rel_dir", tts_hf_rel_dir)


    dataset_name = config_runtime_tts["dataset_name"]
    splits = config_runtime_tts["splits"]

    dataset_hf_repo_local = os.path.join(config_user["PATHS"]["HF_REPO_ROOT"], dataset_name)
    print("dataset_hf_repo_local", dataset_hf_repo_local)
    dataset_hf_repo_url = "https://huggingface.co/datasets/" + dataset_name
    print("dataset_hf_repo_url", dataset_hf_repo_url)

    subsets_and_promptsets = config_runtime_tts["subsets_and_promptsets"]
    subsets = subsets_and_promptsets.keys()
    tts_engines_and_settings = config_runtime_tts["tts_engines_and_settings"]
    tts_engines = tts_engines_and_settings.keys()

    for subset in subsets:
        for split in splits:
            out_df_split = pd.DataFrame([], columns=get_meta_header_tts())
            print("Generating TTS dataset: {} for subset {} and split {}".format(dataset_name, subset, split))
            dir_tts_subset = os.path.join(tts_data_dir, dataset_name, subset, split)
            os.makedirs(dir_tts_subset, exist_ok=True)

            dir_tts_hf = os.path.join(tts_hf_rel_dir, dataset_name, subset, split)
            os.makedirs(dir_tts_hf, exist_ok=True)

            # read prompts for the subset
            promptset_source = subsets_and_promptsets[subset]["promptset_source"]
            print("promptset_source", promptset_source)
            promptset_type = subsets_and_promptsets[subset]["promptset_type"]
            print("promptset_type", promptset_type)
            sample_prompts = subsets_and_promptsets[subset]["sample_prompts"]
            print("sample_prompts", sample_prompts)

            if(sample_prompts == "True"):
                sample_prompts = True
            else:
                sample_prompts = False

            print("Reading prompts for subset: ", subset)

            if (sample_prompts):
                sample_size = int(subsets_and_promptsets[subset]["sample_size"])
                sample_type = subsets_and_promptsets[subset]["sample_type"]

                print("sample_size", sample_size)
                print("sample_type", sample_type)

                df_prompts = read_prompts(promptset_source, promptset_type, True, sample_size, sample_type)
            else:
                df_prompts = read_prompts(promptset_source, promptset_type, False)

            print("Promptset length: ", len(df_prompts))

            spk_index = 1
            for tts_engine in tts_engines:
                for tts_voice in tts_engines_and_settings[tts_engine]["voices"]:
                    # pad spk_index to 5 digits 
                    # each engine voice combination will have globally unique, but not fixed speaker_id
                    speaker_id = str(spk_index).zfill(4)
                    tts_system = initialize_tts_system(tts_engine, tts_voice, config_user)
                    print("Generating TTS recordings with engine {} and voice {}".format(tts_engine, tts_voice))

                    out_dir_meta = os.path.join(dir_tts_subset)
                    out_df_spk = generate_speech_and_meta_for_tts_voice(df_prompts, tts_system, out_dir_meta, subset, split, speaker_id)
                    # TODO - introduce fixed mapping of tts voices to speaker_id
                    spk_index += 1
                    out_df_split = pd.concat([out_df_split, out_df_spk], axis=0)

        print("Out DF split len: ", len(out_df_split))
        # Release and upload subset to HF
        prepare_hf_release(out_df_split, dir_tts_subset, dir_tts_hf)
        upload_subset_to_hf(subset, dir_tts_hf, dataset_hf_repo_local, dataset_hf_repo_url, overwrite=False, secret_repo=False)
        