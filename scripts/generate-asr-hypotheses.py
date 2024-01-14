import sys
import argparse
import configparser
import os
import openai
import pandas as pd
import time
import json
import datasets

#import azure.cognitiveservices.speech as speechsdk
#from google.cloud import speech_v1p1beta1 as speech
#from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig

def set_asr_systems_configuration(config_user):
    # Google config
    google_model_types_str = config_user.get("CLOUD_ASR_SETTINGS", "GOOGLE_MODELS")
    google_model_types = google_model_types_str.split(", ")

    # Azure config
    azure_region = config_user.get("CLOUD_ASR_SETTINGS", "AZURE_REGION")
    azure_model_types_str = config_user.get("CLOUD_ASR_SETTINGS", "AZURE_MODELS")
    azure_model_types = azure_model_types_str.split(", ")

    # Set up Google Cloud API key
    google_api_key_path = config_user.get("API_KEYS", "GOOGLE_API_KEY_FILE")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_api_key_path

    # Set up Azure API key and region
    azure_api_key = config_user.get("API_KEYS", "AZURE_API_KEY")

    force_hyps_gen_azure = config_user.get("CLOUD_ASR_SETTINGS", "FORCE_HYPS_GEN_AZURE")
    force_hyps_gen_google = config_user.get("CLOUD_ASR_SETTINGS", "FORCE_HYPS_GEN_GOOGLE")
    force_hyps_gen_whisper_cloud = config_user.get("CLOUD_ASR_SETTINGS", "FORCE_HYPS_GEN_WHISPER_CLOUD")

    # Set up whisper_cloud ASR API key
    whisper_cloud_api_key = config_user.get("API_KEYS", "WHISPER_API_KEY")
    openai.api_key = whisper_cloud_api_key

def generate_asr_hypotheses(args, config_user, config_common):
    if (args.input_dataset == "all"):
        datasets = config_common["BIGOS_CORPORA"]
    else:
        datasets = [args.input_dataset]

    if (args.splits == "all"):
        splits = config_common["SPLITS"]
    else:
        splits = [args.splits]

    asr_hyp_cache_dir = args.asr_hyps_cache_dir
    asr_hyp_out_dir = args.output_dir

    # TODO: add custom path to audio files
    col_name_audiopath="audiopath_local"

    # load HF dataset
    for input_dataset in datasets:
        hf_dataset = datasets.load_dataset(input_dataset)

        for split in splits:
            print(f"Generating hypotheses for {input_dataset} {split}")
            # get paths to audio files
            print ("Retrieving audio from column: " + col_name_audiopath)
            audio_paths = hf_dataset[split][col_name_audiopath]

            
            transcribe_whisper_cloud(audio_paths, config_user)
            transcribe_azure(audio_paths, config_user)
            transcribe_google(audio_paths, config_user)
  
            # Save the updated DataFrame to a new TSV file
            df_input.to_csv(output_file, sep="\t", index=False)
            print(f"Results saved to {output_file}")

def get_hyp_from_cache(df_cache, audio_fp_local, col_name_asr_hyp):
                    #TODO - refactor
                    if os.path.exists(cache_file):
        print("cache_file: " + cache_file)
        cache_file_exists=True
        df_cache =  pd.read_csv(cache_file, sep="\t", index_col=col_name_audiopath)
        print(df_cache)
    else:
        print("The cache file does not exist")
        cache_file_exists=False


    print(f"Retrieving ASR hyp from cache for engine {col_name_asr_hyp} and {audio_fp_local}")
    try:
        cached_hyp = df_cache.at[audio_fp_local, col_name_asr_hyp]
        print(f"Cached transcription:{cached_hyp}")
        return(cached_hyp)
    except Exception as e:
        print("Error in retrieval from cache e.g. hyp not available in cache")
        return("")
    
def transcribe_google(audio_fp, col_name_asr_hyp):
    # check asr systems to generate hypotheses
    google_model_types_str = config_user.get("CLOUD_ASR_SETTINGS", "GOOGLE_MODELS")
    google_model_types = google_model_types_str.split(", ")
    
    for audio_path in audio_paths:
    print("Generating ASR hyps for: " + audio_path)
    #google
            for model_type in google_model_types:
                print("Generating hypotheses for Google "+ model_type)
                google_transcriptions = []
                col_name_asr_hyp = "hyp_google_" + model_type

                for _, row in df_input.iterrows():
                    audio_fp_local=row[col_name_audiopath]
                    audio_fp = os.path.join(bigos_repo_root_dir, audio_fp_local)
                    if(force_hyps_gen_google == "true" or cache_file_exists==False):
                        google_transcription = transcribe_google(audio_fp,col_name_asr_hyp)
                    else:
                        google_transcription_cached = get_hyp_from_cache(df_cache, audio_fp_local, col_name_asr_hyp)
                        if(google_transcription_cached == ""):
                            google_transcription = transcribe_google(audio_fp,col_name_asr_hyp)
                        else:
                            google_transcription=google_transcription_cached
                    google_transcriptions.append(google_transcription)

                df_input[col_name_asr_hyp] = google_transcriptions

    print(f"Generating transcription for: {col_name_asr_hyp}\n{audio_fp}")
    try:
        client = speech.SpeechClient()
        with open(audio_fp, "rb") as f:
            content = f.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(language_code=lang_code, enable_automatic_punctuation=True)
        response = client.recognize(config=config, audio=audio)
        result = response.results[0].alternatives[0].transcript
    except IndexError:
        result = "Index error"
    except Exception as e:
        print(f"Other error: {e}")
        result = "Other error"
    return result

    
def transcribe_azure(audio_fp, col_name_asr_hyp):
    azure_model_types_str = config_user.get("CLOUD_ASR_SETTINGS", "AZURE_MODELS")
    azure_model_types = azure_model_types_str.split(", ")
     #azure
            for model_type in azure_model_types:
                print("Generating hypotheses for Azure "+ model_type)
                azure_transcriptions = []
                col_name_asr_hyp = "hyp_azure_" + model_type
                
                for _, row in df_input.iterrows():
                    audio_fp_local=row[col_name_audiopath]
                    audio_fp = os.path.join(bigos_repo_root_dir, audio_fp_local)
                    print(f"Generating transcription for: {audio_fp} and {col_name_asr_hyp}")
                    if(force_hyps_gen_azure == "true" or cache_file_exists==False):
                        azure_transcription = transcribe_azure(audio_fp, col_name_asr_hyp)
                    else:
                        azure_transcription_cached = get_hyp_from_cache(df_cache, audio_fp_local, col_name_asr_hyp)
                        if(azure_transcription_cached==""):
                            azure_transcription = transcribe_azure(audio_fp,col_name_asr_hyp)
                        else:
                            azure_transcription=azure_transcription_cached
                    azure_transcriptions.append(azure_transcription)

                df_input[col_name_asr_hyp] = azure_transcriptions

    print(f"Generating transcription for: {col_name_asr_hyp}\n{audio_fp}")
    try:
        speech_config = speechsdk.SpeechConfig(subscription=azure_api_key, region=azure_region)
        speech_config.speech_recognition_language=lang_code
        audio_config = speechsdk.AudioConfig(filename=audio_fp)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Azure: {}".format(result.text))
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(result.no_match_details))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")        
    except IndexError:
        result = "Index error"
    except Exception as e:
        print(f"Other error: {e}")
    return(result.text)

def transcribe_whisper_cloud(audio_fp, config_user)->DataFrame:

    whisper_cloud_model_types_str = config_user.get("CLOUD_ASR_SETTINGS", "WHISPER_CLOUD_MODELS")
    whisper_cloud_model_types = whisper_cloud_model_types_str.split(", ")

                #whisper cloud
            print("Generating hypotheses for Whisper cloud")
            whisper_transcriptions= []
            col_name_asr_hyp="hyp_whisper_cloud"
            for _, row in df_input.iterrows():
                audio_fp_local=row[col_name_audiopath]
                audio_fp = os.path.join(bigos_repo_root_dir, audio_fp_local)
                if(force_hyps_gen_whisper == "true" or cache_file_exists==False):
                    whisper_transcription = transcribe_whisper(audio_fp, col_name_asr_hyp)
                else:
                    whisper_transcription_cached = get_hyp_from_cache(df_cache, audio_fp_local, col_name_asr_hyp)
                    if(whisper_transcription_cached==""):
                        whisper_transcription = transcribe_whisper(audio_fp, col_name_asr_hyp)
                    else:
                        whisper_transcription = whisper_transcription_cached
                whisper_transcriptions.append(whisper_transcription)

            df_input[col_name_asr_hyp] = whisper_transcriptions

       
    print(f"Generating transcription for: {col_name_asr_hyp}\n{audio_fp}")
    try:
        audio_data = open(audio_fp, "rb")
        # check if request is valid and what version is used
        response = openai.Audio.transcribe("whisper-1", audio_data)
        result=response.text
        time.sleep(1)
    except IndexError:
        result ="Index error"
    except openai.error.InvalidRequestError as e:
        if "Audio file is too short" in str(e):
            print(f"Error: {e}")
            result = "Invalid audio file error  "
    except Exception as e:
        print(f"Other error: {e}")
        result = "Other error"
    return result

def transcribe_whisper_local(audio_fp, col_name_asr_hyp):
    #whisper local_model_types_str = config_user.get("LOCAL_ASR_SETTINGS", "WHISPER_LOCAL_MODELS")
    #whisper_local_model_types = whisper_local_model_types_str.split(", ")


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='Reads TSV with speech recordings and generates hypotheses with local whisper ASR')

    # Default location of config files.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    common_config_path = os.path.join(script_dir, '../config/common/config.json')
    user_config_path = os.path.join(script_dir, '../config/user-specific/config.ini')

    # read config files
    config_user = configparser.ConfigParser()
    config_user.read(user_config_path)
    config_common = json.load(open(common_config_path))

    # get root path to repo
    bigos_repo_root_dir = config_user.get("PATHS", "BIGOS_REPO_DIR")

    # Optional CLI arguments. If not specified, the default values are used.
    # User can specify custom location of output files or cache.
    parser.add_argument('--output_dir', type=str, help='Custom directory to save results to', default=os.path.join(bigos_repo_root_dir, "data/asr_hypotheses"))
    parser.add_argument('--asr_hyps_cache_dir', type=str, help='Custom location cache files storing already generated ASR transcriptions',
                        default=os.path.join(bigos_repo_root_dir, "data/asr_hyps_cache"))

    # User can specify custom dataset to process. HF format required "account/dataset_name"
    # Add missing import statements here
    parser.add_argument('--input_dataset', type=str, help='Custom dataset to read audio paths from', default='all')
    parser.add_argument('--split', type=str, help='Split to convert. Default = all splits from the common config file',default='all')

    args = parser.parse_args()

    generate_asr_hypotheses(args, config_user, config_common)

    print("Hypotheses generated successfully.")
    """
