import os
import json
from datetime import datetime
import librosa

class BaseTTSSystem:
    def __init__(self, system, voice, language_code) -> None:
        
        self.system = system
        self.voice = voice
        self.language_code = language_code
        self.sampling_rate = 16000 # TODO: make it configurable

        self.year = datetime.now().year
        self.quarter = (datetime.now().month-1)//3 + 1
        self.version = "{}Q{}".format(self.year, self.quarter)
        
        self.codename = "{}_{}".format(system.lower(), voice.lower())
        self.name = "{} - {}".format(system.upper(), voice.upper())
        print("Initializing ASR system {}, voice {}, version {}".format(system, voice, self.version))

        self.common_cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../data/tts_audio_cache")
        os.makedirs(self.common_cache_dir, exist_ok=True)

        # Set up cache for already processed prompts and generated audio samples
        self.cache = {}
        self.cache_file = os.path.join(self.common_cache_dir, self.codename + ".tts_audio_cache.jsonl")
        self.cache_dir = os.path.join(self.common_cache_dir, self.codename + "_audio_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                # read as JSONL file
                for line in f:
                    self.cache.update(json.loads(line))

    def get_voice(self):
        return self.voice
    
    # TODO consider saving only ID of file and dataset name, not the full path, to make cache reusable across systems
    def gen_audio(self, audio_filename:str, prompt_text:str) -> str:
        # Load the audio into memory
        print("Generating audio file with {}".format(self.get_name()))
        target_audio_file = audio_filename
        print("Target audio path:", target_audio_file)

        filename_from_prompt_text = "{}.wav".format(str(abs(hash(prompt_text))))
        target_audio_cache = os.path.join(self.cache_dir, filename_from_prompt_text)
        print("Target audio cache path:", target_audio_cache)

        # check if prompt_text is not empty
        if prompt_text == "":
            print("Prompt text is empty. Skipping audio generation.")
            return None 

        # check if audio file already exists
        if os.path.exists(target_audio_file):
            print("Audio file already exists. Skipping audio generation.")
            return target_audio_file
        else:
            print("Audio file does not exist. Checking cache.")
            
            if os.path.exists(target_audio_cache):
                print("Audio file exists in cache. Copying from cache.")
                os.system("cp {} {}".format(target_audio_cache, target_audio_file))
                return target_audio_file
            else:
                print("Audio file does not exist in cache. Generating audio.")
                # generate audio
                self.process_prompt(target_audio_file, prompt_text)
                return target_audio_file

    def get_name(self):
        return self.name
    
    def get_codename(self):
        return self.codename
    
    def get_voice(self):
        return self.voice
    
    def get_system(self):
        return self.system
    
    def get_version(self):
        return self.version

    def get_sampling_rate(self):
        return self.sampling_rate

    def process_prompt(self, target_audio_file, prompt_text):
        raise NotImplementedError("This method is not implemented in the base class.")
