
import os
import json

class BaseLocalASRSystem:
    
    def initialize_model(self, model_config):
        raise NotImplementedError

    def process_audio(self, audio_sample, model):
        raise NotImplementedError
    
    def get_name(self):
        raise NotImplementedError

class BaseCloudASRSystem:
    def __init__(self, system, model) -> None:
        
        self.system = system
        self.model = model
        
        self.codename = "Cloud_ASR_{}_{}".format(self.system.upper(), self.model.upper())
        self.name = "Cloud ASR - {} - {}".format(self.system.upper(), self.model.upper())
        print("Initializing Cloud ASR System for system {} and model {}".format(system, model))

        self.common_cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/asr_hyps_cache")
        os.makedirs(self.common_cache_dir, exist_ok=True)

        # Set up cache for already processed audio samples
        self.cache = {}
        self.cache_dir = os.path.join(self.common_cache_dir, self.codename)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(self.cache_dir, "asr_cache.json")
        
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)


    def get_model(self):
        return self.model
    
    def process_audio(self, speech_file:str) -> str:
        # Load the audio into memory
        print("Processing audio with {}".format(self.get_name()))
        print("Filename:", os.path.basename(speech_file))

        # Load results from cache if possible
        asr_hyp = self.get_hyp_from_cache(speech_file)
        if asr_hyp is not None:
            print("ASR hypothesis loaded from cache")
            print("Hypothesis: ", asr_hyp, "\n")
            return asr_hyp
        else:
            return self.generate_asr_hyp(speech_file)
        
    def generate_asr_hyp(self, speech_file):
        raise NotImplementedError
    
    def get_name(self):
        return self.name
    
    def get_codename(self):
        return self.codename
    
    def get_hyp_from_cache(self, audio_sample):
        if audio_sample in self.cache:
            return self.cache[audio_sample]
        else:
            return None
        
    def update_cache(self, audio_sample, asr_hyp):
        self.cache[audio_sample] = asr_hyp
    
    def save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)
