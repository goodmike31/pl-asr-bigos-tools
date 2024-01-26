import os
import json
from datetime import datetime
class BaseASRSystem:
    def __init__(self, system, model, language_code) -> None:
        
        self.system = system
        self.model = model
        self.language_code = language_code

        # add version encoded as YYQ (year and quarter) to codename to control for changes in the ASR system and model over time
        # assumes that the ASR system and model are evaluated at most once per quarter
        # get current year and quarter
        # TODO consider making it more flexible, e.g. by allowing to specify the version in the config file

        self.year = datetime.now().year
        self.quarter = (datetime.now().month-1)//3 + 1
        version = "{}Q{}".format(self.year, self.quarter)
        self.version = version
        self.codename = "ASR_{}_{}".format(self.system.upper(), self.model.upper(), self.version)
        self.name = "ASR - {} - {}".format(self.system.upper(), self.model.upper(), self.version)
        print("Initializing ASR system {}, model {}, version {}".format(system, model, version))

        self.common_cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../data/asr_hyps_cache")
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
        metadata = {
            'model': self.model,
            'codename': self.codename,
            'hypothesis_generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.cache[audio_sample] = {
            'asr_hyp': asr_hyp,
            'metadata': metadata
        }
    
    def save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)
    
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
