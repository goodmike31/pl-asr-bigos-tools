import os
import json
from datetime import datetime
import librosa

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
        self.version = "{}Q{}".format(self.year, self.quarter)
        
        self.codename = "{}_{}".format(system.lower(), model.lower())
        self.name = "{} - {}".format(system.upper(), model.upper())
        print("Initializing ASR system {}, model {}, version {}".format(system, model, self.version))

        self.common_cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../data/asr_hyps_cache")
        os.makedirs(self.common_cache_dir, exist_ok=True)

        # Set up cache for already processed audio samples
        self.cache = {}
        self.cache_file = os.path.join(self.common_cache_dir, self.codename + ".asr_cache.jsonl")
        
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                # read as JSONL file
                for line in f:
                    self.cache.update(json.loads(line))

    def get_model(self):
        return self.model
    
    # TODO consider saving only ID of file and dataset name, not the full path, to make cache reusable across systems
    def process_audio(self, speech_file:str) -> str:
        # Load the audio into memory
        print("Processing audio with {}".format(self.get_name()))
        print("Filename:", os.path.basename(speech_file))

        # Check if the files exists
        if not os.path.exists(speech_file):
            print("File does not exist")
            return ""
        if os.path.getsize(speech_file) == 0:
            print("File is empty")
            return ""
        
        # check if audio length exceeds 1 minute
        if librosa.get_duration(filename=speech_file) > 60:
            print("Audio length exceeds 1 minute")
            return ""

        # Load results from cache if possible
        asr_hyp = self.get_hyp_from_cache(speech_file, self.version)
        if asr_hyp is not None:
            print("ASR hypothesis loaded from cache")
            print("Hypothesis: ", asr_hyp, "\n")
            return asr_hyp
        else:
            return self.generate_asr_hyp(speech_file)

    def get_name(self):
        return self.name
    
    def get_codename(self):
        return self.codename
    
    def get_model(self):
        return self.model
    
    def get_system(self):
        return self.system
    
    def get_version(self):
        return self.version
    
    def get_hyp_from_cache(self, audio_sample, version):
        # check if audio sample is in cache
        if audio_sample in self.cache:
            # check if version is in cache
            if version in self.cache[audio_sample]:
                return self.cache[audio_sample][version]['asr_hyp']
            else:
                return None
        else:
            return None
        
    def update_cache(self, audio_sample, asr_hyp):
        metadata = {
            'asr_hyp': asr_hyp,
            'system': self.system,
            'model': self.model,
            'version': self.version,
            'codename': self.codename,
            'hyp_gen_date': datetime.now().strftime("%Y%m%d")
        }
        self.cache[audio_sample] = {self.version: metadata}
    
    def save_cache(self):
        with open(self.cache_file, "w") as f:
            # save as JSONL file
            for audio_sample in self.cache:
                # save JSONL using audio sample as key
                json.dump({audio_sample: self.cache[audio_sample]}, f)
                f.write("\n")