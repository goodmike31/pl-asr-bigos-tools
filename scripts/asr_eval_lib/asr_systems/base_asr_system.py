import os
import json
from datetime import datetime
import librosa

class BaseASRSystem:
    def __init__(self, system, model, language_code) -> None:
        
        self.system = system
        self.model = model
        self.language_code = language_code
        self.max_audio_length_to_process_sec = 300

        # add version encoded as YYQ (year and quarter) to codename to control for changes in the ASR system and model over time
        # assumes that the ASR system and model are evaluated at most once per quarter
        # get current year and quarter
        # TODO consider making it more flexible, e.g. by allowing to specify the version in the config file

        #self.year = datetime.now().year
        #self.quarter = (datetime.now().month-1)//3 + 1
        self.version = "2024Q1"
        # TODO - add version as input argument to control ASR version somehow
        #"{}Q{}".format(self.year, self.quarter)
        
        self.codename = "{}_{}".format(system.lower(), model.lower())
        self.name = "{} - {}".format(system.upper(), model.upper())
        print("Initializing ASR system {}, model {}, version {}".format(system, model, self.version))

        self.common_cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../data/asr_hyps_cache")
        os.makedirs(self.common_cache_dir, exist_ok=True)

        # Set up cache for already processed audio samples
        self.cache = {}
        self.cache_file = os.path.join(self.common_cache_dir, self.codename + ".asr_cache.jsonl")
        print("Reading cache: ", self.cache_file)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                # read as JSONL file
                for line in f:
                    self.cache.update(json.loads(line))
        else:
            print("Cache file does not exist")
            
    def get_model(self):
        return self.model
    
    # TODO consider saving only ID of file and dataset name, not the full path, to make cache reusable across systems
    def process_audio(self, speech_file:str) -> str:
        # Load the audio into memory
        print("Processing audio with {}".format(self.get_name()))
        print("Filename:", os.path.basename(speech_file))
        print("Path:", speech_file)
        audio_duration = round(librosa.get_duration(path=speech_file),2)
        print("Audio duration [s]: ", audio_duration)

        # Check if the files exists
        if not os.path.exists(speech_file):
            print("File does not exist")
            return ""
        if os.path.getsize(speech_file) == 0:
            print("File is empty")
            return ""
        
        # check if audio length exceeds 1 minute
        if audio_duration > self.max_audio_length_to_process_sec:
            print("Audio length exceeds max allowed duration of {} seconds. Skipping".format(self.max_audio_length_to_process_sec))
            return ""

        # Load results from cache if possible
        print("Checking cache:")
        asr_hyp = self.get_hyp_from_cache(speech_file, self.version)
        if asr_hyp is not None and asr_hyp != "":
            return asr_hyp
        else:
            print("Hypothesis in cache not available or empty. Generating new hypothesis.")
            asr_hyp = self.generate_asr_hyp(speech_file)
            self.update_cache(speech_file, asr_hyp)
            return asr_hyp

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
    
    def get_hyp_from_cache(self, audio_path, version):
        # check if audio sample is in cache
        #print("Checking if audio path is in cache.")
        if audio_path in self.cache:
            # check if version is in cache
            if version in self.cache[audio_path]:
                asr_hyp = self.cache[audio_path][version]['asr_hyp']
                print("READ from cache based on audiopath.\nAudio sample: {}\nHypothesis: {} ".format(audio_path, asr_hyp))
                return asr_hyp
        else:    
            #print("Checking if audio filename is in cache.")
            # get filename without path
            audio_filename = os.path.basename(audio_path)
            #print("Filename: ", audio_filename)
            # the cache key contains full path, so we need to check if the filename is in the cache by iterating over all keys
            for key in self.cache:
                key_filename = os.path.basename(key)
                #print("Key: ", key_filename)
                if key_filename == audio_filename:
                    # print("Filename {} found in cache".format(audio_filename))
                    # print(self.cache[key])
                    # print("Version input: ", version)
                    # check if version is in cache
                    if version in self.cache[key]:
                        asr_hyp = self.cache[key][version]['asr_hyp']
                        print("READ from cache based on filename.\nAudio sample: {}\nHypothesis: {} ".format(key, asr_hyp))
                        # update cache with new key
                        self.cache[audio_path] = asr_hyp
                        return asr_hyp
                    else:
                        #print("Filename {} found in cache, but version {} not in cache".format(audio_filename, version))
                        return None
            #print("Audio filename {} not found in cache".format(audio_filename))
            return None
    
    def update_cache(self, audio_path, asr_hyp):
        metadata = {
            'asr_hyp': asr_hyp,
            'system': self.system,
            'model': self.model,
            'version': self.version,
            'codename': self.codename,
            'hyp_gen_date': datetime.now().strftime("%Y%m%d")
        }
        self.cache[audio_path] = {self.version: metadata}
        print("UPDATED cache.\nAudio sample: {}\nHypothesis: {} ".format(audio_path, asr_hyp))
        self.save_cache()
    
    def save_cache(self):
        print("Saving cache")
        with open(self.cache_file, "w") as f:
            # save as JSONL file
            for audio_path in self.cache:
                # save JSONL using audio sample as key
                json.dump({audio_path: self.cache[audio_path]}, f)
                f.write("\n")

    def get_cached_hyps(self):
        return self.cache