import os
import json
from datetime import datetime
import librosa
import sys

# Get the parent directory
repo_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
print("repo_root_dir", repo_root_dir)

# Add the parent directory to sys.path
sys.path.insert(0, repo_root_dir)

from scripts.utils.utils import read_config_ini, read_config_json

# Load the user-specific config file
config_user_path = os.path.join(repo_root_dir, 'config/user-specific/config.ini')
print("config_user_path", config_user_path)

config_user = read_config_ini(config_user_path)
bigos_eval_data_dir = config_user["PATHS"]["BIGOS_EVAL_DATA_REPO_PATH"]

class BaseASRSystem:
    """
    Base class for all ASR systems in the BIGOS benchmark.
    
    This class provides common functionality for all ASR systems including:
    - Caching of previously processed audio samples
    - Interface for processing audio and generating transcriptions
    - Version control for ASR systems
    - Standard methods for retrieving system information
    
    All ASR system implementations should inherit from this class and 
    implement the generate_asr_hyp method.
    
    Attributes:
        system (str): Name of the ASR system (e.g., "google", "azure")
        model (str): Name of the specific model being used
        language_code (str): ISO language code (e.g., "pl-PL")
        version (str): Version identifier (YearQuarter format)
        codename (str): Unique identifier for this system and model
        name (str): Human-readable name of the system
        cache (dict): Cache of processed audio samples
        max_audio_length_to_process_sec (int): Maximum audio duration to process in seconds
        bigos_eval_data_dir (str): Directory for storing evaluation data
        common_cache_dir (str): Directory for storing cached hypotheses
        cache_file (str): Path to the cache file for this ASR system
    """

    def __init__(self, system, model, language_code) -> None:
        """
        Initialize the ASR system.
        
        Args:
            system (str): Name of the ASR system
            model (str): Name of the specific model being used
            language_code (str): ISO language code (e.g., "pl-PL")
        """
        self.system = system
        self.model = model
        self.language_code = language_code
        self.max_audio_length_to_process_sec = 300
        self.bigos_eval_data_dir = bigos_eval_data_dir

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
        #remove "/" from codename
        self.codename = self.codename.replace("/", "_")
        
        self.name = "{} - {}".format(system.upper(), model.upper())
        print("Initializing ASR system {}, model {}, version {}".format(system, model, self.version))

        self.common_cache_dir = os.path.join(self.bigos_eval_data_dir, "asr_hyps_cache")
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
        """
        Get the model name.
        
        Returns:
            str: The model name
        """
        return self.model
    
    def process_audio(self, speech_file: str, force_hyps) -> str:
        """
        Process an audio file and return the transcription.
        
        This method handles the caching logic and calls generate_asr_hyp()
        which should be implemented by subclasses. It also handles various
        error conditions and edge cases like empty files or files that are
        too long.
        
        Args:
            speech_file (str): Path to the audio file
            force_hyps (bool): Whether to force generation of new hypotheses
                               regardless of cache status
            
        Returns:
            str: Transcription text or error indicator ("EMPTY" or "INVALID")
        """
        # Load the audio into memory
        print("Processing audio with {}".format(self.get_name()))
        print("Filename:", os.path.basename(speech_file))
        print("Path:", speech_file)

        # check librosa version
        print("Librosa version: ", librosa.__version__)
        if librosa.__version__ < "0.10.0":
            audio_duration = round(librosa.get_duration(path=speech_file),2)
        else:
            audio_duration = round(librosa.get_duration(filename=speech_file),2)
        
        print("Audio duration [s]: ", audio_duration)

        # Check if the files exists
        if not os.path.exists(speech_file):
            print("File does not exist")
            return ""
        if os.path.getsize(speech_file) == 0:
            print("File is empty")
            return ""
        
        # check if audio length exceeds maximum allowed duration
        if audio_duration > self.max_audio_length_to_process_sec:
            print("Audio length exceeds max allowed duration of {} seconds. Skipping".format(self.max_audio_length_to_process_sec))
            return ""

        if not force_hyps:
            # Load results from cache if possible
            print("Checking cache:")
            asr_hyp = self.get_hyp_from_cache(speech_file, self.version)

            # Generate new hypothesis if cache is empty or None
            if asr_hyp is None:
                print("Hypothesis in cache not available. Generating new hypothesis.")
            elif asr_hyp == "INVALID":
                print("Hypothesis in cache is invalid. Generating new hypothesis.")
            elif asr_hyp == "":
                print("Hypothesis in cache is the empty string. Generating new hypothesis.")
            else:
                print("Hypothesis in cache is VALID: {}. Returning.".format(asr_hyp))
                return asr_hyp

        asr_hyp = self.generate_asr_hyp(speech_file)
        print("NEW ASR hypothesis: ", asr_hyp)

        # Handle newly generated hypothesis
        if asr_hyp == "":
            print("ASR hypothesis is EMPTY AGAIN. Saving value EMPTY in cache.")
            self.update_cache(speech_file, "EMPTY")
            return "EMPTY"
        elif asr_hyp is None:
            print("ASR hypothesis is None. Trying again.")
            asr_hyp = self.generate_asr_hyp(speech_file)
            print("NEW ASR hypothesis: ", asr_hyp)
            if asr_hyp == "":
                print("ASR hypothesis is EMPTY AGAIN. Saving value EMPTY in cache.")
                self.update_cache(speech_file, "EMPTY")
                return "EMPTY"
            elif asr_hyp is None:
                print("ASR hypothesis is None AGAIN. Saving value INVALID in cache.")
                self.update_cache(speech_file, "INVALID")
                return "INVALID"
        else:
            print("Non empty hyp - saving in cache.")
            self.update_cache(speech_file, asr_hyp)
        
        return asr_hyp
            
    def get_name(self):
        """
        Get the human-readable name of the system.
        
        Returns:
            str: Human-readable name combining system and model
        """
        return self.name
    
    def get_codename(self):
        """
        Get the system codename (unique identifier).
        
        Returns:
            str: System codename
        """
        return self.codename
    
    def get_system(self):
        """
        Get the system name.
        
        Returns:
            str: System name
        """
        return self.system
    
    def get_version(self):
        """
        Get the system version.
        
        Returns:
            str: System version in YearQuarter format
        """
        return self.version
    
    def get_hyp_from_cache(self, audio_path, version):
        """
        Retrieve a hypothesis from the cache for the given audio path and version.
        
        This method first tries to find the exact audio path in the cache. If not found,
        it will try to match by filename only.
        
        Args:
            audio_path (str): Full path to the audio file
            version (str): Version of the ASR system
            
        Returns:
            str or None: The cached hypothesis or None if not found
        """
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
        """
        Update the cache with a new hypothesis.
        
        Args:
            audio_path (str): Path to the audio file
            asr_hyp (str): ASR hypothesis to cache
        """
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
        """
        Save the cache to disk in JSONL format.
        """
        print("Saving cache")
        with open(self.cache_file, "w") as f:
            # save as JSONL file
            for audio_path in self.cache:
                # save JSONL using audio sample as key
                json.dump({audio_path: self.cache[audio_path]}, f)
                f.write("\n")

    def get_cached_hyps(self):
        """
        Get all cached hypotheses.
        
        Returns:
            dict: Dictionary of all cached hypotheses
        """
        return self.cache
    
    def generate_asr_hyp(self, speech_file):
        """
        Generate an ASR hypothesis for the given audio file.
        
        This method must be implemented by subclasses.
        
        Args:
            speech_file (str): Path to the audio file
            
        Returns:
            str: The ASR hypothesis
            
        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement the generate_asr_hyp method")