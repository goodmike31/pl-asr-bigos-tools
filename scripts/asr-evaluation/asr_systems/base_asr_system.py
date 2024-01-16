
import os

class BaseLocalASRSystem:
    
    def initialize_model(self, model_config):
        raise NotImplementedError

    def process_audio(self, audio_sample, model):
        raise NotImplementedError
    
    def get_name(self):
        raise NotImplementedError

class BaseCloudASRSystem:
    #def __init__(self) -> None:
        #print("set common cache dir")
        #self.common_cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/asr_hyps_cache")
        #os.makedirs(self.common_cache_dir, exist_ok=True)

    def initialize_model(self, model_config):
        raise NotImplementedError

    def process_audio(self, audio_sample,):
        raise NotImplementedError
