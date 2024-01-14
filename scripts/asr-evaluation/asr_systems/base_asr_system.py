
class BaseLocalASRSystem:
    def initialize_model(self, model_config):
        raise NotImplementedError

    def process_audio(self, audio_sample, model):
        raise NotImplementedError

class BaseCloudASRSystem:
    def initialize_model(self, model_config):
        raise NotImplementedError

    def process_audio(self, audio_sample,):
        raise NotImplementedError
