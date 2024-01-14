from .base_asr_system import BaseLocalASRSystem

class TestLocalASRSystem(BaseLocalASRSystem):
    def initialize_model(self, model_config):
        # Specific initialization logic for this local ASR system
        pass

    def process_audio(self, audio_sample, model):
        # Specific processing logic for this local ASR system
        pass
