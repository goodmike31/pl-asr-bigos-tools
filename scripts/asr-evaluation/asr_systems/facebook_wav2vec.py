from .base_asr_system import BaseASRSystem
#from transformers import Wav2Vec2ForCTC, AutoProcessor
from huggingsound import SpeechRecognitionModel
import librosa

class FacebookWav2Vec(BaseASRSystem):
    def __init__(self, system, model, language_code:str = "pl-PL", sampling_rate:int = 16000) -> None:
        super().__init__(system, model, language_code)
        # convert ISO-639-1 to ISO-639-3
        self.wav2vec_model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-" + model)

    def generate_asr_hyp(self, speech_file):
        #TODO add conversion to wav2vec supported input format
        try:
            result = self.wav2vec_model.transcribe([speech_file])
            hyp = result[0]["transcription"]
            print("Hyp:", hyp)
            
        except Exception as e:
            print(f"Other error: {e}")
        
        self.update_cache(speech_file, hyp)
        return hyp