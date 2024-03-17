from .base_asr_system import BaseASRSystem
import whisper

class WhisperLocalASR(BaseASRSystem):
    def __init__(self, system, model, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        super().__init__(system, model, language_code)
        self.whisper_local_language = self.language_code.split("-")[0]
        self.whisper_local_model = whisper.load_model(model)  # You can choose different model sizes

    def generate_asr_hyp(self, speech_file):
        try:
            result = self.whisper_local_model.transcribe(speech_file, language=self.whisper_local_language)
            hyp=result["text"]
            print("Hyp:", hyp)
        except Exception as e:
            print(f"Other error: {e}")
        
        self.update_cache(speech_file, hyp)
        return hyp