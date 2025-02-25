from .base_asr_system import BaseASRSystem
import whisper
import torch

torch.cuda.empty_cache()

class WhisperLocalASR(BaseASRSystem):
    def __init__(self, system, model, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        super().__init__(system, model, language_code)
        self.whisper_local_language = self.language_code.split("-")[0]
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # temporary solution to avoid crashing the GPU on local machine
        #device = "cpu"

        if not torch.cuda.is_available():
            raise Warning("Please use GPU for better inference speed.")
        
        try:
            self.whisper_local_model_default = whisper.load_model(model, device=self.device)  # You can choose different model sizes
        except Exception as e:
            print(f"Default device model loading fail: {e}")
            print("Using CPU model")
            self.device = "cpu" 
            self.whisper_local_model_default = whisper.load_model(model, device="cpu")  # You can choose different model sizes
        if (self.device == "cuda"):
            self.whisper_local_model_cpu = whisper.load_model(model, device="cpu")  # backup CPU model
        
    def generate_asr_hyp(self, speech_file):
        try:
            print("Using default device for decoding: ", self.device)

            result = self.whisper_local_model_default.transcribe(speech_file, language=self.whisper_local_language)
            hyp=result["text"]
            print("Hyp:", hyp)
        except Exception as e:
                print("Default device generation fail. Using CPU")
                try:
                    result = self.whisper_local_model_cpu.transcribe(speech_file, language=self.whisper_local_language)
                    hyp=result["text"]
                    print("Hyp:", hyp)
                except Exception as e:
                    print(f"Other error: {e}")
                    exit()

        
        self.update_cache(speech_file, hyp)
        return hyp