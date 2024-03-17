from .base_asr_system import BaseASRSystem
import openai
from pathlib import Path


class WhisperCloudASR(BaseASRSystem):
    def __init__(self, system, model, credentials:str, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        super().__init__(system, model, language_code)
        openai.api_key = credentials
        self.language_code = language_code
        
    def generate_asr_hyp(self, speech_file):
        try:   
            # Create transcription from audio file
            transcription = openai.audio.transcriptions.create(
                model=self.model,
                file=Path(speech_file))
            print(transcription)       
            hyp = transcription.text
            #time.sleep(1)
        except IndexError:
            print("Index error")
        except openai.BadRequestError as e:
            if "Audio file is too short" in str(e):
                print(f"Error: {e}")
        except Exception as e:
            print(f"Other error: {e}")
        
        self.update_cache(speech_file, hyp)
        return hyp

