import os
import json
from .base_asr_system import BaseASRSystem
import openai
import time

class WhisperCloudASR(BaseASRSystem):
    def __init__(self, system, model, credentials:str, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        super().__init__(system, model)
        openai.api_key = credentials
        self.language_code = language_code
        
    def generate_asr_hyp(self, speech_file):
        try:
            audio_data = open(speech_file, "rb")
            # check if request is valid and what version is used
            response = openai.Audio.transcribe(self.model, audio_data)
            hyp=response.text
            #time.sleep(1)
        except IndexError:
            print("Index error")
        except openai.error.InvalidRequestError as e:
            if "Audio file is too short" in str(e):
                print(f"Error: {e}")
        except Exception as e:
            print(f"Other error: {e}")
        
        self.update_cache(speech_file, hyp)
        return hyp

