import os
import json
from .base_asr_system import BaseASRSystem
#import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig, ResultReason, CancellationReason

class AzureCloudASR(BaseASRSystem):
    def __init__(self, system, model, credentials:str, region:str, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        super().__init__(system, model)

        # Set up the speech sdk configuration
        self.speech_config = SpeechConfig(subscription=credentials, region=region)
        self.speech_config.speech_recognition_language=language_code
        
    def generate_asr_hyp(self, speech_file):

        try:
            audio_config = AudioConfig(filename=speech_file)
            recognizer = SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
            result = recognizer.recognize_once_async().get()
            hyp = result.text
            if result.reason == ResultReason.RecognizedSpeech:
                print("Azure: {}".format(hyp))
            elif result.reason == ResultReason.NoMatch:
                print("No speech could be recognized: {}".format(result.no_match_details))
            elif result.reason == ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")        
        except IndexError:
            result = "Index error"
        except Exception as e:
            print(f"Other error: {e}")
        self.update_cache(speech_file, hyp)
        return(hyp)
