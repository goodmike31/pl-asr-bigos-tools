import os
import json
from .base_asr_system import BaseCloudASRSystem
from google.cloud import speech
#import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig, ResultReason, CancellationReason
import openai
import time

class GoogleCloudASR(BaseCloudASRSystem):
    #https://cloud.google.com/speech-to-text/docs/transcription-model#speech_transcribe_model_selection-python
    def __init__(self, system, model, credentials:str, language_code:str = "pl-PL", enable_automatic_punctuation:bool = True, sampling_rate:int = 16000):
        super().__init__(system, model)

        # system specific handling of creadentials. Can be API key or path to credentials file        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

        # Initialize the Google Cloud Speech client
        self.client = speech.SpeechClient()
        
        # Set up the configuration
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=language_code,
            enable_automatic_punctuation=enable_automatic_punctuation,
            model=self.get_model(),
            sample_rate_hertz=sampling_rate,
        )
        
    def generate_asr_hyp(self, speech_file:str) -> str:
        # if not available in cache, process audio
        with open(speech_file, "rb") as audio_file:
            audio_content = audio_file.read()
        
        # Create an audio object
        audio = speech.RecognitionAudio(content=audio_content)

        # Call the Google Cloud Speech API
        response = self.client.recognize(config=self.config, audio=audio)

        # Process and return the recognition result
        # For simplicity, we're returning the transcript of the first result.
        # In a real application, you might want to handle multiple segments.
        for result in response.results:
            print("ASR hypothesis generated from audio sample")
            hyp=result.alternatives[0].transcript
            print("Transcript: {}".format(hyp))
            print("Confidence: {}".format(round(result.alternatives[0].confidence),-2))
            self.update_cache(speech_file, hyp)
            return hyp

        """
        To return object with all results:
        
        -> speech.RecognizeResponse:

        for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print("-" * 20)
        print(f"First alternative of result {i}")
        print(f"Transcript: {alternative.transcript}")

        return response
        """

        return None

class AzureCloudASR(BaseCloudASRSystem):
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

class WhisperCloudASR(BaseCloudASRSystem):
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

