import os
from .base_asr_system import BaseCloudASRSystem
from google.cloud import speech

class GoogleCloudASR(BaseCloudASRSystem):
    #https://cloud.google.com/speech-to-text/docs/transcription-model#speech_transcribe_model_selection-python
    def __init__(self, credentials_path:str, language_code:str = "pl-PL", model:str = "default", enable_automatic_punctuation:bool = True, sampling_rate:int = 16000):
        # Set the path to the credentials file
        print("Initializing Google Cloud ASR System")
        print("credentials_path", credentials_path)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        # Initialize the Google Cloud Speech client
        self.client = speech.SpeechClient()

        # Set up the configuration
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=language_code,
            enable_automatic_punctuation=enable_automatic_punctuation,
            model=model,
            sample_rate_hertz=sampling_rate,
        )

    def process_audio(self, speech_file:str) -> str:
        # Load the audio into memory
        print("Processing audio with Google Cloud ASR System")
        print("speech_file", speech_file)
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
            print("Transcript: {}".format(result.alternatives[0].transcript))
            return result.alternatives[0].transcript

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
    def initialize_model(self, model, config):
        # Specific initialization logic for this cloud ASR system
        print("Initializing Azure Cloud ASR System")
        pass

    def process_audio(self, audio_sample)->str:
        # Specific processing logic for this cloud ASR system
        print("Processing audio with Azure Cloud ASR System")
        pass
