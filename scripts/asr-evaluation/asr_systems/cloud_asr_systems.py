import os
import json
from .base_asr_system import BaseCloudASRSystem
from google.cloud import speech

class GoogleCloudASR(BaseCloudASRSystem):
    #https://cloud.google.com/speech-to-text/docs/transcription-model#speech_transcribe_model_selection-python
    def __init__(self, credentials_path:str, language_code:str = "pl-PL", model:str = "default", enable_automatic_punctuation:bool = True, sampling_rate:int = 16000):
        
        # TODO move to base class constructor
        self.common_cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/asr_hyps_cache")
        os.makedirs(self.common_cache_dir, exist_ok=True)
      
        # Set the path to the credentials file
        print("Initializing Google Cloud ASR System for model {}".format(model))
        
        self.model = model
        
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
        self.codename = "GOOGLE_CLOUD_ASR_{}".format(self.model.upper())
        self.name = "Google Cloud ASR - {}".format(self.model.upper())

        # Set up cache for already processed audio samples
        self.cache = {}
        self.cache_dir = os.path.join(self.common_cache_dir, self.codename)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(self.cache_dir, "asr_cache.json")
        
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)


    def process_audio(self, speech_file:str) -> str:
        # Load the audio into memory
        print("Processing audio with {}".format(self.get_name()))
        print("speech_file", speech_file)

        # Load results from cache if possible
        asr_hyp = self.get_hyp_from_cache(speech_file)
        if asr_hyp is not None:
            print("ASR hypothesis loaded from cache")
            print(asr_hyp)
            return asr_hyp
        
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
            print("Transcript: {}".format(result.alternatives[0].transcript))
            print("Confidence: {}".format(result.alternatives[0].confidence))
            self.update_cache(speech_file, result.alternatives[0].transcript)
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
    
    def get_name(self):
        return self.name
    
    def get_codename(self):
        return self.codename
    
    def get_hyp_from_cache(self, audio_sample):
        if audio_sample in self.cache:
            return self.cache[audio_sample]
        else:
            return None
        
    def update_cache(self, audio_sample, asr_hyp):
        self.cache[audio_sample] = asr_hyp
    
    def save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

class AzureCloudASR(BaseCloudASRSystem):
    def initialize_model(self, model, config):
        # Specific initialization logic for this cloud ASR system
        print("Initializing Azure Cloud ASR System")
        pass

    def process_audio(self, audio_sample)->str:
        # Specific processing logic for this cloud ASR system
        print("Processing audio with Azure Cloud ASR System")
        pass
