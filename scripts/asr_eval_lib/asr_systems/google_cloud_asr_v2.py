import os
from .base_asr_system import BaseASRSystem
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

class GoogleCloudASRV2(BaseASRSystem):
    """Google Cloud Speech-to-Text API V2 implementation for the BIGOS framework.
    
    Provides integration with the Google Cloud Speech-to-Text API V2.
    
    Attributes:
        client (SpeechClient): Google Cloud Speech V2 client.
        config (cloud_speech.RecognitionConfig): Configuration for speech recognition.
        project_id (str): Google Cloud project ID.
    """
    
    def __init__(self, system, model, credentials:str, project_id:str, language_code:str = "pl-PL", enable_automatic_punctuation:bool = True, sampling_rate:int = 16000):
        """Initialize the Google Cloud Speech-to-Text V2 ASR system.
        
        Args:
            system (str): Identifier for the ASR system type ('google_v2').
            model (str): The specific model to use ('short' or 'long').
            credentials (str): Path to Google Cloud credentials JSON file.
            project_id (str): Google Cloud project ID.
            language_code (str, optional): Language code. Defaults to "pl-PL".
            enable_automatic_punctuation (bool, optional): Whether to enable automatic punctuation. Defaults to True.
            sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
        """
        super().__init__(system, model,language_code)
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

        self.project_id = project_id

        # Initialize the Google Cloud Speech client
        self.client = SpeechClient()
        
        # Set up the configuration
        self.config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[language_code],
            model=model,
        )
        if (model == "short"):
            self.max_audio_length_to_process_sec = 30
        elif (model == "long"):
            self.max_audio_length_to_process_sec = 60
  
        
    def generate_asr_hyp(self, speech_file:str) -> str:
        """Generate transcription for an audio file using Google Cloud Speech-to-Text V2.
        
        Args:
            speech_file (str): Path to the audio file to transcribe.
            
        Returns:
            str: The transcription result, or None if no results were returned.
        """
        project_id = self.project_id
        # if not available in cache, process audio
        with open(speech_file, "rb") as audio_file:
            audio_content = audio_file.read()
        
        request = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{project_id}/locations/global/recognizers/_",
            config=self.config,
            content=audio_content,
        )

        # Call the Google Cloud Speech API
        response = self.client.recognize(request=request)

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
