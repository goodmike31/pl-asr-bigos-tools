import os
from .base_asr_system import BaseASRSystem
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

class GoogleCloudASRV2(BaseASRSystem):
    #https://cloud.google.com/speech-to-text/docs/transcription-model#speech_transcribe_model_selection-python
    def __init__(self, system, model, project_id:str, language_code:str = "pl-PL", enable_automatic_punctuation:bool = True, sampling_rate:int = 16000):
        super().__init__(system, model,language_code)

        self.project_id = project_id

        # Initialize the Google Cloud Speech client
        self.client = SpeechClient()
        
        # Set up the configuration
        self.config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_code=language_code,
            enable_automatic_punctuation=enable_automatic_punctuation,
            model=self.get_model(),
            sample_rate_hertz=sampling_rate,
        )
        
    def generate_asr_hyp(self, speech_file:str) -> str:
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
