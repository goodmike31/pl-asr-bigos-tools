import os
from .base_asr_system import BaseASRSystem
from google.cloud import speech

class GoogleCloudASR(BaseASRSystem):
    #https://cloud.google.com/speech-to-text/docs/transcription-model#speech_transcribe_model_selection-python
    def __init__(self, system, model, credentials:str, language_code:str = "pl-PL", enable_automatic_punctuation:bool = True, sampling_rate:int = 16000):
        super().__init__(system, model, language_code)

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
        if (model == "default"):
            self.max_audio_length_to_process_sec = 60
        elif (model == "command_and_search"):
            self.max_audio_length_to_process_sec = 60
        elif (model == "latest_short"):
            self.max_audio_length_to_process_sec = 30
        elif (model == "latest_long"):
            self.max_audio_length_to_process_sec = 60    

        
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
