from .base_asr_system import BaseASRSystem
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig, ResultReason, CancellationReason

class AzureCloudASR(BaseASRSystem):
    """Microsoft Azure Speech Service implementation for the BIGOS framework.
    
    Provides integration with the Microsoft Azure Speech-to-Text API.
    
    Attributes:
        speech_config (SpeechConfig): Azure speech recognition configuration.
    """
    
    def __init__(self, system, model, credentials:str, region:str, language_code:str = "pl-PL", sampling_rate:int = 16000) -> None:
        """Initialize the Azure Speech Service ASR system.
        
        Args:
            system (str): Identifier for the ASR system type ('azure').
            model (str): The specific model to use.
            credentials (str): Azure Speech Service subscription key.
            region (str): Azure region for the Speech Service.
            language_code (str, optional): Language code. Defaults to "pl-PL".
            sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
        """
        super().__init__(system, model, language_code)

        # Set up the speech sdk configuration
        self.speech_config = SpeechConfig(subscription=credentials, region=region)
        self.speech_config.speech_recognition_language=language_code
        
    def generate_asr_hyp(self, speech_file):
        """Generate transcription for an audio file using Azure Speech Service.
        
        Args:
            speech_file (str): Path to the audio file to transcribe.
            
        Returns:
            str: The transcription result.
        """
        try:
            audio_config = AudioConfig(filename=speech_file)
            recognizer = SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
            try:
                result = recognizer.recognize_once()
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
            except Exception as e:
                print(f"Error generating outputs: {e}. Skipping generation and returing empty hypothesis.")
                hyp = ""        
        except Exception as e:
            print(f"Other error: {e}")
        self.update_cache(speech_file, hyp)
        return(hyp)
