from .base_asr_system import BaseASRSystem
import openai
from pathlib import Path


class WhisperCloudASR(BaseASRSystem):
    """OpenAI Whisper Cloud ASR system implementation for the BIGOS framework.
    
    Provides integration with OpenAI's Whisper API for speech recognition.
    """
    
    def __init__(self, system, model, credentials, language_code="pl-PL", sampling_rate=16000):
        """Initialize the OpenAI Whisper Cloud ASR system.
        
        Args:
            system (str): Identifier for the ASR system type ('whisper_cloud').
            model (str): The specific Whisper model to use.
            credentials (str): OpenAI API key.
            language_code (str, optional): Language code. Defaults to "pl-PL".
            sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
        """
        super().__init__(system, model, language_code)
        openai.api_key = credentials
        self.language_code = language_code
        
    def generate_asr_hyp(self, speech_file):
        """Generate transcription for an audio file using OpenAI Whisper API.
        
        Args:
            speech_file (str): Path to the audio file to transcribe.
            
        Returns:
            str: The transcription result.
        """
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

