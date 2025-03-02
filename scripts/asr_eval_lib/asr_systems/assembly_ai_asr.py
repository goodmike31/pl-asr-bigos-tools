from .base_asr_system import BaseASRSystem
import assemblyai as aai
from pathlib import Path


class AssemblyAIASR(BaseASRSystem):
    """AssemblyAI ASR system implementation for the BIGOS framework.
    
    Provides integration with the AssemblyAI speech recognition API.
    
    Attributes:
        transcriber (aai.Transcriber): AssemblyAI transcriber instance.
        config (aai.TranscriptionConfig): Configuration for the transcription.
    """
    
    def __init__(self, system, model, credentials:str, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        """Initialize the AssemblyAI ASR system.
        
        Args:
            system (str): Identifier for the ASR system type ('assembly_ai').
            model (str): The specific model to use ('best' or 'nano').
            credentials (str): AssemblyAI API key.
            language_code (str, optional): Language code. Defaults to "pl-PL".
            sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
        
        Raises:
            SystemExit: If an unsupported model is specified.
        """
        print(system, model, language_code)
        super().__init__(system, model, language_code)
        
        aai.settings.api_key = credentials
        
        self.transcriber = aai.Transcriber()

        if model == "best":
            speech_model=aai.SpeechModel.best
        elif model == "nano":
            speech_model=aai.SpeechModel.nano
        else:
            print("Model {} not supported".format(model))
            exit()
        lang_code_short = language_code.split("-")[0]
        print("Language code short: ", lang_code_short)
        self.config = aai.TranscriptionConfig(language_code=lang_code_short, speech_model=speech_model)
        
    def generate_asr_hyp(self, speech_file):
        """Generate transcription for an audio file using AssemblyAI API.
        
        Args:
            speech_file (str): Path to the audio file to transcribe.
            
        Returns:
            str: The transcription result.
        """
        try:   
            # Create transcription from audio file
            print("Transcribing audio file: ", speech_file)
            transcript_obj = self.transcriber.transcribe(speech_file, config=self.config)
            #print("Transcript object: ", transcript_obj)
            hyp = transcript_obj.text
            #print("Generated hyp inside assembly ASR class: ", hyp)       
            #time.sleep(1)
        except Exception as e:
            print(f"Other error: {e}")
        
        self.update_cache(speech_file, hyp)
        return hyp

