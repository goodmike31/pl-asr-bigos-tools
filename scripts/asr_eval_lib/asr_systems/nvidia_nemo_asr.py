from .base_asr_system import BaseASRSystem
import nemo.collections.asr as nemo_asr

import torch

torch.cuda.empty_cache()


class NvidiaNemoASR(BaseASRSystem):
    """NVIDIA NeMo ASR system implementation for the BIGOS framework.
    
    Provides integration with NVIDIA's NeMo toolkit for speech recognition.
    
    Attributes:
        nemo_asr_model: The loaded NeMo ASR model (can be EncDecHybridRNNTCTCBPEModel or EncDecCTCModel).
    """
    
    def __init__(self, system, model, language_code="pl-PL", sampling_rate=16000):
        """Initialize the NVIDIA NeMo ASR system.
        
        Args:
            system (str): Identifier for the ASR system type ('nemo').
            model (str): The specific NeMo model to use.
            language_code (str, optional): Language code. Defaults to "pl-PL".
            sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
            
        Raises:
            ValueError: If an unsupported model type is specified.
        """
        super().__init__(system, model, language_code)
        # check if model name contains "fastconformer" in its name to determine model loading method
        if "fastconformer" in model:
            self.nemo_asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=model)
        elif "quartznet" in model:
            self.nemo_asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model)
        else:
            raise ValueError(f"Unknown model type: {model}")
        
    def generate_asr_hyp(self, speech_file):
        """Generate transcription for an audio file using NVIDIA NeMo.
        
        Args:
            speech_file (str): Path to the audio file to transcribe.
            
        Returns:
            str: The transcription result.
        """
        try:
            asr_output = self.nemo_asr_model.transcribe(paths2audio_files=[speech_file])
            if "fastconformer" in self.model:
                hyp = asr_output[0][0]
            elif "quartznet" in self.model:
                hyp = asr_output[0]
            print("Hyp:", hyp)
        except Exception as e:
            print(f"Other error: {e}")
        
        self.update_cache(speech_file, hyp)
        return hyp