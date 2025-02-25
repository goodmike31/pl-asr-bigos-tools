from .base_asr_system import BaseASRSystem
import nemo.collections.asr as nemo_asr

import torch

torch.cuda.empty_cache()


class NvidiaNemoASR(BaseASRSystem):
    def __init__(self, system, model, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        super().__init__(system, model, language_code)
        # check if model name contains "fastconformer" in its name to determine model loading method
        if "fastconformer" in model:
            self.nemo_asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=model)
        elif "quartznet" in model:
            self.nemo_asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model)
        else:
            raise ValueError(f"Unknown model type: {model}")
        
    def generate_asr_hyp(self, speech_file):
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