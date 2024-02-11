from .base_asr_system import BaseASRSystem
import nemo.collections.asr as nemo_asr


class NvidiaNemoASR(BaseASRSystem):
    def __init__(self, system, model, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        super().__init__(system, model, language_code)
        self.nemo_asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_pl_fastconformer_hybrid_large_pc")

    def generate_asr_hyp(self, speech_file):
        try:
            hyp = self.nemo_asr_model.transcribe(paths2audio_files=[speech_file])[0][0]
            print("Hyp:", hyp)
        except Exception as e:
            print(f"Other error: {e}")
        
        self.update_cache(speech_file, hyp)
        return hyp