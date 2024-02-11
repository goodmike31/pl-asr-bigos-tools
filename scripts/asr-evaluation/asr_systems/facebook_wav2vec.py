from .base_asr_system import BaseASRSystem
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

class FacebookWav2Vec(BaseASRSystem):
    def __init__(self, system, model, language_code:str = "pl-PL", sampling_rate:int = 16000) -> None:
        super().__init__(system, model, language_code)
        # convert ISO-639-1 to ISO-639-3
        self.model = model
        #TODO - make customizable with system and model parameters
        if (model == "xls-r-1b-polish"):
            self.w2v_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-" + model)
            self.w2v_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-" + model)
        elif (model == "large-xlsr-53-polish"):
            self.w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-" + model)
            self.w2v_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-" + model)
        else:
            raise ValueError(f"Model {model} is not supported")
        self.sampling_rate = sampling_rate

    def generate_asr_hyp(self, speech_file):
        #TODO add conversion to wav2vec supported input format
        try:
            speech_array, sampling_rate = librosa.load(speech_file, sr=self.sampling_rate)

            inputs = self.w2v_processor(speech_array, sampling_rate=16_000, return_tensors="pt")
            
            #outputs = self.w2v_model(inputs).logits
            outputs = self.w2v_model(**inputs).logits
            
            ids = torch.argmax(outputs, dim=-1)[0]
            hyp = self.w2v_processor.decode(ids)

            print(f"Hyp:   {hyp}")

        except Exception as e:
            print(f"Other error: {e}")
        
        self.update_cache(speech_file, hyp)
        return hyp