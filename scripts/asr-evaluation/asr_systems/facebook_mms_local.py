from .base_asr_system import BaseASRSystem
from transformers import Wav2Vec2ForCTC, AutoProcessor
import librosa
import torch    

#https://huggingface.co/docs/transformers/v4.36.1/model_doc/mms

lang_code_693_3 = {
    "pl-PL":"pol"
    }

class FacebookMMS(BaseASRSystem):
    def __init__(self, system, model, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        super().__init__(system, model, language_code)
        # convert ISO-639-1 to ISO-639-3
        self.model = model
        self.mms_lang = lang_code_693_3[language_code]
        self.mms_model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-" + model)
        self.mms_model.load_adapter(self.mms_lang)

        self.processor = AutoProcessor.from_pretrained("facebook/mms-" + model)
        self.processor.tokenizer.set_target_lang(self.mms_lang)
        self.sampling_rate = sampling_rate

    def generate_asr_hyp(self, speech_file):
        #TODO add conversion to wav2vec supported input format
        try:
            speech_array, sampling_rate = librosa.load(speech_file, sr=self.sampling_rate)

            inputs = self.processor(speech_array, sampling_rate=16_000, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.mms_model(**inputs).logits
            
            ids = torch.argmax(outputs, dim=-1)[0]
            hyp = self.processor.decode(ids)

            print(f"Hyp:   {hyp}")

        except Exception as e:
            print(f"Other error: {e}")
            hyp=""
        
        if hyp != "":
            self.update_cache(speech_file, hyp)
        
        return hyp