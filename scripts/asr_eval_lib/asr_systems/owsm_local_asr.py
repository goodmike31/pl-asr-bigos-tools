from .base_asr_system import BaseASRSystem
import torch
torch.cuda.empty_cache()
import librosa
import soundfile
from espnet2.bin.s2t_inference import Speech2Text

class OWSMLocalASR(BaseASRSystem):
    def __init__(self, system, model, language_code:str = "pl-PL",sampling_rate:int = 16000) -> None:
        super().__init__(system, model, language_code)
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # temporary solution to avoid crashing the GPU on local machine
        device = "cpu"

        if not torch.cuda.is_available():
            raise Warning("Please use GPU for better inference speed.")

        #change to CPU for large models which crushes the GPU
        #if model in [] 
        self.language_code = self.map_language_code(language_code)
        print("Language code: ", self.language_code)


        s2t = Speech2Text.from_pretrained(
            model,
            device=device,
            lang_sym=self.language_code
        )
        self.s2t = s2t

    def map_language_code(self, language_code):
        # modify the way you map the language code
        #TODO extend to other languages
        if (self.language_code == "pl-PL"):
            language_code = "pol"

        return language_code
    

    def generate_asr_hyp(self, speech_file):
        try:
            print ("Generate ASR hyp function for ASR system: {} .\n Generating hypothesis for: {}".format(self.codename, speech_file))

            speech, rate = soundfile.read(speech_file)
            result = self.s2t(speech)
            text = result[0][-2]
            print("text:", text)
            
            # remove token with language code from the hypothesis
            hyp = text[4:]

            print("Hyp:", hyp)
        except Exception as e:
            print(f"Other error: {e}")
            exit()
        
        self.update_cache(speech_file, hyp)
        return hyp