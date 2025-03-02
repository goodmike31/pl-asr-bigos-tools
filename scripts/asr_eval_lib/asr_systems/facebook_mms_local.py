from .base_asr_system import BaseASRSystem
from transformers import Wav2Vec2ForCTC, AutoProcessor
import librosa
import torch    

#https://huggingface.co/docs/transformers/v4.36.1/model_doc/mms

lang_code_693_3 = {
    "pl-PL":"pol"
    }

class FacebookMMS(BaseASRSystem):
    """Facebook MMS (Massively Multilingual Speech) ASR system implementation.
    
    Provides integration with Facebook's MMS models for speech recognition.
    
    Attributes:
        mms_lang (str): ISO-639-3 language code for MMS model.
        mms_model (Wav2Vec2ForCTC): The loaded MMS model.
        processor (AutoProcessor): Processor for the MMS model.
        sampling_rate (int): Audio sampling rate.
    """
    
    def __init__(self, system, model, language_code="pl-PL", sampling_rate=16000):
        """Initialize the Facebook MMS ASR system.
        
        Args:
            system (str): Identifier for the ASR system type ('mms').
            model (str): The specific MMS model to use.
            language_code (str, optional): Language code. Defaults to "pl-PL".
            sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
        """
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
        """Generate transcription for an audio file using Facebook MMS.
        
        Args:
            speech_file (str): Path to the audio file to transcribe.
            
        Returns:
            str: The transcription result.
        """
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