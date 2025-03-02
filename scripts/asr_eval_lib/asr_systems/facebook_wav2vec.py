from .base_asr_system import BaseASRSystem
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

class FacebookWav2Vec(BaseASRSystem):
    """Facebook Wav2Vec2 ASR system implementation for the BIGOS framework.
    
    Provides integration with Facebook's Wav2Vec2 models for speech recognition.
    
    Attributes:
        w2v_processor (Wav2Vec2Processor): Processor for the Wav2Vec2 model.
        w2v_model (Wav2Vec2ForCTC): The loaded Wav2Vec2 model.
        sampling_rate (int): Audio sampling rate.
    """
    
    def __init__(self, system, model, language_code="pl-PL", sampling_rate=16000):
        """Initialize the Facebook Wav2Vec2 ASR system.
        
        Args:
            system (str): Identifier for the ASR system type ('wav2vec2').
            model (str): The specific Wav2Vec2 model to use.
            language_code (str, optional): Language code. Defaults to "pl-PL".
            sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
            
        Raises:
            ValueError: If an unsupported model is specified.
        """
        super().__init__(system, model, language_code)
        # convert ISO-639-1 to ISO-639-3
        self.model = model
        # TODO - move max audio length to process param to user-specific asr-system related config. Default value = 30
        self.max_audio_length_to_process_sec = 25
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
        """Generate transcription for an audio file using Facebook Wav2Vec2.
        
        Args:
            speech_file (str): Path to the audio file to transcribe.
            
        Returns:
            str: The transcription result.
        """
        try:
            speech_array, sampling_rate = librosa.load(speech_file, sr=self.sampling_rate)
            #print("Speech array length: ", len(speech_array))
            inputs = self.w2v_processor(speech_array, sampling_rate=16_000, return_tensors="pt")
            #print("Input read")
            #print("Input type: ", type(inputs))
            #outputs = self.w2v_model(inputs).logits
            try:
                outputs = self.w2v_model(**inputs).logits
            except Exception as e:
                print(f"Error generating outputs: {e}. Skipping generation and returing empty hypothesis.")
                return ""
            #print("Outputs generated")
            ids = torch.argmax(outputs, dim=-1)[0]
            #print("IDS generated")

            # add error handling for decoding
            hyp = self.w2v_processor.decode(ids)
            print(f"Hyp:   {hyp}")
        
        except Exception as e:
            print(f"Other error: {e}")
            hyp=""

        #if hypothesis is not empty, update cache
        if hyp != "":
            self.update_cache(speech_file, hyp)
                
        return hyp