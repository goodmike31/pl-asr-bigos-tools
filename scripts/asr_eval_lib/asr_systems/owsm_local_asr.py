from .base_asr_system import BaseASRSystem
import torch
torch.cuda.empty_cache()
import librosa
import soundfile
from espnet2.bin.s2t_inference import Speech2Text

class OWSMLocalASR(BaseASRSystem):
    """Open Whisper-style Speech Models (OWSM) ASR system implementation.
    
    Provides integration with OWSM for speech recognition.
    
    Attributes:
        s2t (Speech2Text): Primary OWSM model instance.
        s2t_cpu (Speech2Text): Fallback CPU OWSM model instance.
    """
    
    def __init__(self, system, model, language_code="pl-PL", sampling_rate=16000):
        """Initialize the OWSM ASR system.
        
        Args:
            system (str): Identifier for the ASR system type ('owsm_local').
            model (str): The specific OWSM model to use.
            language_code (str, optional): Language code. Defaults to "pl-PL".
            sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
            
        Raises:
            Warning: If GPU is not available for better inference speed.
        """
        super().__init__(system, model, language_code)
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # temporary solution to avoid crashing the GPU on local machine
        #device = "cpu"

        if not torch.cuda.is_available():
            raise Warning("Please use GPU for better inference speed.")

        #change to CPU for large models which crushes the GPU
        self.language_code = self.map_language_code(language_code)
        print("Language code: ", self.language_code)

        #default decoder for default device
        s2t = Speech2Text.from_pretrained(
            model,
            device=device,
            lang_sym=self.language_code
        )
        
        self.s2t = s2t

        # fallback CPU
        s2t_cpu = Speech2Text.from_pretrained(
            model,
            device="cpu",
            lang_sym=self.language_code
        )
        self.s2t_cpu = s2t_cpu

    def map_language_code(self, language_code):
        """Map standard language codes to OWSM-specific language codes.
        
        Args:
            language_code (str): Standard language code (e.g., "pl-PL").
            
        Returns:
            str: OWSM-specific language code (e.g., "pol").
        """
        # modify the way you map the language code
        #TODO extend to other languages
        if (self.language_code == "pl-PL"):
            language_code = "pol"

        return language_code
    

    def generate_asr_hyp(self, speech_file):
        """Generate transcription for an audio file using OWSM.
        
        Handles both short (<30s) and long (>30s) audio files appropriately.
        Falls back to CPU processing if GPU processing fails.
        
        Args:
            speech_file (str): Path to the audio file to transcribe.
            
        Returns:
            str: The transcription result.
        """
        # check audio duration
        duration = librosa.get_duration(filename=speech_file)
        if duration < 30:
            print("Audio duration is less than 30s. Normal decoding")
            
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
                print("Default device generation fail. Using CPU")
                try:
                    speech, rate = soundfile.read(speech_file)
                    result = self.s2t_cpu(speech)
                    text = result[0][-2]
                    print("text:", text)
                    hyp = text[4:]
                    print("Hyp:", hyp)

                except Exception as e:
                    print(f"Other error: {e}")
                    exit()
        else:
            print("Audio duration is more than 30s. Using decode_long function.")
            try:
                speech, rate = soundfile.read(speech_file)
                result = self.s2t.decode_long(speech)
                # given the list of tuples (start_time, end_time, text) in result, extract all text_fields and join them as single string
                text = " ".join([x[2] for x in result])

                print("text:", text)
                hyp = text[4:]
                print("Hyp:", hyp)

            except Exception as e:
                print("Default device generation fail for long audio. Using CPU")
                try:
                    speech, rate = soundfile.read(speech_file)
                    result = self.s2t_cpu.decode_long(speech)
                    text = " ".join([x[2] for x in result])
                    print("text:", text)
                    hyp = text[4:]
                    print("Hyp:", hyp)

                except Exception as e:
                    print(f"Other error: {e}")
                    exit()
                
        self.update_cache(speech_file, hyp)
        return hyp
