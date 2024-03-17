from .base_tts_systems import BaseTTSSystem
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason, CancellationReason
from azure.cognitiveservices.speech.audio import AudioOutputConfig
import os

class AzureCloudTTS(BaseTTSSystem):
    def __init__(self, system, voice, credentials:str, region:str, language_code:str = "pl-PL", sampling_rate:int = 16000) -> None:
        super().__init__(system, voice, language_code)

        # Set up the speech sdk configuration
        self.speech_config = SpeechConfig(subscription=credentials, region=region)
        # supported voices: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=stt#prebuilt-neural-voices

        self.speech_config.speech_synthesis_voice_name = self.voice

        
    def process_prompt(self, audio_filename:str, prompt_text:str) -> str:
        audio_config = AudioOutputConfig(filename=audio_filename)
        speech_synthesizer = SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

        speech_synthesis_result = speech_synthesizer.speak_text_async(prompt_text).get()

        if speech_synthesis_result.reason == ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}]".format(prompt_text))
        elif speech_synthesis_result.reason == ResultReason.Canceled:
            cancellation_details = speech_synthesis_result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == CancellationReason.Error:
                if cancellation_details.error_details:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")

        print("Speech synthesis status: ", speech_synthesis_result.reason)
        print("Speech synthesized for text [{}]".format(prompt_text))
        print(f'Audio content written to file "{audio_filename}"')

        print(speech_synthesis_result)
        del speech_synthesizer
