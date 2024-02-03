
from .base_asr_system import BaseASRSystem
from .google_cloud_asr import GoogleCloudASR
from .azure_cloud_asr import AzureCloudASR
from .whisper_cloud_asr import WhisperCloudASR
from .whisper_local_asr import WhisperLocalASR
from .facebook_mms_local import FacebookMMS
from .facebook_wav2vec import FacebookWav2Vec

def initialize_asr_system(system, model, config_file):
    return asr_system_factory(system, model, config_file)

def asr_system_factory(system, model, config):
    if system == 'google':
        google_api_key_path = config.get("API_KEYS", "GOOGLE_API_KEY_FILE")
        return GoogleCloudASR(system, model, google_api_key_path)
    
    elif system == 'azure':
        azure_api_key_path = config.get("API_KEYS", "AZURE_API_KEY")
        azure_region = config.get("CLOUD_ASR_SETTINGS", "AZURE_REGION")
        return AzureCloudASR(system, model, azure_api_key_path, azure_region)
    
    elif system == 'whisper_cloud':
        openai_api_key = config.get("API_KEYS", "WHISPER_API_KEY")
        return WhisperCloudASR(system, model, openai_api_key)
    
    elif system == 'whisper_local':
        return WhisperLocalASR(system, model)
    
    elif system == 'mms':
        return FacebookMMS(system, model)
    
    elif system == 'wav2vec2':
        return FacebookWav2Vec(system, model)

    else:
        raise ValueError(f"Unknown ASR system type: {system}")