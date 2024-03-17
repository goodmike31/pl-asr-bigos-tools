from .azure_cloud_tts import AzureCloudTTS

def initialize_tts_system(system, model, config_file):
    return tts_system_factory(system, model, config_file)

def tts_system_factory(system, model, config):
    if system == 'azure':
        azure_api_key_path = config.get("CREDENTIALS", "AZURE_API_KEY")
        azure_region = config.get("CLOUD_ASR_SETTINGS", "AZURE_REGION")
        return AzureCloudTTS(system, model, azure_api_key_path, azure_region)
    
    else:
        raise ValueError(f"Unknown TTS system type: {system}")