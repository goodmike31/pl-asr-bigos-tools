
from .base_asr_system import BaseLocalASRSystem, BaseCloudASRSystem
from .test_local import TestLocalASRSystem
from .test_cloud import TestCloudASRSystem
from .cloud_asr_systems import GoogleCloudASR, AzureCloudASR

def asr_system_factory(system, model, config):
    if system == 'specific_local':
        return TestLocalASRSystem(config)
    elif system == 'specific_cloud':
        return TestCloudASRSystem(config)
    elif system == 'google':
        google_api_key_path = config.get("API_KEYS", "GOOGLE_API_KEY_FILE")
        return GoogleCloudASR(system, model, google_api_key_path)
    elif system == 'azure':
        azure_api_key_path = config.get("API_KEYS", "AZURE_API_KEY")
        azure_region = config.get("CLOUD_ASR_SETTINGS", "AZURE_REGION")
        return AzureCloudASR(system, model, azure_api_key_path, azure_region)
    else:
        raise ValueError(f"Unknown ASR system type: {system}")

def initialize_asr_system(system, model, config_file):
    return asr_system_factory(system, model, config_file)