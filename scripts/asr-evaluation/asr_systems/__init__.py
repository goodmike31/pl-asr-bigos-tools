
from .base_asr_system import BaseLocalASRSystem, BaseCloudASRSystem
from .test_local import TestLocalASRSystem
from .test_cloud import TestCloudASRSystem
from .cloud_asr_systems import GoogleCloudASR, AzureCloudASR

def asr_system_factory(system_type, model, config):
    if system_type == 'specific_local':
        return TestLocalASRSystem(config)
    elif system_type == 'specific_cloud':
        return TestCloudASRSystem(config)
    elif system_type == 'google':
        #def __init__(self, credentials_path, language_code = "pl-PL", model="default", enable_automatic_punctuation=True):
        google_api_key_path = config.get("API_KEYS", "GOOGLE_API_KEY_FILE")
        print("google_api_key_path", google_api_key_path)
        return GoogleCloudASR(google_api_key_path)
    elif system_type == 'azure':
        return AzureCloudASR(config)
    else:
        raise ValueError(f"Unknown ASR system type: {system_type}")

def initialize_asr_system(system, model, config_file):
    return asr_system_factory(system, model, config_file)