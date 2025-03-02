
from .base_asr_system import BaseASRSystem
from .google_cloud_asr import GoogleCloudASR
from .google_cloud_asr_v2 import GoogleCloudASRV2
from .azure_cloud_asr import AzureCloudASR
from .whisper_cloud_asr import WhisperCloudASR
from .whisper_local_asr import WhisperLocalASR
from .facebook_mms_local import FacebookMMS
from .facebook_wav2vec import FacebookWav2Vec
from .nvidia_nemo_asr import NvidiaNemoASR
from .assembly_ai_asr import AssemblyAIASR

# failing when running locally (CUDA error)
#from .owsm_local_asr import OWSMLocalASR

# if you added a new ASR system, import it here

def initialize_asr_system(system, model, config_file):
    return asr_system_factory(system, model, config_file)

def asr_system_factory(system, model, config):
    if system == 'google':
        google_api_key_path = config.get("CREDENTIALS", "GOOGLE_API_KEY_FILE")
        return GoogleCloudASR(system, model, google_api_key_path)
    
    elif system == 'google_v2':
        google_api_key_path = config.get("CREDENTIALS", "GOOGLE_API_KEY_FILE")
        project_id = config.get("CREDENTIALS", "GOOGLE_PROJECT_ID")
        return GoogleCloudASRV2(system, model, google_api_key_path, project_id)
    
    elif system == 'azure':
        azure_api_key_path = config.get("CREDENTIALS", "AZURE_API_KEY")
        azure_region = config.get("CLOUD_ASR_SETTINGS", "AZURE_REGION")
        return AzureCloudASR(system, model, azure_api_key_path, azure_region)
    
    elif system == 'whisper_cloud':
        openai_api_key = config.get("CREDENTIALS", "WHISPER_API_KEY")
        return WhisperCloudASR(system, model, openai_api_key)

    elif system == 'assembly_ai':
        assemblyai_api_key = config.get("CREDENTIALS", "ASSEMBLYAI_API_KEY")
        return AssemblyAIASR(system, model, assemblyai_api_key)
        
    elif system == 'whisper_local':
        return WhisperLocalASR(system, model)
    
    elif system == 'mms':
        return FacebookMMS(system, model)
    
    elif system == 'wav2vec2':
        return FacebookWav2Vec(system, model)
    
    elif system == 'nemo':
        return NvidiaNemoASR(system, model)

    # Failiing when running locally (CUDA error)
    #elif system == 'owsm_local':
    #    return OWSMLocalASR(system, model)
    
    # Add your ASR system here
       
    else:
        raise ValueError(f"Unknown ASR system type: {system}")