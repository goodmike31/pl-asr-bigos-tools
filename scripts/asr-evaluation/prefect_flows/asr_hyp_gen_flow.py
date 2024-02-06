from prefect import flow
from prefect_flows.tasks import load_hf_dataset_split, gen_hyps_from_audio_samples
from asr_systems import initialize_asr_system

@flow(name="ASR Hypothesis Generation Flow")
def asr_hyp_gen_flow(config_user, config_common, config_runtime):

    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]

    for system in systems:
        for model in config_runtime["systems"][system]["models"]:
            asr_system = initialize_asr_system(system, model, config_user)  # Assume this is defined
            print("ASR system initialized")
            for dataset_name in datasets:
                for subset in subsets:
                    for split in splits:
                        hf_dataset = load_hf_dataset_split(dataset_name, split, subset)
                        audio_paths = hf_dataset["audiopath_local"]
                        gen_hyps = gen_hyps_from_audio_samples(audio_paths, asr_system)
                        print(gen_hyps)
    #print(final_result)