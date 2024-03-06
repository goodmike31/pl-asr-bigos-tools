from prefect import flow
from prefect_flows.tasks import load_hf_dataset_split, gen_hyps_from_audio_samples
from asr_systems import initialize_asr_system

@flow(name="ASR Hypothesis Generation Flow")
def asr_hyp_gen(config_user, config_common, config_runtime):

    datasets = config_runtime["datasets"]
    subsets = config_runtime["subsets"]
    splits = config_runtime["splits"]
    systems = config_runtime["systems"]
    max_samples_per_subset = config_runtime["max_samples_per_subset"]

    for system in systems:
        for model in config_runtime["systems"][system]["models"]:
            asr_system = initialize_asr_system(system, model, config_user)  # Assume this is defined
            print("ASR system initialized")
            for dataset_name in datasets:
                for subset in subsets:
                    for split in splits:
                        hf_dataset = load_hf_dataset_split(dataset_name, split, subset)
                        audio_paths = hf_dataset["audiopath_local"]
                        # limit the number of audio paths for testing
                        audio_paths = audio_paths[:max_samples_per_subset]
                        gen_hyps = gen_hyps_from_audio_samples(audio_paths, asr_system)
                        print("Generated or retrieved hypotheses for {} samples for subset {} and split {}".format(len(gen_hyps), subset, split) )