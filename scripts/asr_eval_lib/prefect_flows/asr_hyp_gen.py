from prefect import flow
from prefect_flows.tasks import load_hf_dataset_split, gen_hyps_from_audio_samples
from asr_systems import initialize_asr_system

@flow(name="ASR Hypothesis Generation Flow")
def asr_hyp_gen(config_user, config_common, config_runtime, force_hyps=False):

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
                        print("Loading dataset: {} \nsplit: {}\n subset: {}".format(dataset_name, split, subset))
                        try:
                            hf_dataset = load_hf_dataset_split(dataset_name, subset, split)
                        except Exception as e:
                            print("Failed to load dataset {} \nsplit: {}\n subset: {}\n with error: {}".format(dataset_name, split, subset, e))
                            print("Trying force download")
                            hf_dataset = load_hf_dataset_split(dataset_name, subset,  split, force_download=True)
                            exit(1)
                        print("Loaded dataset {} \nsplit: {}\n subset: {}".format(dataset_name, split, subset))
                        print("Number of samples in dataset: ", len(hf_dataset))
                        
                        audio_paths = hf_dataset["audiopath_local"]
                        # limit the number of audio paths for testing
                        audio_paths = audio_paths[:max_samples_per_subset]
                        gen_hyps = gen_hyps_from_audio_samples(audio_paths, asr_system, force_hyps)
                        print("Generated or retrieved hypotheses for {} samples for subset: {}\n and split: {}\n".format(len(gen_hyps), subset, split) )