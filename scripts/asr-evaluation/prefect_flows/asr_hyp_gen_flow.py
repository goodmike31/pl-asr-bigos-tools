from prefect import flow
from prefect_flows.tasks import load_config, load_hf_dataset, select_split_of_dataset, select_subset_of_dataset, gen_hyps_from_audio_samples, save_results
from asr_systems import initialize_asr_system

@flow(name="ASR Processing Flow")
def asr_hyp_gen_flow(config_user, config_common, datasets_to_process, subsets, splits, systems, models):

    for system in systems:
        for model in models[system]:
            asr_system = initialize_asr_system(system, model, config_user)  # Assume this is defined
            print("ASR system initialized")
            print(asr_system)
            for dataset_name in datasets_to_process:
                for subset in subsets:
                    hf_dataset = load_hf_dataset(dataset_name, subset)
                    print(hf_dataset)
                    for split in splits:
                        hf_dataset = select_split_of_dataset(hf_dataset, split)
                        print(hf_dataset)
                        audio_paths = hf_dataset["audiopath_local"]
                        gen_hyps = gen_hyps_from_audio_samples(audio_paths, asr_system)
                        print(gen_hyps)
    #print(final_result)