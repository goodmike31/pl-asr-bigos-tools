from prefect import flow
from prefect_flows.tasks import load_config, load_hf_dataset, select_split_of_dataset, select_subset_of_dataset, process_audio_sample, save_results
from asr_systems import initialize_asr_system

@flow(name="ASR Processing Flow")
def asr_hyp_gen_flow(config_user, config_common, datasets_to_process, subsets, splits, systems, models):

    for system in systems:
        for model in models:
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
                        # TODO add filtering based on different views? Or is it better to filter results after hypothesis generation?
                        generated_hyps = []
                        for audiopath in hf_dataset["audiopath_local"]:
                            print("Processing sample {}".format(audiopath))
                            result = process_audio_sample(audiopath, asr_system)
                            print(result)
                            generated_hyps.append(result)
                        print(generated_hyps)
                        #final_result = save_results(generated_hyps)
    #print(final_result)