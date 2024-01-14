from prefect import task
from datasets import load_dataset

@task
def load_config(config_path):
    # Implement your config loading logic
    pass
    print("Loading config from {}".format(config_path))

@task
def process_audio_sample(sample, asr_system):
    print("Processing audio sample {} for ASR system {}".format(sample, asr_system))
    # Implement logic to process a single audio sample
    asr_hyp = asr_system.process_audio(sample)
    return(asr_hyp)
    

@task
def load_hf_dataset(dataset_name, subset="all"):
    hf_dataset = load_dataset(dataset_name, subset)
    return hf_dataset


@task
def select_split_of_dataset(dataset, split):
    dataset = dataset[split]
    return dataset

@task
def select_subset_of_dataset(dataset, subset):
    dataset = dataset[subset]
    return dataset

@task
def save_results(results):
    # Implement logic to save results
    pass

