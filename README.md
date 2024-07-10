# pl-asr-bigos-tools
This repository contains tools for benchmarking ASR systems using BIGOS corpora. <br>

[How to use](#how-to-use)
# Design considerations

| Aspect              | Considerations                                                                            |
|---------------------|-------------------------------------------------------------------------------------------|
| **Metrics**         | Support for well-established ASR evaluation metrics.                                      |
| **Extensibility**   | Straightforward integration of new datasets, normalization methods, metrics, and new ASR systems. |
| **Availability**    | Publicly accessible and intuitive presentation of results. (see [Polish ASR leaderboard](https://huggingface.co/spaces/amu-cai/pl-asr-leaderboard))                                |
| **Comprehensiveness** | Performance analysis across scenarios, system params, and user groups.                  |
### Table 1: Design considerations for BIGOS evaluation tools

# About BIGOS corpora
BIGOS (Benchmark Intended Grouping of Open Speech) corpora aims at simplifying the access and use of publicly available ASR speech datasets.<br>
Currently BIGOS corpora is available for Polish language.<br>
Two BIGOS corpora types are available at the Hugging Face platform: <br>
* [BIGOS V2](https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2) - containing mostly read speech.<br>
* [PELCRA for BIGOS](https://huggingface.co/datasets/pelcra/pl-asr-pelcra-for-bigos) - containing mostly conversational speech<br>

## Relevant work
[BIGOS V2](https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2) and [PELCRA for BIGOS](https://huggingface.co/datasets/pelcra/pl-asr-pelcra-for-bigos)c* Evaluate publicly available ASR systems - [Polish ASR leaderboard](https://huggingface.co/spaces/amu-cai/pl-asr-leaderboard) .<br>
* Evaluate community-provided ASR systems - [2024 PolEval challenge](https://beta.poleval.pl/gonito/challenge/2024-asr-bigos).<br>

## Relevant publications
* [BIGOS V1](https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos) [paper](https://annals-csis.org/proceedings/2023/drp/pdf/1609.pdf)<br>
* [BIGOS V2](https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2) [paper](TODO arxiv)<br>

# How to Use
This section provides instructions on how to use the provided Makefile for various runtime configurations and evaluation tasks. 
Please ensure all necessary configurations and dependencies are set up before proceeding.

### Prerequisites
Ensure you have the following prerequisites:
- Python 3.x
- Required Python packages (install via `requirements.txt`)

### Configuration
You need to provide user-specific configuration e.g. Cloud API keys.
To do so edit "template.ini" and save as "config.ini" (`./config/user-specific/config.ini`)
ll.

To validate if configuration is valid, run: make test

### Use-cases
#### Running all evaluation steps
To run all evaluation steps for a specific eval_config:
make eval-e2e EVAL_CONFIG=<eval_config_name>

To run the evaluation for all eval_configs:
make eval-e2e-all

#### Running specific evaluation step
##### Generate ASR hypotheses for specific runtime config


##### Generate report about ASR hypotheses 
make hyps-stats EVAL_CONFIG=<eval_config_name>

#####

#####

### Forced processing
By default, if specific intermediary results exists, the processing is skipped.
To force regeneration of hypotheses, evaluation scores calculation etc, completent the command with "force"
For example:
To force the evaluation for all eval_configs:
make eval-e2e-all-force

To force the evaluation for a specific eval_config:
make eval-e2e-force EVAL_CONFIG=<eval_config_name>

### Replicating BIGOS V2 benchmark results 
To run evaluation for BIGOS V2 dataset run:
make eval-e2e EVAL_CONFIG=bigos
To replicate exact results, contact micjun@amu.edu.pl to obtain copy of ASR hypotheses.

### Replicating PELCRA for BIGOS benchmark results
To run evaluation for PELCRA for BIGOS dataset run:
make eval-e2e EVAL_CONFIG=pelcra
To replicate exact results, contact micjun@amu.edu.pl to obtain copy of ASR hypotheses.

### Runtime Configuration Creation/Modification
You can run evaluation for various datasets, systems, normalization methods etc.
To add new or edit existing runtime configuration go to "config/eval-scores-gen-specific" folder and add/edit relevant file.

### Adding new ASR system to the BIGOS benchmark
See exemplary implementations of ASR systems classes in scripts/asr_eval_lib/asr_systems.
Create new file with the implementation of new ASR system based on "base_asr_system.py" class.
Add reference to the new ASR system in the "scripts/asr_eval_lib/asr_systems/__init__.py".

### Adding new dataset to the BIGOS benchmark
Open existing config for already supported dataset e.g. "config/eval-scores-gen-specific/bigos.json
Modify it and save as the new configuration as "config/eval-scores-gen-specific/<dataset_name>.json".
Make sure that new dataset follows the BIGOS format and is publicly available.
To run the evaluation for new dataset:
make eval-e2e EVAL_CONFIG=<dataset_name>
### Generating TTS Synthetic Test Set 
To generate a synthetic test set:
make tts-set-gen TTS_SET=<tts_set_name>
Replace <tts_set_name> with the appropriate values for your use case.

### Displaying Manifest in Nemo SDE Tool
To display a manifest for a specific dataset and split:
make sde-manifest DATASET=<dataset_name> SPLIT=<split_name>
Replace <dataset_name>, and <split_name>  with the appropriate values for your use case.

