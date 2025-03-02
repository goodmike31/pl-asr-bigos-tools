# BIGOS - Benchmark for Polish ASR Systems

<div align="center">
  <a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/69bddcea866e8210cf483769841282dd-Abstract-Datasets_and_Benchmarks_Track.html"><img src="https://upload.wikimedia.org/wikipedia/en/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg" alt="NeurIPS"></a>
  <a href="https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2"><img src="https://img.shields.io/badge/Dataset-%F0%9F%A4%97%20Hugging_Face-yellow" alt="Hugging Face"></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/Data%20License-CC_BY_NC_SA_4.0-blue" alt="Data License"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/Code%20License-MIT-blue" alt="Code License"></a>
</div>

## News
- [02/25/2025] 🚀 Added support for OWSM (Open Whisper-style Speech Models) ASR system
- [12/10/2024] 🚀 Published BIGOS benchmark paper in NeurIPS Datasets and Benchmarks Track
- [12/01/2024] 🚀 Released updated PolEval evaluation results on the [Polish ASR leaderboard](https://huggingface.co/spaces/amu-cai/pl-asr-leaderboard)

## Overview
BIGOS (Benchmark Intended Grouping of Open Speech) is a framework for evaluating Automatic Speech Recognition (ASR) systems on Polish language datasets. It provides tools for:
- Curating speech datasets in a standardized format
- Generating ASR transcriptions from various engines (commercial and open-source)
- Evaluating transcription quality with standard metrics
- Visualizing and analyzing results

> **Key Benefits**: BIGOS standardizes evaluation across multiple ASR systems and datasets, enabling fair comparison and quantitative analysis of ASR performance on Polish speech.

## Installation

### Prerequisites
- Python 3.10+
- Required system packages:
  ```bash
  sudo apt-get install sox ffmpeg  # Ubuntu/Debian
  brew install sox ffmpeg          # macOS
  ```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pl-asr-bigos-tools.git
   cd pl-asr-bigos-tools
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment:
   - Copy `config/user-specific/template.ini` to `config/user-specific/config.ini`
   - Edit the file with your API keys and paths
   - Validate your configuration with:
     ```bash
     make test-force-hyp
     make test
     ```

## Usage

### Evaluating ASR Systems
The main functionality is accessible through the Makefile:

```bash
# Run evaluation on BIGOS dataset
make eval-e2e EVAL_CONFIG=bigos

# Run evaluation on PELCRA dataset
make eval-e2e EVAL_CONFIG=pelcra

# Generate hypotheses for a specific configuration
make hyp-gen EVAL_CONFIG=bigos

# Calculate statistics for cached hypotheses
make hyps-stats EVAL_CONFIG=bigos

# Force regeneration of evaluation data
make eval-e2e-force EVAL_CONFIG=bigos
```

### Project Architecture

The BIGOS benchmark system follows a modular architecture:

1. **Dataset Management**: Curated datasets in BIGOS format
2. **ASR Systems**: Standardized interface for diverse ASR engines
3. **Hypothesis Generation**: Processing audio through ASR systems
4. **Evaluation**: Calculating metrics and generating reports
5. **Analysis**: Tools for visualizing and interpreting results

The evaluation workflow consists of the following stages:
1. **Preparation**: Loading datasets and preparing processing pipelines
2. **Hypothesis Generation**: Creating transcriptions using specified ASR systems
3. **Evaluation**: Calculating metrics like WER, CER, MER, etc.
4. **Analysis**: Reporting and visualization of results

### Adding New ASR Systems
1. Create a new class in `scripts/asr_eval_lib/asr_systems/` based on the template
2. Register your system in `scripts/asr_eval_lib/asr_systems/__init__.py`
3. Update configuration files in `config/eval-run-specific/`

Example of registering a new ASR system:
```python
# In scripts/asr_eval_lib/asr_systems/__init__.py
from .your_new_asr_system import YourNewASRSystem

def asr_system_factory(system, model, config):
    # Existing code...
    
    elif system == 'your_system':
        # Configuration for your new system
        return YourNewASRSystem(system, model, other_params)
    
    # More systems...
```

### Adding New Datasets
1. Open an existing config file (e.g., `config/eval-run-specific/bigos.json`)
2. Save a modified version as `config/eval-run-specific/<dataset_name>.json`
3. Ensure your dataset follows the BIGOS format and is publicly available
4. Run the evaluation with:
   ```bash
   make eval-e2e EVAL_CONFIG=<dataset_name>
   ```

### Generating TTS Synthetic Test Sets
To generate a synthetic test set:
```bash
make tts-set-gen TTS_SET=<tts_set_name>
```
Replace `<tts_set_name>` with the appropriate configuration name (e.g., `amu-med-all`).

### Displaying Manifest in Nemo SDE Tool
To display a manifest for a specific dataset and split:
```bash
make sde-manifest DATASET_SUBSET=<subset_name> SPLIT=<split_name>
```

## Project Structure
- `config/` - Configuration files
  - `common/` - Shared configuration 
  - `eval-run-specific/` - ASR evaluation configuration
  - `tts-set-specific/` - TTS generation configuration
  - `user-specific/` - User-specific settings (API keys, paths)
- `scripts/` - Main implementation code
  - `asr_eval_lib/` - ASR evaluation framework
    - `asr_systems/` - ASR system implementations
    - `eval_utils/` - Evaluation metrics and utilities
    - `prefect_flows/` - Prefect workflow definitions
  - `tts_gen_lib/` - Speech synthesis for test data
  - `utils/` - Common utilities
- `data/` - Working directory for datasets and results (gitignored)

## Supported ASR Systems
The benchmark currently supports the following ASR systems:
- Google Cloud Speech-to-Text (v1 and v2)
- Microsoft Azure Speech-to-Text
- OpenAI Whisper (Cloud and Local)
- AssemblyAI
- NVIDIA NeMo
- Facebook MMS
- Facebook Wav2Vec
- OWSM (Open Whisper-style Speech Models)

## Datasets
The framework is designed to work with datasets in the BIGOS format:
- [BIGOS V2](https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2) - Primarily read speech
- [PELCRA for BIGOS](https://huggingface.co/datasets/pelcra/pl-asr-pelcra-for-bigos) - Primarily conversational speech

## Troubleshooting

### Common Issues

- **API Key Access**: If encountering authentication errors, verify your API keys in `config.ini`
- **Missing Dependencies**: If experiencing import errors, run `pip install -r requirements.txt`
- **Permission Issues**: For file access errors, check directory permissions in your configuration
- **Disk Space**: ASR hypothesis caching requires substantial disk space; monitor usage in the `data/` directory

## Project Roadmap

The following TODO items represent ongoing development priorities:

### Documentation
- [ ] Add detailed docstrings to all classes and functions
- [ ] Create a comprehensive API reference
- [ ] Add examples for extending with new metrics
- [ ] Document the data format specification in detail

### Code Quality
- [ ] Add type hints to improve code readability and IDE support
- [ ] Implement more robust error handling in ASR system implementations
- [ ] Add logging throughout the codebase (replace print statements)
- [ ] Standardize configuration approach (choose either JSON or INI consistently)

### Features
- [ ] Add support for new ASR systems (e.g., Meta Seamless, Amazon Transcribe)
- [ ] Implement additional evaluation metrics (e.g., semantic metrics)
- [ ] Create a web interface for results visualization
- [ ] Add support for languages beyond Polish
- [ ] Implement audio preprocessing options (e.g., noise reduction, normalization)

### Testing
- [ ] Expand test coverage for core components
- [ ] Add integration tests for complete evaluation flows
- [ ] Create fixtures for testing without API access

### Infrastructure
- [ ] Containerize the application with Docker
- [ ] Create a CI/CD pipeline for automated testing
- [ ] Implement a proper Python package structure
- [ ] Add infrastructure for distributed processing

## Contributing
Contributions to BIGOS are welcome! Please see [DEVELOPER.md](DEVELOPER.md) for guidance.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Citation
If you use this benchmark in your research, please cite:
```
@inproceedings{NEURIPS2024_69bddcea,
 author = {Junczyk, Micha\l },
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {57439--57471},
 publisher = {Curran Associates, Inc.},
 title = {BIGOS V2 Benchmark for Polish ASR: Curated Datasets and Tools for Reproducible Evaluation},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/69bddcea866e8210cf483769841282dd-Paper-Datasets_and_Benchmarks_Track.pdf},
 volume = {37},
 year = {2024}
}
```
