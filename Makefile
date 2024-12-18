
# RUNTIME CONFIGURATION
EVAL_CONFIGS = bigos pelcra bigos-med bigos-diagnostic

# USER CONFIGURATION - ASR EVALUATION
EVAL_CONFIG ?= 
DATASET_SUBSET ?= 
SPLIT ?= 

# USER CONFIGURATION - TTS DATASET GENERATION
TTS_SET ?=

# HELPER VARIABLES AND SCRIPTS
TODAY=$(shell date +'%Y%m%d')
READ_INI = ./scripts/utils/	read_ini.py

# PATHS TO CONFIGURATION FILES
USER_CONFIG_FILE = ./config/user-specific/config.ini

# PATHS EXTRACTED FROM CONFIGURATION FILES
NEMO_MANIFEST_DIR = $(shell python3 read_ini.py PATHS NEMO_MANIFEST_DIR $(USER_CONFIG_FILE))

NEMO_REPO_DIR = $(shell python3 read_ini.py PATHS NEMO_REPO_DIR $(USER_CONFIG_FILE))
SDE_PATH = $(NEMO_REPO_DIR)/tools/speech_data_explorer/data_explorer.py

LOCAL_DATA_DIR = $(shell python3 read_ini.py PATHS LOCAL_DATA_DIR $(USER_CONFIG_FILE))

# PATHS TO GENERATED FILES
HYPS_STATS_FILE = $(LOCAL_DATA_DIR)/asr_hyps_cache/stats/cached_hyps_stats-$(DATASET)-$(TODAY).csv

.PHONY: eval-e2e test

test-force-hyps:
	@echo "Running tests"
	# @python -m pytest tests/
	@for runtime_config in test; do \
		echo "Running e2e eval pipeline for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --eval_config=$$runtime_config --force=True --force_hyps=True; \
	done

test:
	@echo "Running tests"
	# @python -m pytest tests/
	@for runtime_config in test; do \
		echo "Running e2e eval pipeline for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --eval_config=$$runtime_config --force=True; \
	done

eval-e2e-all-force:
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running e2e eval pipeline for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --eval_config=$$runtime_config --force=True; \
	done

eval-e2e-all:
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running e2e eval pipeline for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --eval_config=$$runtime_config; \
	done

eval-e2e-force:
	@echo "Running e2e eval pipeline"
	@python scripts/asr_eval_lib/main.py --eval_config $(EVAL_CONFIG) --force=True

eval-e2e:
	@echo "Running e2e eval pipeline"
	@python scripts/asr_eval_lib/main.py --eval_config $(EVAL_CONFIG)

all:
	@echo "Running e2e eval pipeline"
	@python scripts/asr_eval_lib/main.py --eval_config $(EVAL_CONFIG)

# Eval prep flows - ASR hyps generation
hyps-stats:
	@echo "Running hyps stats flow"
	@python scripts/asr_eval_lib/main.py --flow="HYP_STATS" --eval_config=$(EVAL_CONFIG)
	@cat $(HYPS_STATS_FILE)

hyps-stats-force:
	@echo "Running hyps stats flow"
	@python scripts/asr_eval_lib/main.py --flow="HYP_STATS" --eval_config=$(EVAL_CONFIG) --force_hyps=True
	@cat $(HYPS_STATS_FILE)

hyp-gen:
	@echo "Running hyps generation flow"
	@python scripts/asr_eval_lib/main.py --flow="HYP_GEN" --eval_config=$(EVAL_CONFIG)

hyp-gen-force:
	@echo "Running hyps generation flow"
	@python scripts/asr_eval_lib/main.py --flow="HYP_GEN" --eval_config=$(EVAL_CONFIG) --force_hyps=True


#################################################################

# Eval prep flows - ASR hyps input preparation
eval-data-prep-all-force:
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running eval prep flow for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_PREP --eval_config=$$runtime_config --force=True; \
	done
	
eval-data-prep-all:
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running eval prep flow for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_PREP --eval_config=$$runtime_config; \
	done

eval-data-prep:
	@echo "Running eval prep flow"
	@python scripts/asr_eval_lib/main.py --flow="EVAL_PREP" --eval_config=$(EVAL_CONFIG)

eval-data-prep-force:
	@echo "Running eval prep flow"
	@python scripts/asr_eval_lib/main.py --flow="EVAL_PREP" --eval_config=$(EVAL_CONFIG) --force=True

# Evaluation flows - metrics calculation
eval-scores-gen-all-force:
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running eval run flow for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$$runtime_config --force=True; \
	done

eval-scores-gen-all:
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running eval run flow for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$$runtime_config; \
	done

eval-scores-gen:
	@echo "Running eval run flow"
	@python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$(EVAL_CONFIG)

eval-scores-gen-force:
	@echo "Running eval run flow with forced generation of hyps"
	@python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$(EVAL_CONFIG) --force=True

# Eval prep - synthetic speech recordings
tts-set-gen:
	@echo "Generating synthetic test set for TTS_SET=$(TTS_SET)"
	@python scripts/tts_gen_lib/main.py --flow=TTS_SET_GEN --tts_set_config=$(TTS_SET)

sde-manifest:
	@echo "Showing manifest for DATASET_SUBSET=$(DATASET_SUBSET) SPLIT=$(SPLIT) in SDE tool"
	@python $(SDE_PATH) -a $(NEMO_MANIFEST_DIR)/$(DATASET_SUBSET)-$(SPLIT).jsonl

# Prepare dataset for manual inspection in Argilla

prep-eval-results-inspection:
	@echo "Preparing manual inspection of eval results for EVAL_CONFIG=$(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --eval_config=$(EVAL_CONFIG) --flow="PREP_EVAL_RESULTS_INSPECTION"

# Save manual inspection results in Argilla as separate HF dataset

# TODO - Save manual inspection results on Hugging Face hub

# TODO - Post-process manual inspection results to include on Leaderboard
# TODO - Automatic eval results post-processing
# TODO - preannotate ASR errors