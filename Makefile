################################################################################
# BIGOS TOOLS - MAKEFILE
#
# This Makefile provides commands for ASR evaluation, TTS dataset generation,
# and other speech processing utilities.
################################################################################

#===============================================================================
# RUNTIME CONFIGURATION
#===============================================================================
EVAL_CONFIGS = poleval_test_a
# Additional configs: bigos pelcra bigos-med bigos-diagnostic

#===============================================================================
# USER CONFIGURATION
#===============================================================================
# ASR EVALUATION
EVAL_CONFIG ?= 
DATASET_SUBSET ?= 
SPLIT ?= 

# TTS DATASET GENERATION
TTS_SET ?=

#===============================================================================
# HELPER VARIABLES AND SCRIPTS
#===============================================================================
TODAY := $(shell date +'%Y%m%d')
READ_INI := ./scripts/utils/read_ini.py

#===============================================================================
# PATHS TO CONFIGURATION FILES
#===============================================================================
USER_CONFIG_FILE := ./config/user-specific/config.ini

#===============================================================================
# PATHS EXTRACTED FROM CONFIGURATION FILES
#===============================================================================
NEMO_MANIFEST_DIR := $(shell python3 $(READ_INI) PATHS NEMO_MANIFEST_DIR $(USER_CONFIG_FILE))
NEMO_REPO_DIR := $(shell python3 $(READ_INI) PATHS NEMO_REPO_DIR $(USER_CONFIG_FILE))
SDE_PATH := $(NEMO_REPO_DIR)/tools/speech_data_explorer/data_explorer.py
LOCAL_DATA_DIR := $(shell python3 $(READ_INI) PATHS LOCAL_DATA_DIR $(USER_CONFIG_FILE))

#===============================================================================
# PATHS TO GENERATED FILES
#===============================================================================
HYPS_STATS_FILE := $(LOCAL_DATA_DIR)/asr_hyps_cache/stats/cached_hyps_stats-$(DATASET)-$(TODAY).csv

# Declare all phony targets
.PHONY: help test test-force-hyps eval-e2e eval-e2e-all eval-e2e-force eval-e2e-all-force \
        hyps-stats hyps-stats-force hyp-gen hyp-gen-force \
        eval-data-prep eval-data-prep-force eval-data-prep-all eval-data-prep-all-force \
        eval-scores-gen eval-scores-gen-force eval-scores-gen-all eval-scores-gen-all-force \
        tts-set-gen sde-manifest prep-eval-results-inspection all

#===============================================================================
# HELP TARGET
#===============================================================================
help:
	@echo "BIGOS TOOLS MAKEFILE COMMANDS"
	@echo "-----------------------------------------------------------------------------------"
	@echo "TEST COMMANDS:"
	@echo "  test                        Run tests without forcing hypothesis regeneration"
	@echo "  test-force-hyps             Run tests with forcing hypothesis regeneration"
	@echo 
	@echo "END-TO-END EVALUATION:"
	@echo "  eval-e2e                    Run end-to-end evaluation pipeline for EVAL_CONFIG"
	@echo "  eval-e2e-force              Run forced end-to-end evaluation pipeline"
	@echo "  eval-e2e-all                Run end-to-end evaluation for all EVAL_CONFIGS"
	@echo "  eval-e2e-all-force          Run forced end-to-end evaluation for all EVAL_CONFIGS"
	@echo 
	@echo "HYPOTHESIS GENERATION:"
	@echo "  hyps-stats                  Generate statistics for ASR hypotheses"
	@echo "  hyps-stats-force            Force generation of ASR hypotheses statistics"
	@echo "  hyp-gen                     Generate ASR hypotheses"
	@echo "  hyp-gen-force               Force generation of ASR hypotheses"
	@echo 
	@echo "EVALUATION DATA PREPARATION:"
	@echo "  eval-data-prep              Prepare evaluation data"
	@echo "  eval-data-prep-force        Force preparation of evaluation data"
	@echo "  eval-data-prep-all          Prepare evaluation data for all configs"
	@echo "  eval-data-prep-all-force    Force preparation of evaluation data for all configs"
	@echo 
	@echo "EVALUATION METRICS CALCULATION:"
	@echo "  eval-scores-gen             Generate evaluation scores"
	@echo "  eval-scores-gen-force       Force generation of evaluation scores"
	@echo "  eval-scores-gen-all         Generate evaluation scores for all configs"
	@echo "  eval-scores-gen-all-force   Force generation of evaluation scores for all configs"
	@echo 
	@echo "TTS AND DATASET UTILITIES:"
	@echo "  tts-set-gen                 Generate synthetic test set for TTS_SET"
	@echo "  sde-manifest                Show manifest in Speech Data Explorer tool"
	@echo "  prep-eval-results-inspection Prepare manual inspection of evaluation results"
	@echo "-----------------------------------------------------------------------------------"
	@echo "USAGE EXAMPLE: make eval-e2e EVAL_CONFIG=poleval_test_a"

#===============================================================================
# TEST COMMANDS
#===============================================================================
test:
	@echo "Running tests"
	# @python -m pytest tests/
	@for runtime_config in test; do \
		echo "Running e2e eval pipeline for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --eval_config=$$runtime_config --force=True; \
	done

test-force-hyps:
	@echo "Running tests with forced hypothesis regeneration"
	# @python -m pytest tests/
	@for runtime_config in test; do \
		echo "Running e2e eval pipeline for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --eval_config=$$runtime_config --force=True --force_hyps=True; \
	done

#===============================================================================
# END-TO-END EVALUATION COMMANDS
#===============================================================================
eval-e2e:
	@echo "Running end-to-end evaluation pipeline for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --eval_config $(EVAL_CONFIG)

eval-e2e-force:
	@echo "Running forced end-to-end evaluation pipeline for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --eval_config $(EVAL_CONFIG) --force=True

eval-e2e-all:
	@echo "Running end-to-end evaluation for all configs ($(EVAL_CONFIGS))"
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running e2e eval pipeline for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --eval_config=$$runtime_config; \
	done

eval-e2e-all-force:
	@echo "Running forced end-to-end evaluation for all configs ($(EVAL_CONFIGS))"
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running e2e eval pipeline for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --eval_config=$$runtime_config --force=True; \
	done

all: eval-e2e

#===============================================================================
# ASR HYPOTHESES GENERATION
#===============================================================================
hyps-stats:
	@echo "Generating ASR hypotheses statistics for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --flow="HYP_STATS" --eval_config=$(EVAL_CONFIG)
	@cat $(HYPS_STATS_FILE)

hyps-stats-force:
	@echo "Forcing generation of ASR hypotheses statistics for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --flow="HYP_STATS" --eval_config=$(EVAL_CONFIG) --force_hyps=True
	@cat $(HYPS_STATS_FILE)

hyp-gen:
	@echo "Generating ASR hypotheses for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --flow="HYP_GEN" --eval_config=$(EVAL_CONFIG)

hyp-gen-force:
	@echo "Forcing generation of ASR hypotheses for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --flow="HYP_GEN" --eval_config=$(EVAL_CONFIG) --force_hyps=True

#===============================================================================
# ASR EVALUATION DATA PREPARATION
#===============================================================================
eval-data-prep:
	@echo "Preparing evaluation data for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --flow="EVAL_PREP" --eval_config=$(EVAL_CONFIG)

eval-data-prep-force:
	@echo "Forcing preparation of evaluation data for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --flow="EVAL_PREP" --eval_config=$(EVAL_CONFIG) --force=True

eval-data-prep-all:
	@echo "Preparing evaluation data for all configs ($(EVAL_CONFIGS))"
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running eval prep flow for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_PREP --eval_config=$$runtime_config; \
	done

eval-data-prep-all-force:
	@echo "Forcing preparation of evaluation data for all configs ($(EVAL_CONFIGS))"
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running eval prep flow for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_PREP --eval_config=$$runtime_config --force=True; \
	done

#===============================================================================
# EVALUATION METRICS CALCULATION
#===============================================================================
eval-scores-gen:
	@echo "Generating evaluation scores for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$(EVAL_CONFIG)

eval-scores-gen-force:
	@echo "Forcing generation of evaluation scores for $(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$(EVAL_CONFIG) --force=True

eval-scores-gen-all:
	@echo "Generating evaluation scores for all configs ($(EVAL_CONFIGS))"
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running eval run flow for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$$runtime_config; \
	done

eval-scores-gen-all-force:
	@echo "Forcing generation of evaluation scores for all configs ($(EVAL_CONFIGS))"
	@for runtime_config in $(EVAL_CONFIGS); do \
		echo "Running eval run flow for runtime_config $$runtime_config"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$$runtime_config --force=True; \
	done

#===============================================================================
# TTS AND DATASET UTILITIES
#===============================================================================
tts-set-gen:
	@echo "Generating synthetic test set for TTS_SET=$(TTS_SET)"
	@python scripts/tts_gen_lib/main.py --flow=TTS_SET_GEN --tts_set_config=$(TTS_SET)

sde-manifest:
	@echo "Showing manifest for DATASET_SUBSET=$(DATASET_SUBSET) SPLIT=$(SPLIT) in Speech Data Explorer tool"
	@python $(SDE_PATH) -a $(NEMO_MANIFEST_DIR)/$(DATASET_SUBSET)-$(SPLIT).jsonl

prep-eval-results-inspection:
	@echo "Preparing manual inspection of evaluation results for EVAL_CONFIG=$(EVAL_CONFIG)"
	@python scripts/asr_eval_lib/main.py --eval_config=$(EVAL_CONFIG) --flow="PREP_EVAL_RESULTS_INSPECTION"

#===============================================================================
# TODO ITEMS
#===============================================================================
# TODO - Save manual inspection results on Hugging Face hub
# TODO - Post-process manual inspection results to include on Leaderboard
# TODO - Automatic eval results post-processing
# TODO - preannotate ASR errors