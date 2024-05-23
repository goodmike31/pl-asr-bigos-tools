
# BIGOS PELCRA or TEST
PROJECTS=TEST BIGOS PELCRA AMU-MED DIAGNOSTIC
PROJECT ?= 
TTS_SET ?=
TODAY=$(shell date +'%Y%m%d')

HYPS_STATS_FILE = ./data/asr_hyps_cache/stats/cached_hyps_stats-$(PROJECT)-$(TODAY).csv
SDE_PATH = ../../github/NeMo/tools/speech_data_explorer/data_explorer.py
MANIFESTS_DIR = ./data/manifests

DATASET = 
SPLIT = 

.PHONY: eval-e2e run-tests hyps-stats


test-e2e:
	@echo "Running e2e pipeline on test configuration"
	@python scripts/asr_eval_lib/main.py

eval-e2e:
	@echo "Running e2e eval pipeline"
	@python scripts/asr_eval_lib/main.py --eval_config $(PROJECT)

all:
	@echo "Running e2e eval pipeline"
	@python scripts/asr_eval_lib/main.py --eval_config $(PROJECT)

# Eval prep flows - ASR hyps generation
hyps-stats:
	@echo "Running hyps stats flow"
	@python scripts/asr_eval_lib/main.py --flow="HYP_STATS" --eval_config=$(PROJECT)
	@cat $(HYPS_STATS_FILE)

hyp-gen:
	@echo "Running hyps generation flow"
	@python scripts/asr_eval_lib/main.py --flow="HYP_GEN" --eval_config=$(PROJECT)


#################################################################

# Eval prep flows - ASR hyps input preparation
eval-prep-force-all:
	@for project in $(PROJECTS); do \
		echo "Running eval prep flow for project $$project"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_PREP --eval_config=$$project --force=True; \
	done
	
eval-prep-all:
	@for project in $(PROJECTS); do \
		echo "Running eval prep flow for project $$project"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_PREP --eval_config=$$project; \
	done

eval-prep:
	@echo "Running eval prep flow"
	@python scripts/asr_eval_lib/main.py --flow="EVAL_PREP" --eval_config=$(PROJECT)

eval-prep-force:
	@echo "Running eval prep flow"
	@python scripts/asr_eval_lib/main.py --flow="EVAL_PREP" --eval_config=$(PROJECT) --force=True

# Evaluation flows - metrics calculation
eval-run-force-all:
	@for project in $(PROJECTS); do \
		echo "Running eval run flow for project $$project"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$$project --force=True; \
	done

eval-run-all:
	@for project in $(PROJECTS); do \
		echo "Running eval run flow for project $$project"; \
		python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$$project; \
	done

eval-run:
	@echo "Running eval run flow"
	@python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$(PROJECT)

eval-run-force:
	@echo "Running eval run flow with forced generation of hyps"
	@python scripts/asr_eval_lib/main.py --flow=EVAL_RUN --eval_config=$(PROJECT) --force=True

# Eval prep - synthetic speech recordings
tts-set-gen:
	@echo "Generating synthetic test set for TTS_SET=$(TTS_SET)"
	@python scripts/tts_gen_lib/main.py --flow=TTS_SET_GEN --tts_set_config=$(TTS_SET)

sde-manifest:
	@echo "Showing manifest for DATASET=$(DATASET) SPLIT=$(SPLIT) in SDE tool"
	@python $(SDE_PATH) -a $(MANIFESTS_DIR)/$(DATASET)-$(SPLIT).jsonl