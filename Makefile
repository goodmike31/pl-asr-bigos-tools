
# BIGOS PELCRA or TEST
PROJECT ?= 
TODAY=$(shell date +'%Y%m%d')
ifeq ($(PROJECT),BIGOS)
	TEST=BIGOS
else ifeq ($(PROJECT),PELCRA)
	TEST=PELCRA
else ifeq ($(PROJECT),SYNTH)
	TEST=SYNTH
else
    $(error Invalid PROJECT value. Use 'BIGOS', 'PELCRA', 'SYNTH', 'TEST' as PROJECT.)
endif

AUDIO_ID="null"

HYPS_STATS_FILE = ./data/asr_hyps_cache/stats/cached_hyps_stats-$(PROJECT)-$(TODAY).csv
.PHONY: e2e-eval run-tests hyps-stats

test-e2e:
	@echo "Running e2e pipeline on test configuration"
	python scripts/asr-evaluation/main.py

all:
	@echo "Running e2e eval pipeline"
	python scripts/asr-evaluation/main.py --eval_config $(PROJECT)

# Eval prep flows - ASR hyps generation
hyps-stats:
	@echo "Running hyps stats flow"
	@python scripts/asr-evaluation/main.py --flow="HYP_STATS" --eval_config=$(PROJECT)
	@cat $(HYPS_STATS_FILE)

eval-prep:
	@echo "Running eval prep flow"
	@python scripts/asr-evaluation/main.py --flow="EVAL_PREP" --eval_config=$(PROJECT)

# Evaluation flows
eval-run:
	@echo "Running eval run flow"
	python scripts/asr-evaluation/main.py --flow=EVAL_RUN --eval_config=$(PROJECT) 

eval-run-force:
	@echo "Running eval run flow with forced generation of hyps"
	python scripts/asr-evaluation/main.py --flow=EVAL_RUN --eval_config=$(PROJECT) --force=True


# Eval prep - synthetic speech recordings
gen-synth-audio:
	@echo "Generating synthetic audio"
	SYNTH_EVAL_SET_PROJECT=AMU-MEDICAL
	python scripts/asr-evaluation/main.py --flow=GEN_SYNTH_AUDIO --synth_eval_set_config=$(SYNTH_EVAL_SET_PROJECT)
	