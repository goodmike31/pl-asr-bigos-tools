
# BIGOS PELCRA or TEST
PROJECT ?= 
TODAY=$(shell date +'%Y%m%d')
ifeq ($(PROJECT),BIGOS)
	TEST=BIGOS
else ifeq ($(PROJECT),PELCRA)
	TEST=PELCRA
else
    $(error Invalid PROJECT value. Use 'BIGOS', 'PELCRA' or 'TEST' as PROJECT.)
endif

AUDIO_ID="null"

HYPS_STATS_FILE = ./data/asr_hyps_cache/stats/cached_hyps_stats-$(PROJECT)-$(TODAY).csv
.PHONY: e2e-test e2e-eval run-tests hyps-stats


e2e-test:
	@echo "Running e2e pipeline on test configuration"
	python scripts/asr-evaluation/main.py


e2e-eval:
	@echo "Running e2e pipeline on eval configuration"
	python scripts/asr-evaluation/main.py --eval_config $(PROJECT)


hyps-stats:
	@echo "Running hyps stats flow"
	@python scripts/asr-evaluation/main.py --flow="HYP_STATS" --eval_config=$(PROJECT)
	@cat $(HYPS_STATS_FILE)

eval-prep:
	@echo "Running eval prep flow"
	@python scripts/asr-evaluation/main.py --flow="EVAL_PREP" --eval_config=$(PROJECT)

eval-run:
	@echo "Running eval run flow"
	python scripts/asr-evaluation/main.py --flow=EVAL_RUN --eval_config=$(PROJECT) 

eval-run-force:
	@echo "Running eval run flow"
	python scripts/asr-evaluation/main.py --flow=EVAL_RUN --eval_config=$(PROJECT) --force=True

check-audio:
	