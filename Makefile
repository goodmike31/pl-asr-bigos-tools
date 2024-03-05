.PHONY: e2e-test run-tests

# BIGOS PELCRA or TEST
PROJECT := 

e2e-test:
	@echo "Running e2e pipeline on test configuration"
	python scripts/asr-evaluation/main.py


e2e-eval:
	@echo "Running e2e pipeline on eval configuration"
	python scripts/asr-evaluation/main.py --eval_config $(PROJECT)