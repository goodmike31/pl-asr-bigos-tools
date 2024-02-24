.PHONY: e2e-test run-tests

e2e-test:
	@echo "Running e2e pipeline on test configuration"
	python scripts/asr-evaluation/main.py