PY ?= python
export HF_HUB_OFFLINE ?= 1

.PHONY: env-check test test-all smoke lint fetch-small

env-check:
	$(PY) scripts/env_check.py

test:
	$(PY) -m pytest

test-all:
	$(PY) -m pytest -m "not cuda" --timeout 1800

smoke:
	$(PY) -m pytest -m "slow" tests/integration --timeout 1800

lint:
	ruff check vivlmoe scripts tests
	ruff format --check vivlmoe scripts tests

fetch-small:
	HF_HUB_OFFLINE=0 $(PY) scripts/fetch_models.py --only small
