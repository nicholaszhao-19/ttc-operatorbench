UV ?= uv
PYTHON_VERSION ?= 3.12
PYTHONPATH := src
UV_CACHE_DIR := .uv-cache

.PHONY: check lint type test sync

check: lint type test

lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --python $(PYTHON_VERSION) --group dev ruff check .

type:
	PYTHONPATH=$(PYTHONPATH) UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --python $(PYTHON_VERSION) --group dev mypy src tests

test:
	PYTHONPATH=$(PYTHONPATH) UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --python $(PYTHON_VERSION) --group dev pytest

sync:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --all-groups
