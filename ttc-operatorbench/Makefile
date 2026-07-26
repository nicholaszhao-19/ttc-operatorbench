UV ?= uv
PYTHON_VERSION ?= 3.12
PYTHONPATH := src
UV_CACHE_DIR := .uv-cache
MPLCONFIGDIR ?= .uv-cache/matplotlib

.PHONY: check lint type test verify-results sync sync-evalplus sync-hf sync-pilot

check: lint type test verify-results

lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --python $(PYTHON_VERSION) --with ruff --no-project ruff check .

type:
	PYTHONPATH=$(PYTHONPATH) UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR) $(UV) run --python $(PYTHON_VERSION) --with matplotlib --with mypy --with pydantic --with pytest --with typer --no-project mypy src tests

test:
	PYTHONPATH=$(PYTHONPATH) UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR) $(UV) run --python $(PYTHON_VERSION) --with matplotlib --with pydantic --with pytest --no-project pytest

verify-results:
	PYTHONPATH=$(PYTHONPATH) UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --python $(PYTHON_VERSION) --with pydantic --with typer --no-project python -m ttc_operatorbench verify-results

sync:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --group dev

sync-hf:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --group dev --group hf

sync-evalplus:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --group dev --group evalplus

sync-pilot:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync --group dev --group hf --group evalplus
