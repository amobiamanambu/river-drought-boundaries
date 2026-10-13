PYTHON ?= python
DATA_ROOT ?= data/river-drought-boundaries-data-v1.0.0

.PHONY: install validate sample test check

install:
	$(PYTHON) -m pip install -e .

validate:
	$(PYTHON) -m river_drought_boundaries.validate $(DATA_ROOT)

sample:
	$(PYTHON) examples/run_sample.py --check

test:
	$(PYTHON) -m unittest discover -s tests -v

check: test sample
