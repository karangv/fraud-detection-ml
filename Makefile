PYTHON ?= python
CONFIG ?= config.yaml
INPUT ?= new.csv
OUTPUT ?= scored.csv

export PYTHONPATH := src

.PHONY: train evaluate predict test

train:
	$(PYTHON) -m fraud.train --config $(CONFIG)

evaluate:
	$(PYTHON) -m fraud.evaluate --config $(CONFIG)

predict:
	$(PYTHON) -m fraud.predict --config $(CONFIG) --input $(INPUT) --output $(OUTPUT)

test:
	$(PYTHON) -m pytest
