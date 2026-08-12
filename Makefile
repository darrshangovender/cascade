.PHONY: install test quickstart calibrate benchmark lint clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

quickstart:
	python examples/quickstart.py

calibrate:
	python examples/calibration_demo.py

benchmark:
	python benchmarks/run.py

lint:
	ruff check .

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info benchmarks/results.json
