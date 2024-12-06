install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

.PHONY: data
data:
	python3 data_processing.py
run:
	python3 main.py