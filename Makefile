install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

.PHONY: data
data:
	venv/bin/python3 data_processing.py
run:
	venv/bin/python3 main.py