install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest --nbval cluster-analysis/cluster-analysis.ipynb

format:
	isort *.py
	black *.py

lint-notebooks:
	python -m pytest --nbval cluster-analysis/cluster-analysis.ipynb

all: install format lint-notebooks