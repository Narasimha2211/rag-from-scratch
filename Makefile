.PHONY: install install-dev lint typecheck test coverage check run streamlit clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	ruff check .

typecheck:
	mypy rag_from_scratch.py streamlit_app.py

test:
	pytest -v

coverage:
	pytest -v --cov=rag_from_scratch --cov-report=term-missing

check: lint typecheck test

run:
	python rag_from_scratch.py --query "Why do we use overlap in chunking?"

streamlit:
	python -m streamlit run streamlit_app.py

clean:
	rm -rf .pytest_cache .coverage __pycache__ tests/__pycache__
