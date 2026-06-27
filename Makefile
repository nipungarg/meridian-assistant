.PHONY: install ingest api app cli eval eval-quick test fmt clean

PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

install:                ## create venv + install everything
	python3 -m venv .venv && $(PIP) install -U pip && $(PIP) install -e .

ingest:                 ## extract PDFs -> chunk -> embed -> Chroma
	$(PY) -m meridian.ingestion.build_index --reset

api:                    ## run the mock Booking API (port 8000)
	$(PY) -m uvicorn meridian.api.mock_booking_api:app --port 8000

app:                    ## run the Streamlit chat UI
	$(PY) -m streamlit run src/meridian/app/streamlit_app.py

cli:                    ## chat in the terminal
	$(PY) -m meridian.cli

eval:                   ## full eval (retrieval + answer + action + handoff + RAGAS)
	$(PY) eval/run_eval.py

eval-quick:             ## eval without RAGAS
	$(PY) eval/run_eval.py --quick

test:                   ## unit tests (no network)
	$(PY) -m pytest -q

clean:                  ## remove the local vector store
	rm -rf data/chroma
