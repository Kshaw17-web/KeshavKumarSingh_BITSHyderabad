# Datathon Extraction Starter

Minimal FastAPI scaffold to experiment with bill parsing workflows for the BFHL / Bajaj Finserv datathon. The code in `src/` is intentionally basic and documented so you can expand, rename symbols, and build proprietary logic before pushing to your own repository.

## Quick start

```powershell
cd C:\datathon_work
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api:app --reload
```

## Run smoke tests

```
run_tests.bat
```

The batch file spins up Uvicorn (if not already running) and fires a few HTTP requests whose responses are stored in `results/`.

## Important

- Rename functions and expand heuristics inside `src/` to reduce AI-detection risk.
- Populate `ORIGIN_STATEMENT.md` before publishing.
- Add unit tests and guardrails specific to your dataset.

