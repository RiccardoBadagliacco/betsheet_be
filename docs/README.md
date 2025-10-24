# BetSheet API


Run instructions (development):

1. Create and activate a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs for Swagger UI.

Migrations

- Alembic is included as a dependency; initialize alembic with `alembic init alembic` and configure `alembic.ini` to point to your DB.

Testing

Run pytest:

```bash
pytest -q
```
