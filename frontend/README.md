# Network IDS Frontend

React + Vite interface for the FastAPI inference service.

## Run

```bash
npm install
npm run dev
```

The frontend expects the API at `http://127.0.0.1:8080`. Override it with
`VITE_API_URL` when needed.

The legacy Streamlit dashboard remains in `dashboard/` during the migration.
