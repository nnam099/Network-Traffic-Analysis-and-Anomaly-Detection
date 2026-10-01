# Network Anomaly Detection Frontend

React + Vite interface for the FastAPI inference service. This is the primary
demonstration UI; the Python model and API live in `../src/ids/`.

## Run

```bash
npm ci
npm run dev
```

The frontend expects the API at `http://127.0.0.1:8080`. Override it with
`VITE_API_URL` when needed.

Open <http://127.0.0.1:5173> after starting the backend. For an API on another
port, run `VITE_API_URL=http://127.0.0.1:8081 npm run dev`.
