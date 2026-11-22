# RootCause dashboard

A small Vite + React dashboard for the FastAPI API. It includes issue
investigation, an 8D report view, a weekly defect chart, a local human-review
demo, document Q&A, and the guarded read-only SQL agent.

## Run locally

From the repository root, start the backend and Postgres using the root setup
instructions. Then open another PowerShell window:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The dashboard calls `http://localhost:8000` by
default. To use another API origin, set `VITE_API_BASE_URL` in a
`frontend/.env.local` file before running the dev server. The backend browser
allow-list is configured with `CORS_ORIGINS`, defaulting to
`http://localhost:5173`.

Build the static dashboard with:

```powershell
npm ci
npm run build
```

The **Review** page is a UI demonstration only. Approve/reject decisions and
comments stay in browser memory and are not sent to the backend.

## Screenshot

_Screenshot placeholder: add a dashboard screenshot here._
