# SCEX-CPDP Render deployment

Flask web app for running a memory-conscious LOPO cross-project defect prediction experiment on uploaded PROMISE/Jureczko-style CSV files.

## Project files

- `app.py` — Flask application, low-memory CSV loader, ML pipeline and routes
- `templates/index.html` — upload form and results dashboard
- `requirements.txt` — Python dependencies
- `render.yaml` — Render Blueprint configuration
- `.gitignore` — excludes local virtual environments and temporary data

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Deploy on Render

1. Create a GitHub repository and upload these files, preserving the `templates/index.html` path.
2. Push the repository to GitHub.
3. In Render, choose **New + → Blueprint** and select the repository containing `render.yaml`.
   Alternatively create a Web Service with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --threads 1 --timeout 300`
4. Deploy. Open the service URL and visit `/health` to check the service.
5. Upload at least two CSV files from distinct projects.

## Input data requirements

- CSV format.
- Each file needs a defect count column named `bug`, `bugs`, `defect`, `defects`, `number_of_bugs`, `bug_count`, `bugcount`, or `numberofbugs` (case-insensitive).
- The app maps defect count > 0 to defective (1), otherwise non-defective (0).
- At least two distinct projects are required.
- The app infers project names from filenames. Example: `ant-1.7.csv` maps to `ant`; review the output project labels.
- Cross-project features are the exact intersection of numeric column names across all uploaded projects.

## Memory controls

Default deployment environment variables in `render.yaml`:
- `MAX_ROWS_PER_FILE=10000`
- `MAX_FEATURES=30`
- `MAX_CONTENT_LENGTH_MB=30`

To reduce memory further, use `MAX_ROWS_PER_FILE=3000` and `MAX_FEATURES=15`.
To increase the row cap, update the environment variable in Render. Do not raise it without checking your instance memory.

The app loads each uploaded CSV separately, reads only numeric columns and the target, downcasts features to float32, and uses single-worker models. Large files or many uploaded projects can still exceed a small Render instance's memory or request timeout.

## Scientific limitations to disclose

- This deployed app is a reproducible research prototype, not proof of the paper's claims.
- Balanced class weights are a cost-sensitive baseline, not a fully calibrated false-negative/false-positive cost function.
- Filename-derived project labels need manual validation.
- Exact feature-name intersection is conservative; datasets with different schemas need an explicit, justified harmonization map.
- NASA MDP data should be evaluated separately unless schema harmonization is established.
- Download results after each run; temporary storage on Render may be ephemeral.
