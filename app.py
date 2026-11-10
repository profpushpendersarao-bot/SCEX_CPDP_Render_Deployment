import os
import re
import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-before-production")
UPLOAD_DIR = Path(os.environ.get("UPLOAD_FOLDER", "/tmp/scex_cpdp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "30"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv"}

TARGET_ALIASES = {
    "bug", "bugs", "defect", "defects", "number_of_bugs",
    "bug_count", "bugcount", "numberofbugs"
}
ID_HINTS = ("name", "file", "class", "id", "version", "module", "project")
MAX_ROWS_PER_FILE = int(os.environ.get("MAX_ROWS_PER_FILE", "20000"))
MAX_FEATURES = int(os.environ.get("MAX_FEATURES", "40"))
SEED = 42

MODELS = {
    "Dummy majority": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        ))
    ]),
    "Random forest (cost-aware class weights)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=1, random_state=SEED
        ))
    ]),
    "Extra trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", n_jobs=1, random_state=SEED
        ))
    ])
}


def normalized_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def detect_target(columns):
    aliases = {normalized_name(x) for x in TARGET_ALIASES}
    return next((c for c in columns if normalized_name(c) in aliases), None)


def safe_project_name(filename):
    stem = Path(filename).stem.lower()
    known = [
        "ant", "camel", "ivy", "jedit", "log4j", "lucene",
        "poi", "synapse", "velocity", "xalan", "xerces"
    ]
    for project in known:
        if re.search(rf"(^|[^a-z0-9]){re.escape(project)}([^a-z0-9]|$)", stem):
            return project
    stem = re.sub(r"[-_ ]?(v|version)?\d+(?:\.\d+)*$", "", stem)
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "project"


def load_csv_low_memory(path, project_name, max_rows=MAX_ROWS_PER_FILE):
    # Inspect headers first, then probe a small sample for numeric fields.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    target = detect_target(header)
    if target is None:
        raise ValueError(
            f"{Path(path).name}: no recognized defect target column. "
            f"Expected one of: {', '.join(sorted(TARGET_ALIASES))}"
        )

    probe = pd.read_csv(path, nrows=100, low_memory=True)
    numeric_cols = probe.select_dtypes(include=[np.number]).columns.tolist()
    del probe
    gc.collect()

    usecols = list(dict.fromkeys(numeric_cols + [target]))
    frame = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=True)

    if max_rows and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=SEED)

    target_values = pd.to_numeric(frame[target], errors="coerce")
    valid = target_values.notna()
    frame = frame.loc[valid].copy()
    y = (target_values.loc[valid] > 0).astype("int8")

    X = frame.drop(columns=[target], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    drop_cols = [
        c for c in X.columns
        if any(h in str(c).lower() for h in ID_HINTS)
    ]
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")

    if X.shape[1]:
        X = X.loc[:, X.nunique(dropna=True) > 1]
    if MAX_FEATURES and X.shape[1] > MAX_FEATURES:
        X = X.iloc[:, :MAX_FEATURES]

    result = X.reset_index(drop=True)
    result["__label__"] = y.reset_index(drop=True).astype("int8")
    result["__project__"] = project_name

    del frame, X, y, target_values
    gc.collect()
    return result, target


def metrics_for(y_true, y_pred, scores):
    values = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    if len(np.unique(y_true)) == 2:
        values["roc_auc"] = roc_auc_score(y_true, scores)
        values["pr_auc"] = average_precision_score(y_true, scores)
    else:
        values["roc_auc"] = None
        values["pr_auc"] = None
    return values


def run_lopo(files):
    project_frames = {}
    file_notes = []

    # Load each file individually; only selected uploads are processed.
    for storage_path, original_name in files:
        project = safe_project_name(original_name)
        frame, target_col = load_csv_low_memory(storage_path, project)
        project_frames.setdefault(project, []).append(frame)
        file_notes.append({
            "file": original_name,
            "project": project,
            "rows_loaded": int(len(frame)),
            "target_column": str(target_col),
            "defect_rate": round(float(frame["__label__"].mean()), 4)
        })
        del frame
        gc.collect()

    data_by_project = {}
    for project, parts in project_frames.items():
        data_by_project[project] = pd.concat(parts, ignore_index=True, sort=False)
        del parts
    project_frames.clear()
    gc.collect()

    projects = sorted(data_by_project)
    if len(projects) < 2:
        raise ValueError(
            "LOPO requires at least two distinct projects. Upload CSVs from "
            "two or more projects; multiple versions of one project count as one project."
        )

    common = set(data_by_project[projects[0]].columns) - {"__label__", "__project__"}
    for project in projects[1:]:
        common &= set(data_by_project[project].columns) - {"__label__", "__project__"}
    features = sorted(common)

    if len(features) < 2:
        raise ValueError(
            "Fewer than two common numeric feature names were found across projects. "
            "These files may use different schemas; harmonize features before comparing them."
        )

    # Keep one common schema across all projects.
    for project in projects:
        data_by_project[project] = data_by_project[project][features + ["__label__"]].copy()
    gc.collect()

    results = []
    for held_out in projects:
        train_df = pd.concat(
            [data_by_project[p] for p in projects if p != held_out],
            ignore_index=True
        )
        test_df = data_by_project[held_out]
        X_train = train_df[features].to_numpy(dtype=np.float32, copy=True)
        y_train = train_df["__label__"].to_numpy(dtype=np.int8, copy=True)
        X_test = test_df[features].to_numpy(dtype=np.float32, copy=True)
        y_test = test_df["__label__"].to_numpy(dtype=np.int8, copy=True)

        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"Training data for held-out project '{held_out}' has only one class. "
                "Add more suitable source projects."
            )

        for model_name, template in MODELS.items():
            model = clone(template)
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
            else:
                raw = model.decision_function(X_test)
                scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            predictions = (scores >= 0.5).astype("int8")
            results.append({
                "held_out_project": held_out,
                "model": model_name,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                **metrics_for(y_test, predictions, scores)
            })
            del model
            gc.collect()

        del train_df, test_df, X_train, y_train, X_test, y_test
        gc.collect()

    result_df = pd.DataFrame(results)
    summary_df = (
        result_df.groupby("model")
        [["accuracy", "precision", "recall", "f1", "mcc",
          "balanced_accuracy", "roc_auc", "pr_auc"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("f1", ascending=False)
    )
    return result_df, summary_df, file_notes, projects, features


@app.errorhandler(413)
def too_large(_error):
    flash(
        f"Upload too large. Current per-request limit is {MAX_CONTENT_LENGTH_MB} MB. "
        "Upload fewer/smaller CSV files or increase MAX_CONTENT_LENGTH_MB.",
        "error"
    )
    return redirect(url_for("index"))


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        max_rows=MAX_ROWS_PER_FILE,
        max_features=MAX_FEATURES,
        max_upload_mb=MAX_CONTENT_LENGTH_MB
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    uploads = request.files.getlist("datasets")
    uploads = [f for f in uploads if f and f.filename]
    if len(uploads) < 2:
        flash("Please upload at least two CSV files from distinct projects.", "error")
        return redirect(url_for("index"))

    saved = []
    temp_dir = Path(tempfile.mkdtemp(prefix="scex_"))
    try:
        for upload in uploads:
            filename = secure_filename(upload.filename)
            if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported file: {filename}. Only CSV files are accepted.")
            destination = temp_dir / filename
            upload.save(destination)
            saved.append((str(destination), filename))

        result_df, summary_df, file_notes, projects, features = run_lopo(saved)

        result_path = temp_dir / "lopo_per_project_results.csv"
        summary_path = temp_dir / "model_macro_summary.csv"
        result_df.to_csv(result_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        # Save report in a per-session folder under /tmp. Render ephemeral disks
        # are not permanent, so users should download outputs immediately.
        report_dir = UPLOAD_DIR / temp_dir.name
        report_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(report_dir / result_path.name, index=False)
        summary_df.to_csv(report_dir / summary_path.name, index=False)
        with open(report_dir / "experiment_metadata.json", "w") as f:
            json.dump({
                "projects": projects,
                "common_features": features,
                "files": file_notes,
                "validation": "leave-one-project-out",
                "defect_label_rule": "defect count > 0",
                "max_rows_per_file": MAX_ROWS_PER_FILE,
                "max_features": MAX_FEATURES
            }, f, indent=2)

        return render_template(
            "index.html",
            max_rows=MAX_ROWS_PER_FILE,
            max_features=MAX_FEATURES,
            max_upload_mb=MAX_CONTENT_LENGTH_MB,
            results=result_df.to_dict(orient="records"),
            summary=summary_df.to_dict(orient="records"),
            files=file_notes,
            projects=projects,
            features=features,
            download_id=temp_dir.name
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    finally:
        # The saved report copy remains in UPLOAD_DIR for download.
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()


@app.route("/download/<download_id>/<filename>")
def download(download_id, filename):
    if filename not in {
        "lopo_per_project_results.csv",
        "model_macro_summary.csv",
        "experiment_metadata.json"
    }:
        return "Not found", 404
    report_dir = (UPLOAD_DIR / download_id).resolve()
    allowed_root = UPLOAD_DIR.resolve()
    if allowed_root not in report_dir.parents:
        return "Not found", 404
    path = report_dir / filename
    if not path.exists():
        return "Report expired. Please run the analysis again.", 404
    return send_file(path, as_attachment=True, download_name=filename)


@app.route("/health")
def health():
    return {"status": "ok", "service": "SCEX-CPDP"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
