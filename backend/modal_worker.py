"""Modal worker — replaces Celery for all background tasks.

This file defines Modal functions for:
- ML experiment training (AutoGluon on Modal cloud)
- Experiment orchestration (dataset building, result processing)
- Training critique generation (LLM-based analysis)
- Robustness audits (post-training validation)
- Auto-improve pipeline (iterative experiment improvement)
- Auto DS research sessions (autonomous research loops)

Deploy: modal deploy modal_worker.py
"""
import os
import modal

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App("agentic-ml")

# ---------------------------------------------------------------------------
# Persistent Volumes
# ---------------------------------------------------------------------------
model_volume = modal.Volume.from_name("agentic-ml-models", create_if_missing=True)

# ---------------------------------------------------------------------------
# Secrets — set these in Modal dashboard or via `modal secret create`
# Required keys: DATABASE_URL, SECRET_KEY, API_KEY_ENCRYPTION_KEY,
#                MODAL_TOKEN_ID, MODAL_TOKEN_SECRET
# ---------------------------------------------------------------------------
env_secret = modal.Secret.from_name("agentic-ml-env")

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# Training image: AutoGluon + heavy ML deps (for actual model training)
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tokenizers>=0.15",
        "autogluon.tabular[all]",
        "pandas",
        "numpy",
        "scikit-learn",
        "torch>=2.2",
    )
)

# Orchestration image: full backend for running task logic
orchestration_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Database
        "sqlalchemy>=2.0.25",
        "psycopg2-binary>=2.9.9",
        "alembic>=1.13.1",
        # Data processing
        "pandas>=2.1.4",
        "numpy>=1.26.3",
        "openpyxl>=3.1.2",
        "pyarrow>=15.0.0",
        "python-docx>=1.1.0",
        # Visualization (needed by some services)
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        # LLM providers
        "openai>=1.12.0",
        "httpx>=0.26.0",
        "google-generativeai>=0.3.0",
        # Auth/crypto
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "bcrypt==4.0.1",
        "cryptography>=42.0.0",
        # Pydantic
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "pydantic[email]>=2.5.0",
        "email-validator>=2.1.0",
        # FastAPI (for schema/model imports)
        "fastapi>=0.109.0",
        # Celery/Redis (import compatibility — not actually used)
        "celery>=5.3.6",
        "redis>=5.0.1",
        # Modal
        "modal>=0.64.0",
        # Utilities
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        "slowapi>=0.1.9",
        "scikit-learn>=1.3.0",
    )
    .env({"PYTHONPATH": "/root/backend", "TASK_BACKEND": "modal"})
    .add_local_dir("app", remote_path="/root/backend/app")
)

# Volume mount path for model storage
VOLUME_MOUNT_PATH = "/models"


# ===========================================================================
# TRAINING FUNCTIONS (from modal_training_standalone.py)
# ===========================================================================

@app.function(
    image=training_image,
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={VOLUME_MOUNT_PATH: model_volume},
)
def train_autogluon_remote(
    dataset_json: str,
    target_column: str,
    task_type: str,
    primary_metric: str | None,
    config: dict,
    experiment_id: str,
    holdout_json: str | None = None,
    download_model: bool = True,
) -> dict:
    """Run AutoGluon training on Modal cloud.

    See modal_training_standalone.py for the full implementation.
    This is a re-export — the actual logic is imported at runtime.
    """
    # Import the standalone training function's logic inline
    # (identical to modal_training_standalone.py train_autogluon_remote body)
    import io
    import sys
    import time
    import shutil
    from pathlib import Path

    import pandas as pd
    from autogluon.tabular import TabularPredictor

    class LogCapture:
        def __init__(self, original_stream):
            self.original = original_stream
            self.captured = []
        def write(self, text):
            self.original.write(text)
            if text.strip():
                self.captured.append(text)
        def flush(self):
            self.original.flush()
        def get_logs(self, max_lines=500):
            lines = "".join(self.captured).split("\n")
            if len(lines) > max_lines:
                lines = lines[:100] + ["... (truncated) ..."] + lines[-400:]
            return "\n".join(lines)

    log_capture = LogCapture(sys.stdout)
    sys.stdout = log_capture

    dataset = pd.read_json(io.StringIO(dataset_json), orient="records")
    dataset_size = len(dataset)

    holdout_df = None
    holdout_size = 0
    print(f"Modal: holdout_json provided = {holdout_json is not None and len(holdout_json or '') > 0}")
    if holdout_json:
        holdout_df = pd.read_json(io.StringIO(holdout_json), orient="records")
        holdout_size = len(holdout_df)
        print(f"Modal: Received holdout set with {holdout_size} samples")

    print(f"Modal: Starting training for experiment {experiment_id}")
    print(f"Modal: Training dataset shape: {dataset.shape}")
    print(f"Modal: Task type: {task_type}, Target: {target_column}")

    if primary_metric is None:
        metric_map = {
            "regression": "root_mean_squared_error",
            "binary": "roc_auc",
            "multiclass": "accuracy",
            "quantile": "pinball_loss",
        }
        primary_metric = metric_map.get(task_type, "accuracy")

    metric_mapping = {
        "rmse": "root_mean_squared_error",
        "mse": "mean_squared_error",
        "mae": "mean_absolute_error",
        "r2": "r2",
        "accuracy": "accuracy",
        "auc": "roc_auc",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "log_loss": "log_loss",
    }
    eval_metric = metric_mapping.get(primary_metric.lower(), primary_metric)

    if task_type == "multiclass":
        multiclass_metric_fixes = {
            "f1": "f1_macro",
            "roc_auc": "roc_auc_ovr_weighted",
            "precision": "precision_macro",
            "recall": "recall_macro",
        }
        if eval_metric in multiclass_metric_fixes:
            old_metric = eval_metric
            eval_metric = multiclass_metric_fixes[eval_metric]
            print(f"Modal: Converted metric '{old_metric}' to '{eval_metric}' for multiclass task")
    elif task_type == "regression":
        regression_defaults = {
            "f1": "root_mean_squared_error",
            "f1_macro": "root_mean_squared_error",
            "accuracy": "root_mean_squared_error",
            "roc_auc": "root_mean_squared_error",
        }
        if eval_metric in regression_defaults:
            old_metric = eval_metric
            eval_metric = regression_defaults[eval_metric]
            print(f"Modal: Converted metric '{old_metric}' to '{eval_metric}' for regression task")

    artifact_path = Path(f"/tmp/autogluon/{experiment_id}")
    if artifact_path.exists():
        shutil.rmtree(artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)

    time_limit = config.get("time_limit", 300)
    presets = config.get("presets", "best_quality")
    num_bag_folds = config.get("num_bag_folds", 8)
    num_stack_levels = config.get("num_stack_levels", 1)

    print(f"Modal: Training with time_limit={time_limit}s, presets={presets}")
    start_time = time.time()

    problem_type_map = {
        "regression": "regression",
        "binary": "binary",
        "multiclass": "multiclass",
        "quantile": "quantile",
    }
    problem_type = problem_type_map.get(task_type, "binary")

    predictor = TabularPredictor(
        label=target_column,
        path=str(artifact_path),
        eval_metric=eval_metric,
        problem_type=problem_type,
    )

    ag_args_fit = {"num_cpus": 2, "num_gpus": 0}

    predictor.fit(
        train_data=dataset,
        time_limit=time_limit,
        presets=presets,
        num_bag_folds=num_bag_folds,
        num_stack_levels=num_stack_levels,
        dynamic_stacking=False,
        verbosity=2,
        ag_args_fit=ag_args_fit,
    )

    training_time = time.time() - start_time

    leaderboard_df = predictor.leaderboard(silent=True)
    leaderboard = leaderboard_df.to_dict(orient="records")
    best_model_name = predictor.model_best

    feature_importances = {}
    try:
        importance_df = predictor.feature_importance(dataset, silent=True)
        if importance_df is not None and not importance_df.empty:
            feature_importances = importance_df["importance"].to_dict()
    except Exception as e:
        print(f"Could not compute feature importances: {e}")

    import math

    def is_valid_score(val):
        if val is None:
            return False
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return False
        return True

    metrics = {}
    try:
        predictor_info = predictor.info()
        best_score = predictor_info.get("best_model_score_val")
        if is_valid_score(best_score):
            metrics[primary_metric] = float(best_score)
            metrics["score_val"] = float(best_score)

        best_model_row = leaderboard_df[leaderboard_df["model"] == best_model_name].iloc[0]
        for score_col in ["score_val", "score_test", "score_holdout"]:
            if score_col in best_model_row:
                val = best_model_row[score_col]
                if is_valid_score(val):
                    metrics["score_val"] = float(val)
                    if primary_metric not in metrics:
                        metrics[primary_metric] = float(val)
                    break

        if not metrics.get(primary_metric):
            eval_result = predictor.evaluate(dataset, silent=True)
            if isinstance(eval_result, dict):
                for k, v in eval_result.items():
                    if is_valid_score(v):
                        metrics[k] = float(v)
                if primary_metric not in metrics and eval_result:
                    first_val = list(eval_result.values())[0]
                    if is_valid_score(first_val):
                        metrics[primary_metric] = float(first_val)
            elif is_valid_score(eval_result):
                metrics[primary_metric] = float(eval_result)
                metrics["score_val"] = float(eval_result)
    except Exception as e:
        print(f"Could not extract detailed metrics: {e}")

    num_models = len(leaderboard)
    print(f"Modal: Training complete. {num_models} models, best={best_model_name}")

    fit_summary = {}
    try:
        fit_summary = predictor.fit_summary(verbosity=0)
    except Exception:
        pass

    # Train metrics for overfitting detection
    train_metrics = {}
    try:
        train_eval = predictor.evaluate(dataset, silent=True)
        if isinstance(train_eval, dict):
            for k, v in train_eval.items():
                if is_valid_score(v):
                    train_metrics[f"train_{k}"] = float(v)
        elif is_valid_score(train_eval):
            train_metrics[f"train_{primary_metric}"] = float(train_eval)
    except Exception:
        pass

    # Holdout evaluation
    holdout_metrics = {}
    holdout_score = None
    if holdout_df is not None and len(holdout_df) > 0:
        try:
            holdout_eval = predictor.evaluate(holdout_df, silent=True)
            if isinstance(holdout_eval, dict):
                for k, v in holdout_eval.items():
                    if is_valid_score(v):
                        holdout_metrics[f"holdout_{k}"] = float(v)
                        if k == eval_metric or k == primary_metric.lower():
                            holdout_score = float(v)
                if holdout_score is None and holdout_eval:
                    first_val = list(holdout_eval.values())[0]
                    if is_valid_score(first_val):
                        holdout_score = float(first_val)
            elif is_valid_score(holdout_eval):
                holdout_score = float(holdout_eval)
                holdout_metrics[f"holdout_{primary_metric}"] = holdout_score
            holdout_metrics["holdout_num_samples"] = len(holdout_df)
        except Exception as e:
            print(f"Could not evaluate on holdout: {e}")

    sys.stdout = log_capture.original
    training_logs = log_capture.get_logs()

    # Save model to persistent volume
    model_saved_to_volume = False
    volume_model_path = None
    try:
        volume_model_dir = Path(f"/models/{experiment_id}")
        if volume_model_dir.exists():
            shutil.rmtree(volume_model_dir)
        shutil.copytree(artifact_path, volume_model_dir)
        model_volume.commit()
        model_saved_to_volume = True
        volume_model_path = str(volume_model_dir)
    except Exception as e:
        print(f"Modal: WARNING - Failed to save model to volume: {e}")

    # Package model archive if requested
    model_archive_b64 = None
    model_too_large = False
    MAX_MODEL_SIZE_MB = 500

    if download_model:
        import tarfile
        import base64
        import io as io_module

        total_size = sum(item.stat().st_size for item in artifact_path.rglob("*") if item.is_file())
        estimated_archive_mb = (total_size / (1024 * 1024)) * 0.7

        if estimated_archive_mb > MAX_MODEL_SIZE_MB:
            model_too_large = True
        else:
            try:
                archive_buffer = io_module.BytesIO()
                with tarfile.open(fileobj=archive_buffer, mode="w:gz") as tar:
                    for item in artifact_path.iterdir():
                        tar.add(item, arcname=item.name)
                archive_buffer.seek(0)
                archive_bytes = archive_buffer.read()
                archive_size_mb = len(archive_bytes) / (1024 * 1024)
                if archive_size_mb > MAX_MODEL_SIZE_MB:
                    model_too_large = True
                else:
                    model_archive_b64 = base64.b64encode(archive_bytes).decode("utf-8")
            except Exception:
                pass

    return {
        "success": True,
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "artifact_path": str(artifact_path),
        "model_archive_b64": model_archive_b64,
        "model_downloaded": model_archive_b64 is not None,
        "model_too_large": model_too_large,
        "model_saved_to_volume": model_saved_to_volume,
        "volume_model_path": volume_model_path,
        "feature_importances": feature_importances,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_score": holdout_score,
        "training_time_seconds": training_time,
        "num_models_trained": num_models,
        "task_type": task_type,
        "backend": "modal",
        "training_logs": training_logs,
        "fit_summary": fit_summary,
        "dataset_size": dataset_size,
        "holdout_size": holdout_size,
    }


# ===========================================================================
# PREDICTION FUNCTIONS
# ===========================================================================

@app.function(
    image=training_image,
    cpu=2,
    memory=8192,
    timeout=300,
    volumes={VOLUME_MOUNT_PATH: model_volume},
)
def predict_remote(experiment_id: str, input_data_json: str) -> dict:
    """Make predictions using a model stored on Modal Volume."""
    import io
    from pathlib import Path
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    model_volume.reload()
    model_path = Path(f"/models/{experiment_id}")
    if not model_path.exists():
        return {"success": False, "error": f"Model not found at {model_path}", "predictions": None}

    try:
        predictor = TabularPredictor.load(str(model_path))
    except Exception as e:
        return {"success": False, "error": f"Failed to load model: {e}", "predictions": None}

    try:
        input_df = pd.read_json(io.StringIO(input_data_json), orient="records")
    except Exception as e:
        return {"success": False, "error": f"Failed to parse input: {e}", "predictions": None}

    try:
        predictions = predictor.predict(input_df)
        probabilities = None
        try:
            prob_df = predictor.predict_proba(input_df)
            probabilities = prob_df.to_dict(orient="records")
        except Exception:
            pass
        return {
            "success": True,
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "num_samples": len(predictions),
        }
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {e}", "predictions": None}


@app.function(
    image=training_image,
    cpu=1,
    memory=4096,
    timeout=60,
    volumes={VOLUME_MOUNT_PATH: model_volume},
)
def check_model_exists(experiment_id: str) -> dict:
    """Check if a model exists on the Modal Volume."""
    from pathlib import Path
    model_volume.reload()
    model_path = Path(f"/models/{experiment_id}")
    exists = model_path.exists()
    model_size_mb = 0
    if exists:
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        model_size_mb = total_size / (1024 * 1024)
    return {
        "exists": exists,
        "experiment_id": experiment_id,
        "model_path": str(model_path) if exists else None,
        "model_size_mb": round(model_size_mb, 2) if exists else None,
    }


@app.function(
    image=training_image,
    cpu=1,
    memory=4096,
    timeout=60,
    volumes={VOLUME_MOUNT_PATH: model_volume},
)
def list_models_on_volume() -> dict:
    """List all models stored on the Modal Volume."""
    from pathlib import Path
    model_volume.reload()
    models_dir = Path("/models")
    models = []
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir():
                total_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                models.append({"experiment_id": item.name, "size_mb": round(total_size / (1024 * 1024), 2)})
    return {"models": models, "count": len(models)}


@app.function(
    image=training_image,
    cpu=1,
    memory=4096,
    timeout=120,
    volumes={VOLUME_MOUNT_PATH: model_volume},
)
def delete_model_from_volume(experiment_id: str) -> dict:
    """Delete a model from the Modal Volume."""
    import shutil
    from pathlib import Path
    model_volume.reload()
    model_path = Path(f"/models/{experiment_id}")
    if not model_path.exists():
        return {"success": False, "error": f"Model {experiment_id} not found"}
    try:
        shutil.rmtree(model_path)
        model_volume.commit()
        return {"success": True, "message": f"Model {experiment_id} deleted"}
    except Exception as e:
        return {"success": False, "error": f"Failed to delete: {e}"}


# ===========================================================================
# ORCHESTRATION FUNCTIONS — replace Celery tasks
# ===========================================================================

def _setup_backend_env():
    """Set up the Python environment for backend imports."""
    import sys
    backend_path = "/root/backend"
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    # Ensure TASK_BACKEND is set for dispatch_task within tasks
    os.environ.setdefault("TASK_BACKEND", "modal")


@app.function(
    image=orchestration_image,
    secrets=[env_secret],
    timeout=7200,  # 2 hours max for experiment
)
def run_experiment_modal(experiment_id: str) -> dict:
    """Run an ML experiment — orchestration + Modal training.

    This replaces the Celery task `app.tasks.run_experiment_modal`.
    """
    _setup_backend_env()
    from app.tasks.experiment_tasks import run_experiment_modal as _task
    # Call the task function directly (self=None, task doesn't use self)
    return _task(None, experiment_id)


@app.function(
    image=orchestration_image,
    secrets=[env_secret],
    timeout=600,  # 10 min for critique
)
def generate_training_critique(experiment_id: str, trial_id: str) -> dict:
    """Generate AI critique of training results.

    This replaces the Celery task `app.tasks.generate_training_critique`.
    """
    _setup_backend_env()
    from app.tasks.experiment_tasks import generate_training_critique as _task
    return _task(None, experiment_id, trial_id)


@app.function(
    image=orchestration_image,
    secrets=[env_secret],
    timeout=600,
)
def run_robustness_audit(experiment_id: str) -> dict:
    """Run robustness audit for a completed experiment.

    This replaces the Celery task `app.tasks.run_robustness_audit`.
    """
    _setup_backend_env()
    from app.tasks.experiment_tasks import run_robustness_audit as _task
    return _task(None, experiment_id)


@app.function(
    image=orchestration_image,
    secrets=[env_secret],
    timeout=7200,
)
def run_auto_improve_pipeline(experiment_id: str, use_enhanced_pipeline: bool = True) -> dict:
    """Run the auto-improve pipeline for an experiment.

    This replaces the Celery task `app.tasks.run_auto_improve_pipeline`.
    """
    _setup_backend_env()
    from app.tasks.experiment_tasks import run_auto_improve_pipeline as _task
    return _task(None, experiment_id, use_enhanced_pipeline)


@app.function(
    image=orchestration_image,
    secrets=[env_secret],
    timeout=86400,  # 24 hours for autonomous research
)
def run_auto_ds_session(session_id: str, initial_dataset_spec_ids: list = None) -> dict:
    """Run an Auto DS autonomous research session.

    This replaces the Celery task `app.tasks.auto_ds_tasks.run_auto_ds_session`.
    """
    _setup_backend_env()
    from app.tasks.auto_ds_tasks import run_auto_ds_session as _task
    return _task(None, session_id, initial_dataset_spec_ids)


@app.function(
    image=orchestration_image,
    secrets=[env_secret],
    timeout=60,
)
def cancel_experiment(experiment_id: str) -> dict:
    """Cancel a running experiment."""
    _setup_backend_env()
    from app.tasks.experiment_tasks import cancel_experiment as _task
    return _task(experiment_id)
