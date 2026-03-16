"""Standalone Modal training function.

This file is intentionally placed OUTSIDE the app package to avoid
importing any local dependencies. Modal uploads this file to the cloud,
and it needs to be completely self-contained.

DO NOT import anything from the app package in this file!
"""
import modal

# Create Modal app
app = modal.App("agentic-ml-training")

# Create a persistent Volume for storing trained models
# This allows models to persist between function calls for remote predictions
model_volume = modal.Volume.from_name("agentic-ml-models", create_if_missing=True)

# Define the container image with AutoGluon and all dependencies
# Install tokenizers first (pre-built wheel) to avoid old version being pulled in
# and failing to compile with modern Rust
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tokenizers>=0.15",  # Pre-built wheel, install first to avoid old version
        "autogluon.tabular[all]",  # All model types including text/multimodal
        "pandas",
        "numpy",
        "scikit-learn",
        "torch>=2.2",
    )
)


# Volume mount path for persistent model storage
VOLUME_MOUNT_PATH = "/models"


@app.function(
    image=training_image,
    cpu=8,  # Use 8 CPUs for fast training
    memory=32768,  # 32GB RAM
    timeout=7200,  # 2 hour max
    volumes={VOLUME_MOUNT_PATH: model_volume},  # Mount volume for persistent storage
)
def train_autogluon_remote(
    dataset_json: str,
    target_column: str,
    task_type: str,
    primary_metric: str | None,
    config: dict,
    experiment_id: str,
    holdout_json: str | None = None,  # Optional holdout data for evaluation
    download_model: bool = True,  # Whether to compress and return model artifacts
) -> dict:
    """Run AutoGluon training on Modal cloud.

    This function runs entirely in the cloud and must be self-contained.

    Args:
        dataset_json: JSON-serialized dataset (training data only)
        target_column: Name of target column
        task_type: ML task type (binary, multiclass, regression, etc.)
        primary_metric: Metric to optimize
        config: AutoML configuration
        experiment_id: Unique experiment ID
        holdout_json: Optional JSON-serialized holdout data for evaluation
        download_model: If True, compress and return model artifacts for local use.
                       If False, skip compression (faster, but predictions won't work locally).

    Returns:
        Dict with training results including:
        - metrics: validation and holdout scores
        - train_metrics: metrics on training data (for overfitting detection)
        - dataset_size: training set size
        - holdout_size: holdout set size (if provided)
        - captured logs
    """
    import io
    import sys
    import time
    import shutil
    from pathlib import Path

    import pandas as pd
    from autogluon.tabular import TabularPredictor

    # Set up log capture for AI analysis
    class LogCapture:
        def __init__(self, original_stream):
            self.original = original_stream
            self.captured = []

        def write(self, text):
            self.original.write(text)
            if text.strip():  # Only capture non-empty lines
                self.captured.append(text)

        def flush(self):
            self.original.flush()

        def get_logs(self, max_lines=500):
            """Get captured logs, truncated if needed."""
            lines = "".join(self.captured).split("\n")
            if len(lines) > max_lines:
                # Keep first 100 and last 400 lines
                lines = lines[:100] + ["... (truncated) ..."] + lines[-400:]
            return "\n".join(lines)

    # Capture stdout for training logs
    log_capture = LogCapture(sys.stdout)
    sys.stdout = log_capture

    # Deserialize dataset (use StringIO to avoid FutureWarning)
    dataset = pd.read_json(io.StringIO(dataset_json), orient="records")
    dataset_size = len(dataset)

    # Deserialize holdout data if provided
    holdout_df = None
    holdout_size = 0
    print(f"Modal: holdout_json provided = {holdout_json is not None and len(holdout_json or '') > 0}")
    if holdout_json:
        holdout_df = pd.read_json(io.StringIO(holdout_json), orient="records")
        holdout_size = len(holdout_df)
        print(f"Modal: Received holdout set with {holdout_size} samples")
    else:
        print("Modal: No holdout data provided - holdout evaluation will be skipped")

    print(f"Modal: Starting training for experiment {experiment_id}")
    print(f"Modal: Training dataset shape: {dataset.shape}")
    print(f"Modal: Task type: {task_type}, Target: {target_column}")

    # Metric mapping
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

    # Fix metrics that are incompatible with the task type
    # AutoGluon requires specific metric variants for multiclass/regression
    if task_type == "multiclass":
        multiclass_metric_fixes = {
            "f1": "f1_macro",  # f1 only works for binary
            "roc_auc": "roc_auc_ovr_weighted",  # roc_auc only works for binary
            "precision": "precision_macro",
            "recall": "recall_macro",
        }
        if eval_metric in multiclass_metric_fixes:
            old_metric = eval_metric
            eval_metric = multiclass_metric_fixes[eval_metric]
            print(f"Modal: Converted metric '{old_metric}' to '{eval_metric}' for multiclass task")
    elif task_type == "regression":
        # Ensure we don't use classification metrics for regression
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

    # Set up artifact path (local to Modal container)
    artifact_path = Path(f"/tmp/autogluon/{experiment_id}")
    if artifact_path.exists():
        shutil.rmtree(artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Configure AutoGluon - NO resource limits in cloud!
    time_limit = config.get("time_limit", 300)  # Default 5 min in cloud
    presets = config.get("presets", "best_quality")  # Use best quality in cloud
    num_bag_folds = config.get("num_bag_folds", 8)
    num_stack_levels = config.get("num_stack_levels", 1)

    print(f"Modal: Training with time_limit={time_limit}s, presets={presets}")

    start_time = time.time()

    # Problem type mapping
    problem_type_map = {
        "regression": "regression",
        "binary": "binary",
        "multiclass": "multiclass",
        "quantile": "quantile",
    }
    problem_type = problem_type_map.get(task_type, "binary")

    # Train with AutoGluon
    predictor = TabularPredictor(
        label=target_column,
        path=str(artifact_path),
        eval_metric=eval_metric,
        problem_type=problem_type,
    )

    # IMPORTANT: Limit Ray parallelism to prevent worker death from memory exhaustion
    # With 8 CPUs and 32GB RAM, we limit each model to 2 CPUs to avoid OOM
    # This prevents the "worker died unexpectedly" Ray errors
    ag_args_fit = {
        "num_cpus": 2,  # Limit CPUs per model to control parallelism
        "num_gpus": 0,  # No GPU
    }

    predictor.fit(
        train_data=dataset,
        time_limit=time_limit,
        presets=presets,
        num_bag_folds=num_bag_folds,
        num_stack_levels=num_stack_levels,
        dynamic_stacking=False,  # Disable to avoid "Learner is already fit" edge case
        verbosity=2,
        ag_args_fit=ag_args_fit,  # Control resource allocation to prevent Ray worker crashes
    )

    training_time = time.time() - start_time

    # Get results
    leaderboard_df = predictor.leaderboard(silent=True)
    leaderboard = leaderboard_df.to_dict(orient="records")
    best_model_name = predictor.model_best

    # Get feature importances
    feature_importances = {}
    try:
        importance_df = predictor.feature_importance(dataset, silent=True)
        if importance_df is not None and not importance_df.empty:
            feature_importances = importance_df["importance"].to_dict()
    except Exception as e:
        print(f"Could not compute feature importances: {e}")

    # Get metrics
    metrics = {}
    try:
        import math

        # Get predictor info for scores
        predictor_info = predictor.info()
        best_score = predictor_info.get("best_model_score_val")

        # Helper to check for valid numeric values
        def is_valid_score(val):
            if val is None:
                return False
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return False
            return True

        if is_valid_score(best_score):
            metrics[primary_metric] = float(best_score)
            metrics["score_val"] = float(best_score)

        # Get additional metrics from leaderboard
        best_model_row = leaderboard_df[
            leaderboard_df["model"] == best_model_name
        ].iloc[0]

        print(f"Modal: Leaderboard columns: {list(leaderboard_df.columns)}")
        print(f"Modal: Best model row: {best_model_row.to_dict()}")

        # Try different column names for the score
        for score_col in ["score_val", "score_test", "score_holdout"]:
            if score_col in best_model_row:
                val = best_model_row[score_col]
                if is_valid_score(val):
                    metrics["score_val"] = float(val)
                    if primary_metric not in metrics:
                        metrics[primary_metric] = float(val)
                    break

        # If still no metrics, evaluate on dataset
        if not metrics.get(primary_metric):
            print("Modal: No metrics from leaderboard, evaluating on data...")
            eval_result = predictor.evaluate(dataset, silent=True)
            print(f"Modal: Evaluation result: {eval_result}")
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

        print(f"Modal: Final metrics: {metrics}")
    except Exception as e:
        print(f"Could not extract detailed metrics: {e}")

    num_models = len(leaderboard)

    print(f"Modal: Training complete. {num_models} models, best={best_model_name}")

    # Get fit summary for detailed analysis
    fit_summary = {}
    try:
        fit_summary = predictor.fit_summary(verbosity=0)
    except Exception as e:
        print(f"Could not get fit summary: {e}")

    # =========================================================================
    # TRAIN METRICS: Evaluate on training data for overfitting detection
    # =========================================================================
    train_metrics = {}
    try:
        print("Modal: Evaluating on training data (for overfitting detection)...")
        train_eval = predictor.evaluate(dataset, silent=True)
        print(f"Modal: Train evaluation result: {train_eval}")

        if isinstance(train_eval, dict):
            for k, v in train_eval.items():
                if is_valid_score(v):
                    train_metrics[f"train_{k}"] = float(v)
        elif is_valid_score(train_eval):
            train_metrics[f"train_{primary_metric}"] = float(train_eval)

        print(f"Modal: Train metrics: {train_metrics}")
    except Exception as e:
        print(f"Could not compute train metrics: {e}")

    # =========================================================================
    # HOLDOUT EVALUATION: Evaluate on held-out data (NEVER seen during training)
    # =========================================================================
    holdout_metrics = {}
    holdout_score = None
    if holdout_df is not None and len(holdout_df) > 0:
        try:
            print("=" * 60)
            print("Modal: HOLDOUT EVALUATION")
            print("=" * 60)
            print(f"Modal: Evaluating on {len(holdout_df)} holdout samples...")

            holdout_eval = predictor.evaluate(holdout_df, silent=True)
            print(f"Modal: Holdout evaluation result: {holdout_eval}")

            if isinstance(holdout_eval, dict):
                for k, v in holdout_eval.items():
                    if is_valid_score(v):
                        holdout_metrics[f"holdout_{k}"] = float(v)
                        if k == eval_metric or k == primary_metric.lower():
                            holdout_score = float(v)
                # If no exact match found, take the first metric
                if holdout_score is None and holdout_eval:
                    first_val = list(holdout_eval.values())[0]
                    if is_valid_score(first_val):
                        holdout_score = float(first_val)
            elif is_valid_score(holdout_eval):
                holdout_score = float(holdout_eval)
                holdout_metrics[f"holdout_{primary_metric}"] = holdout_score

            holdout_metrics["holdout_num_samples"] = len(holdout_df)

            # Compare validation vs holdout for overfitting check
            val_score = metrics.get(primary_metric) or metrics.get("score_val")
            if val_score is not None and holdout_score is not None:
                val_val = abs(val_score) if val_score < 0 else val_score
                holdout_val = abs(holdout_score) if holdout_score < 0 else holdout_score
                diff_pct = ((holdout_val - val_val) / val_val * 100) if val_val != 0 else 0

                print(f"Modal: CV/Val {primary_metric}: {val_score:.4f}")
                print(f"Modal: Holdout {primary_metric}: {holdout_score:.4f}")

                if abs(diff_pct) > 10:
                    print(f"Modal: WARNING - {diff_pct:+.1f}% gap between CV and holdout!")
                elif abs(diff_pct) > 5:
                    print(f"Modal: Moderate gap: {diff_pct:+.1f}%")
                else:
                    print(f"Modal: Scores consistent: {diff_pct:+.1f}%")

            print("=" * 60)

        except Exception as e:
            print(f"Could not evaluate on holdout: {e}")

    # Restore stdout and get captured logs
    sys.stdout = log_capture.original
    training_logs = log_capture.get_logs()

    # =========================================================================
    # SAVE MODEL TO PERSISTENT VOLUME (for remote predictions)
    # =========================================================================
    # Always save to volume so predictions can be made without downloading
    model_saved_to_volume = False
    volume_model_path = None
    try:
        volume_model_dir = Path(f"/models/{experiment_id}")
        if volume_model_dir.exists():
            shutil.rmtree(volume_model_dir)

        # Copy model artifacts to volume
        print(f"Modal: Saving model to persistent volume: {volume_model_dir}")
        shutil.copytree(artifact_path, volume_model_dir)

        # Commit the volume to persist changes
        model_volume.commit()

        model_saved_to_volume = True
        volume_model_path = str(volume_model_dir)
        print(f"Modal: Model saved to volume successfully. Remote predictions available.")
    except Exception as e:
        print(f"Modal: WARNING - Failed to save model to volume: {e}")
        print("Modal: Remote predictions will NOT be available for this model.")

    # Package the model artifacts for download (if requested)
    model_archive_b64 = None
    model_too_large = False
    MAX_MODEL_SIZE_MB = 500  # Skip download if larger than 500MB to avoid timeout

    if download_model:
        # First, check the uncompressed size to estimate archive size
        import tarfile
        import base64
        import io as io_module

        # Calculate approximate size
        total_size = 0
        for item in artifact_path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        uncompressed_size_mb = total_size / (1024 * 1024)
        estimated_archive_mb = uncompressed_size_mb * 0.7  # Assume ~30% compression

        print(f"Modal: Model size: {uncompressed_size_mb:.1f}MB uncompressed, ~{estimated_archive_mb:.1f}MB estimated compressed")

        if estimated_archive_mb > MAX_MODEL_SIZE_MB:
            print(f"Modal: WARNING - Model too large ({estimated_archive_mb:.0f}MB > {MAX_MODEL_SIZE_MB}MB limit)")
            print("Modal: Skipping model download to avoid timeout. Metrics will still be saved.")
            print("Modal: Use 'Download Model' button in UI to fetch model later.")
            model_too_large = True
            download_model = False
        else:
            # Compress and base64 encode the model for transfer back to local machine
            print("Modal: Compressing model artifacts for download...")

            try:
                # Create a tar.gz archive of the model directory
                archive_buffer = io_module.BytesIO()
                with tarfile.open(fileobj=archive_buffer, mode="w:gz") as tar:
                    # Add all files in the artifact path
                    for item in artifact_path.iterdir():
                        tar.add(item, arcname=item.name)

                archive_buffer.seek(0)
                archive_bytes = archive_buffer.read()
                archive_size_mb = len(archive_bytes) / (1024 * 1024)

                # Double-check actual size after compression
                if archive_size_mb > MAX_MODEL_SIZE_MB:
                    print(f"Modal: WARNING - Compressed model still too large ({archive_size_mb:.0f}MB)")
                    print("Modal: Skipping download to avoid timeout.")
                    model_too_large = True
                else:
                    # Base64 encode for safe JSON transport
                    model_archive_b64 = base64.b64encode(archive_bytes).decode("utf-8")
                    print(f"Modal: Model archive created: {archive_size_mb:.2f} MB")

            except Exception as e:
                print(f"Modal: WARNING - Failed to create model archive: {e}")
                print("Modal: Model will NOT be available for local predictions")
    else:
        print("Modal: Skipping model download (download_model=False)")
        print("Modal: Predictions will NOT be available locally - use Download button to get model")

    return {
        "success": True,
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "artifact_path": str(artifact_path),
        "model_archive_b64": model_archive_b64,  # Base64-encoded model archive (None if download_model=False or too large)
        "model_downloaded": model_archive_b64 is not None,  # Whether model artifacts were included
        "model_too_large": model_too_large,  # True if model was skipped due to size
        "model_saved_to_volume": model_saved_to_volume,  # True if model saved to Modal Volume
        "volume_model_path": volume_model_path,  # Path on Modal Volume for remote predictions
        "feature_importances": feature_importances,
        "metrics": metrics,
        "train_metrics": train_metrics,  # Train metrics for overfitting detection
        "holdout_metrics": holdout_metrics,  # Holdout evaluation results
        "holdout_score": holdout_score,  # Primary holdout score
        "training_time_seconds": training_time,
        "num_models_trained": num_models,
        "task_type": task_type,
        "backend": "modal",
        "training_logs": training_logs,
        "fit_summary": fit_summary,
        "dataset_size": dataset_size,  # Training dataset size
        "holdout_size": holdout_size,  # Holdout set size
    }


@app.function(
    image=training_image,
    cpu=2,  # Lightweight for inference
    memory=8192,  # 8GB RAM should be enough for predictions
    timeout=300,  # 5 minute timeout for predictions
    volumes={VOLUME_MOUNT_PATH: model_volume},
)
def predict_remote(
    experiment_id: str,
    input_data_json: str,
) -> dict:
    """Make predictions using a model stored on Modal Volume.

    This function loads a trained model from the persistent volume and makes
    predictions. Use this when the model was too large to download locally.

    Args:
        experiment_id: The experiment ID used during training (model folder name)
        input_data_json: JSON-serialized input data (list of dicts or single dict)

    Returns:
        Dict with predictions and metadata
    """
    import io
    from pathlib import Path

    import pandas as pd
    from autogluon.tabular import TabularPredictor

    print(f"Modal Predict: Loading model for experiment {experiment_id}")

    # Reload volume to get latest data
    model_volume.reload()

    # Load model from volume
    model_path = Path(f"/models/{experiment_id}")
    if not model_path.exists():
        return {
            "success": False,
            "error": f"Model not found at {model_path}. Model may not have been saved to volume.",
            "predictions": None,
        }

    try:
        predictor = TabularPredictor.load(str(model_path))
        print(f"Modal Predict: Model loaded successfully from {model_path}")
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load model: {str(e)}",
            "predictions": None,
        }

    # Parse input data
    try:
        input_df = pd.read_json(io.StringIO(input_data_json), orient="records")
        print(f"Modal Predict: Input shape: {input_df.shape}")
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse input data: {str(e)}",
            "predictions": None,
        }

    # Make predictions
    try:
        predictions = predictor.predict(input_df)
        print(f"Modal Predict: Got {len(predictions)} predictions")

        # Also get prediction probabilities if classification
        probabilities = None
        try:
            prob_df = predictor.predict_proba(input_df)
            probabilities = prob_df.to_dict(orient="records")
        except Exception:
            pass  # Not all models support predict_proba

        return {
            "success": True,
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "num_samples": len(predictions),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction failed: {str(e)}",
            "predictions": None,
        }


@app.function(
    image=training_image,
    cpu=1,
    memory=4096,
    timeout=60,
    volumes={VOLUME_MOUNT_PATH: model_volume},
)
def check_model_exists(experiment_id: str) -> dict:
    """Check if a model exists on the Modal Volume.

    Args:
        experiment_id: The experiment ID to check

    Returns:
        Dict with exists flag and model info
    """
    from pathlib import Path

    model_volume.reload()

    model_path = Path(f"/models/{experiment_id}")
    exists = model_path.exists()

    model_size_mb = 0
    if exists:
        total_size = 0
        for item in model_path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
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
    """List all models stored on the Modal Volume.

    Returns:
        Dict with list of experiment IDs and their sizes
    """
    from pathlib import Path

    model_volume.reload()

    models_dir = Path("/models")
    models = []

    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir():
                total_size = 0
                for file in item.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
                models.append({
                    "experiment_id": item.name,
                    "size_mb": round(total_size / (1024 * 1024), 2),
                })

    return {
        "models": models,
        "count": len(models),
    }


@app.function(
    image=training_image,
    cpu=4,  # More CPU for loading multiple models
    memory=16384,  # 16GB RAM for multiple models
    timeout=600,  # 10 minute timeout
    volumes={VOLUME_MOUNT_PATH: model_volume},
)
def predict_ensemble_remote(
    ensemble_config_json: str,
    input_data_json: str,
) -> dict:
    """Make ensemble predictions using multiple models stored on Modal Volume.

    Loads all member models, gets predictions from each, and combines them
    using weighted averaging.

    Args:
        ensemble_config_json: JSON with ensemble configuration:
            {
                "members": [
                    {"experiment_id": "...", "weight": 0.3},
                    ...
                ],
                "method": "weighted_average"  # or "voting", "median"
            }
        input_data_json: JSON-serialized input data (list of dicts)

    Returns:
        Dict with ensemble predictions and individual member predictions
    """
    import io
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    print("Modal Ensemble: Starting ensemble prediction")

    # Parse inputs
    try:
        ensemble_config = json.loads(ensemble_config_json)
        members = ensemble_config.get("members", [])
        method = ensemble_config.get("method", "weighted_average")
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse ensemble config: {str(e)}",
            "predictions": None,
        }

    if not members:
        return {
            "success": False,
            "error": "No ensemble members specified",
            "predictions": None,
        }

    try:
        input_df = pd.read_json(io.StringIO(input_data_json), orient="records")
        print(f"Modal Ensemble: Input shape: {input_df.shape}")
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse input data: {str(e)}",
            "predictions": None,
        }

    # Reload volume to get latest models
    model_volume.reload()

    # Load models and get predictions from each
    member_predictions = []
    member_probas = []
    valid_weights = []
    is_classification = None

    for member in members:
        exp_id = member.get("experiment_id")
        weight = member.get("weight", 1.0)

        model_path = Path(f"/models/{exp_id}")
        if not model_path.exists():
            print(f"Modal Ensemble: WARNING - Model {exp_id} not found, skipping")
            continue

        try:
            predictor = TabularPredictor.load(str(model_path))
            print(f"Modal Ensemble: Loaded model {exp_id}")

            # Determine if classification
            if is_classification is None:
                problem_type = predictor.problem_type
                is_classification = problem_type in ("binary", "multiclass")

            # Get predictions
            preds = predictor.predict(input_df)
            member_predictions.append({
                "experiment_id": exp_id,
                "predictions": preds.values if hasattr(preds, "values") else preds,
                "weight": weight,
            })
            valid_weights.append(weight)

            # Get probabilities for classification
            if is_classification:
                try:
                    proba = predictor.predict_proba(input_df)
                    member_probas.append({
                        "experiment_id": exp_id,
                        "probas": proba.values if hasattr(proba, "values") else proba,
                        "columns": proba.columns.tolist() if hasattr(proba, "columns") else None,
                        "weight": weight,
                    })
                except Exception:
                    pass

            print(f"Modal Ensemble: Got predictions from {exp_id}")

        except Exception as e:
            print(f"Modal Ensemble: ERROR loading {exp_id}: {e}")
            continue

    if not member_predictions:
        return {
            "success": False,
            "error": "No models could be loaded for ensemble",
            "predictions": None,
        }

    # Normalize weights
    total_weight = sum(valid_weights)
    if total_weight > 0:
        for mp in member_predictions:
            mp["weight"] /= total_weight
        for mp in member_probas:
            mp["weight"] /= total_weight

    # Combine predictions based on method
    print(f"Modal Ensemble: Combining {len(member_predictions)} predictions using {method}")

    try:
        if method == "weighted_average" and not is_classification:
            # Weighted average for regression
            pred_stack = np.stack([mp["predictions"] for mp in member_predictions])
            weights = np.array([mp["weight"] for mp in member_predictions])
            final_predictions = np.sum(pred_stack * weights[:, np.newaxis], axis=0)

        elif method == "weighted_average" and is_classification and member_probas:
            # Weight probabilities then argmax for classification
            first_proba = member_probas[0]
            combined_proba = np.zeros_like(first_proba["probas"])

            for mp in member_probas:
                combined_proba += mp["probas"] * mp["weight"]

            # Get class labels from columns
            columns = first_proba.get("columns")
            if columns:
                final_predictions = [columns[i] for i in np.argmax(combined_proba, axis=1)]
            else:
                final_predictions = np.argmax(combined_proba, axis=1).tolist()

        elif method == "voting" and is_classification:
            # Majority voting for classification
            pred_stack = np.stack([mp["predictions"] for mp in member_predictions])
            from scipy import stats
            final_predictions = stats.mode(pred_stack, axis=0, keepdims=False)[0]

        elif method == "median":
            # Median for regression
            pred_stack = np.stack([mp["predictions"] for mp in member_predictions])
            final_predictions = np.median(pred_stack, axis=0)

        else:
            # Default: simple average
            pred_stack = np.stack([mp["predictions"] for mp in member_predictions])
            final_predictions = np.mean(pred_stack, axis=0)

        # Convert to list
        if hasattr(final_predictions, "tolist"):
            final_predictions = final_predictions.tolist()

        print(f"Modal Ensemble: Generated {len(final_predictions)} ensemble predictions")

        return {
            "success": True,
            "predictions": final_predictions,
            "num_samples": len(final_predictions),
            "members_used": len(member_predictions),
            "member_ids": [mp["experiment_id"] for mp in member_predictions],
            "member_weights": [mp["weight"] for mp in member_predictions],
            "method": method,
            "is_classification": is_classification,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Ensemble combination failed: {str(e)}",
            "predictions": None,
        }


@app.function(
    image=training_image,
    cpu=1,
    memory=4096,
    timeout=120,
    volumes={VOLUME_MOUNT_PATH: model_volume},
)
def delete_model_from_volume(experiment_id: str) -> dict:
    """Delete a model from the Modal Volume.

    Args:
        experiment_id: The experiment ID to delete

    Returns:
        Dict with success status
    """
    import shutil
    from pathlib import Path

    model_volume.reload()

    model_path = Path(f"/models/{experiment_id}")
    if not model_path.exists():
        return {
            "success": False,
            "error": f"Model {experiment_id} not found",
        }

    try:
        shutil.rmtree(model_path)
        model_volume.commit()
        return {
            "success": True,
            "message": f"Model {experiment_id} deleted successfully",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to delete model: {str(e)}",
        }
