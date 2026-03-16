"""Celery tasks for AutoML experiments."""
import logging
import os

import pandas as pd

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.experiment import Experiment, ExperimentStatus
from app.models.dataset_spec import DatasetSpec
from app.models.data_source import DataSource
from app.models.model_version import ModelVersion, ModelStatus
from app.services.automl_runner import get_runner_for_task

logger = logging.getLogger(__name__)
settings = get_settings()


def load_dataset_from_spec(db, dataset_spec: DatasetSpec) -> pd.DataFrame:
    """Load dataset from data sources defined in the spec.

    This function handles:
    1. Loading data from one or more data sources
    2. Applying feature engineering (creating new columns)
    3. Creating target column if it doesn't exist
    4. Selecting specified features and target
    """
    if not dataset_spec.data_sources_json:
        raise ValueError("No data sources configured in dataset spec")

    data_sources_json = dataset_spec.data_sources_json

    # Handle case where data_sources_json is a list of UUIDs directly
    if isinstance(data_sources_json, list):
        data_source_ids = data_sources_json
    else:
        # Handle dict format with various key names
        data_source_ids = data_sources_json.get("source_ids", [])
        if not data_source_ids:
            # Try "sources" key (used by tests and dataset_builder)
            data_source_ids = data_sources_json.get("sources", [])
        if not data_source_ids:
            # Try single source_id for backward compatibility
            source_id = data_sources_json.get("source_id")
            if source_id:
                data_source_ids = [source_id]
        if not data_source_ids:
            # Try "primary" key (used by frontend)
            primary_id = data_sources_json.get("primary")
            if primary_id:
                data_source_ids = [primary_id]

    if not data_source_ids:
        raise ValueError("No data source IDs in dataset spec")

    dfs = []
    for source_id in data_source_ids:
        data_source = db.query(DataSource).filter(
            DataSource.id == source_id
        ).first()

        if not data_source:
            raise ValueError(f"Data source {source_id} not found")

        df = load_from_source(data_source)
        dfs.append(df)

    # For now, just use the first data source
    combined_df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

    # Apply feature engineering if specified in spec_json
    spec_json = dataset_spec.spec_json or {}
    engineered_features = spec_json.get("engineered_features", [])
    if engineered_features:
        from app.services.feature_engineering import apply_feature_engineering_with_feedback
        logger.info(f"Applying {len(engineered_features)} feature engineering steps")
        # Use feedback version to track which features succeed/fail
        result = apply_feature_engineering_with_feedback(combined_df, engineered_features)
        combined_df = result.df

        # Log successful and failed features for visibility
        if result.successful_features:
            logger.info(f"Successfully created {len(result.successful_features)} features: {result.successful_features}")
        if result.failed_features:
            logger.warning(f"Failed to create {len(result.failed_features)} features:")
            for failure in result.failed_features:
                logger.warning(f"  - {failure.get('feature', 'unknown')}: {failure.get('error', 'unknown error')}")

    # Apply filters if specified in spec_json or filters_json
    filters = spec_json.get("filters") or (dataset_spec.filters_json if hasattr(dataset_spec, 'filters_json') else None)
    if filters:
        before_filter_count = len(combined_df)
        combined_df = _apply_filters(combined_df, filters)
        after_filter_count = len(combined_df)
        if before_filter_count != after_filter_count:
            logger.info(f"Applied filters: {before_filter_count} -> {after_filter_count} rows ({before_filter_count - after_filter_count} removed)")

    # Create target column if specified and doesn't exist
    target_creation = spec_json.get("target_creation")
    target_column = dataset_spec.target_column or spec_json.get("target_column")

    if target_creation and target_column and target_column not in combined_df.columns:
        from app.services.feature_engineering import apply_target_creation
        logger.info(f"Creating target column '{target_column}' using formula")
        # Ensure target_creation has the column_name set
        if "column_name" not in target_creation:
            target_creation = {**target_creation, "column_name": target_column}
        combined_df = apply_target_creation(combined_df, target_creation, inplace=False)

        # Drop rows with NaN target (common for shifted targets)
        before_count = len(combined_df)
        combined_df = combined_df.dropna(subset=[target_column])
        if len(combined_df) < before_count:
            logger.info(f"Dropped {before_count - len(combined_df)} rows with missing target values")

    # Apply column selection if specified
    if dataset_spec.feature_columns and dataset_spec.target_column:
        columns = list(dataset_spec.feature_columns) + [dataset_spec.target_column]
        available_cols = [c for c in columns if c in combined_df.columns]

        # CRITICAL FIX: Warn about missing columns instead of silently dropping them
        missing_cols = [c for c in columns if c not in combined_df.columns]
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} expected columns (not in dataset): {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}")

        combined_df = combined_df[available_cols]

    return combined_df


def _apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    """Apply filters to dataframe.

    Args:
        df: Input dataframe
        filters: List of filter dicts with column, operator, value

    Returns:
        Filtered dataframe
    """
    if not filters:
        return df

    for filter_spec in filters:
        if not isinstance(filter_spec, dict):
            logger.warning(f"Skipping invalid filter (not a dict): {filter_spec}")
            continue

        column = filter_spec.get("column")
        operator = filter_spec.get("operator")
        value = filter_spec.get("value")

        if not column:
            logger.warning(f"Skipping filter with no column: {filter_spec}")
            continue

        if column not in df.columns:
            logger.warning(f"Skipping filter for non-existent column: {column}")
            continue

        try:
            if operator == ">=":
                df = df[df[column] >= value]
            elif operator == "<=":
                df = df[df[column] <= value]
            elif operator == ">":
                df = df[df[column] > value]
            elif operator == "<":
                df = df[df[column] < value]
            elif operator == "==":
                df = df[df[column] == value]
            elif operator == "!=":
                df = df[df[column] != value]
            elif operator == "in":
                df = df[df[column].isin(value if isinstance(value, list) else [value])]
            elif operator == "not_in":
                df = df[~df[column].isin(value if isinstance(value, list) else [value])]
            elif operator == "is_null":
                df = df[df[column].isna()]
            elif operator == "is_not_null":
                df = df[df[column].notna()]
            else:
                logger.warning(f"Unknown filter operator: {operator}")
        except Exception as e:
            logger.warning(f"Failed to apply filter {filter_spec}: {e}")

    return df


def load_from_source(data_source: DataSource) -> pd.DataFrame:
    """Load data from a specific data source."""
    config = data_source.config_json or {}

    # Handle file uploads
    if data_source.type.value == "file_upload":
        file_path = config.get("file_path")
        if file_path and os.path.exists(file_path):
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path)
            elif file_path.endswith((".xlsx", ".xls")):
                return pd.read_excel(file_path)
            elif file_path.endswith(".parquet"):
                return pd.read_parquet(file_path)

    raise ValueError(f"Cannot load data from source: {data_source.name}")


@celery_app.task(bind=True, name="run_automl_experiment")
def run_automl_experiment_task(self, experiment_id: str) -> dict:
    """
    Celery task to run an AutoML experiment.

    Args:
        experiment_id: UUID of the experiment record

    Returns:
        dict with status and results
    """
    db = SessionLocal()
    try:
        # Get the experiment record with relationships
        experiment = db.query(Experiment).filter(
            Experiment.id == experiment_id
        ).first()

        if not experiment:
            logger.error(f"Experiment {experiment_id} not found")
            return {"status": "error", "message": "Experiment not found"}

        # Update status to running
        experiment.status = ExperimentStatus.RUNNING
        experiment.celery_task_id = self.request.id
        db.commit()

        # Update task state
        self.update_state(
            state="RUNNING",
            meta={"experiment_id": experiment_id, "progress": 5, "message": "Starting..."}
        )
        logger.info(f"[PROGRESS] Experiment {experiment_id}: 5% - Starting...")

        # Get project and dataset spec
        project = experiment.project
        dataset_spec = experiment.dataset_spec

        if not dataset_spec:
            raise ValueError("Experiment has no dataset spec configured")

        if not project.task_type:
            raise ValueError("Project has no task type configured")

        task_type = project.task_type.value
        logger.info(f"Running AutoML experiment {experiment_id} with task type: {task_type}")

        # Load dataset
        self.update_state(
            state="RUNNING",
            meta={"experiment_id": experiment_id, "progress": 10, "message": "Loading dataset..."}
        )
        logger.info(f"[PROGRESS] Experiment {experiment_id}: 10% - Loading dataset...")

        df = load_dataset_from_spec(db, dataset_spec)
        logger.info(f"[PROGRESS] Experiment {experiment_id}: Loaded dataset with {len(df)} rows, {len(df.columns)} columns")

        # Get target column
        target_column = dataset_spec.target_column or "target"
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available: {list(df.columns)}")

        # Get training config first - needed for validation strategy
        experiment_config = experiment.experiment_plan_json or {}
        automl_config = experiment_config.get("automl_config", {})

        # Get validation strategy from experiment config (set by AI agents)
        validation_strategy = experiment_config.get("validation_strategy", {})
        if validation_strategy:
            logger.info(f"Using agent-specified validation strategy: {validation_strategy.get('split_strategy', 'default')}")
            # Pass validation_strategy to automl_config so the runner uses it
            automl_config["validation_strategy"] = validation_strategy

        # Create/retrieve persistent holdout set for overfitting detection
        holdout_df = None
        try:
            from app.services.holdout_validator import get_or_create_holdout_indices
            df, holdout_df, holdout_indices = get_or_create_holdout_indices(
                df, dataset_spec, db, target_column, task_type,
                validation_strategy=validation_strategy
            )
            logger.info(f"Using holdout validation: {len(holdout_df)} samples held out, {len(df)} for training")
        except Exception as e:
            logger.warning(f"Could not create holdout set: {e}. Training on full dataset.")

        # Set defaults from settings
        if "time_limit" not in automl_config:
            automl_config["time_limit"] = settings.automl_time_limit
        if "presets" not in automl_config:
            automl_config["presets"] = settings.automl_presets

        # Get appropriate runner for task type
        artifacts_dir = os.path.join(settings.artifacts_dir, str(project.id))
        runner = get_runner_for_task(task_type, artifacts_dir)

        # Run training
        self.update_state(
            state="RUNNING",
            meta={"experiment_id": experiment_id, "progress": 20, "message": f"Training {task_type} model..."}
        )
        logger.info(f"[PROGRESS] Experiment {experiment_id}: 20% - Training {task_type} model...")

        result = runner.run_experiment(
            dataset=df,
            target_column=target_column,
            task_type=task_type,
            primary_metric=experiment.primary_metric,
            config=automl_config,
            experiment_id=str(experiment.id),
        )

        # Create model version record
        self.update_state(
            state="RUNNING",
            meta={"experiment_id": experiment_id, "progress": 90, "message": "Saving model..."}
        )
        logger.info(f"[PROGRESS] Experiment {experiment_id}: 90% - Saving model...")

        # Evaluate on holdout set if available (for overfitting detection)
        holdout_score = None
        if holdout_df is not None and len(holdout_df) > 0:
            try:
                from autogluon.tabular import TabularPredictor
                from app.services.holdout_validator import evaluate_on_holdout, record_holdout_score

                # Load the trained predictor
                predictor = TabularPredictor.load(result.artifact_path)

                # Evaluate on holdout
                primary_metric = experiment.primary_metric or "roc_auc"
                holdout_result = evaluate_on_holdout(
                    predictor, holdout_df, target_column, primary_metric, task_type
                )
                holdout_score = holdout_result["score"]

                # Record for overfitting tracking
                validation_result = record_holdout_score(
                    db, experiment, holdout_score, primary_metric
                )

                logger.info(
                    f"Holdout validation: score={holdout_score:.4f}, "
                    f"best={validation_result.best_score:.4f}, "
                    f"recommendation={validation_result.recommendation}"
                )

                # Add holdout score to metrics
                result.metrics["holdout_score"] = holdout_score
                result.metrics["holdout_recommendation"] = validation_result.recommendation

            except Exception as e:
                logger.warning(f"Holdout evaluation failed: {e}")

        model_version = ModelVersion(
            project_id=project.id,
            experiment_id=experiment.id,
            name=f"{experiment.name} - {result.best_model_name}",
            model_type=result.best_model_name,
            artifact_location=result.artifact_path,
            metrics_json=result.metrics,
            feature_importances_json=result.feature_importances,
            status=ModelStatus.TRAINED,
        )
        db.add(model_version)

        # Update experiment status
        experiment.status = ExperimentStatus.COMPLETED
        db.commit()

        logger.info(f"[PROGRESS] Experiment {experiment_id}: 100% - Completed successfully!")

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "model_version_id": str(model_version.id),
            "best_model": result.best_model_name,
            "metrics": result.metrics,
            "training_time": result.training_time_seconds,
            "num_models_trained": result.num_models_trained,
        }

    except Exception as e:
        logger.exception(f"Error running experiment {experiment_id}")

        # Update experiment status to failed
        try:
            experiment = db.query(Experiment).filter(
                Experiment.id == experiment_id
            ).first()
            if experiment:
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = str(e)
                db.commit()
        except Exception:
            pass

        return {"status": "error", "message": str(e)}

    finally:
        db.close()
