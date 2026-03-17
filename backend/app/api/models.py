"""Model version API endpoints."""
import logging
from typing import Literal, Optional
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.project import Project
from app.models.model_version import ModelVersion, ModelStatus
from app.models.validation_sample import ValidationSample
from app.schemas.model_version import (
    ModelVersionCreate,
    ModelVersionResponse,
    ModelPromoteRequest,
    PredictionRequest,
    PredictionResponse,
    ModelExplainRequest,
    ModelExplainResponse,
    ValidationSampleResponse,
    ValidationSamplesListResponse,
    WhatIfRequest,
    WhatIfResponse,
    FeatureStatistics,
    ModelTestingDataResponse,
    RawPredictionRequest,
    RawPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelExportInfoResponse,
    FeaturePipelineInfoResponse,
    RemotePredictionRequest,
    RemotePredictionResponse,
    RemoteModelStatusResponse,
)
from app.services.automl_runner import get_runner_for_task

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


@router.post(
    "/projects/{project_id}/models",
    response_model=ModelVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_model_version(
    project_id: UUID,
    model: ModelVersionCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new model version for a project."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    db_model = ModelVersion(
        project_id=project_id,
        experiment_id=model.experiment_id,
        trial_id=model.trial_id,
        name=model.name,
        model_type=model.model_type,
        artifact_location=model.artifact_location,
        metrics_json=model.metrics_json,
        feature_importances_json=model.feature_importances_json,
        serving_config_json=model.serving_config_json,
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


@router.get(
    "/projects/{project_id}/models",
    response_model=list[ModelVersionResponse],
)
def list_model_versions(
    project_id: UUID,
    status_filter: ModelStatus | None = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List all model versions for a project."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    query = db.query(ModelVersion).filter(ModelVersion.project_id == project_id)
    if status_filter:
        query = query.filter(ModelVersion.status == status_filter)

    return query.all()


@router.get(
    "/models/{model_id}",
    response_model=ModelVersionResponse,
)
def get_model_version(model_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Get a model version by ID."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )
    return model


@router.post(
    "/models/{model_id}/promote",
    response_model=ModelVersionResponse,
)
def promote_model(
    model_id: UUID,
    promote_request: ModelPromoteRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Promote a model to a new status (candidate, shadow, production).

    High-risk models (overfitting_risk="high" or leakage_suspected=true)
    require an override_reason to proceed with promotion. The override
    will be logged in the project's lab notebook for audit purposes.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.models.experiment import Trial
    from app.models.research_cycle import LabNotebookEntry, LabNotebookAuthorType
    from app.services.risk_scoring import get_model_risk_status, format_promotion_block_message

    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Validate status transition
    valid_promote_statuses = [ModelStatus.CANDIDATE, ModelStatus.SHADOW, ModelStatus.PRODUCTION]
    if promote_request.status not in valid_promote_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Can only promote to: {[s.value for s in valid_promote_statuses]}",
        )

    # === Prompt 5: Promotion Gating ===
    # Check for robustness audit data from the associated trial
    risk_info = None
    if model.trial_id:
        trial = db.query(Trial).filter(Trial.id == model.trial_id).first()
        if trial and trial.metrics_json:
            # Extract risk information from trial metrics (populated by robustness audit)
            risk_info = {
                "overfitting_risk": trial.metrics_json.get("overfitting_risk", "unknown"),
                "leakage_suspected": trial.metrics_json.get("leakage_suspected", False),
                "time_split_suspicious": trial.metrics_json.get("time_split_suspicious", False),
                "too_good_to_be_true": trial.metrics_json.get("too_good_to_be_true", False),
                "risk_adjusted_score": trial.metrics_json.get("risk_adjusted_score"),
            }

    # If we have risk info, check if promotion should be blocked
    if risk_info:
        risk_level, requires_override, risk_reason = get_model_risk_status(
            overfitting_risk=risk_info.get("overfitting_risk", "unknown"),
            leakage_suspected=risk_info.get("leakage_suspected", False),
            time_split_suspicious=risk_info.get("time_split_suspicious", False),
            too_good_to_be_true=risk_info.get("too_good_to_be_true", False),
        )

        if requires_override:
            if not promote_request.override_reason:
                # Block promotion - require override
                error_message = format_promotion_block_message(risk_level, risk_reason)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "message": error_message,
                        "risk_level": risk_level,
                        "risk_reason": risk_reason,
                        "requires_override": True,
                    },
                )
            else:
                # Override provided - log to lab notebook
                logger.warning(
                    f"High-risk model {model_id} promoted with override. "
                    f"Risk: {risk_level}, Reason: {promote_request.override_reason}"
                )

                # Create lab notebook entry for audit
                notebook_entry = LabNotebookEntry(
                    project_id=model.project_id,
                    author_type=LabNotebookAuthorType.HUMAN,
                    title=f"High-Risk Model Promotion Override: {model.name}",
                    body_markdown=f"""## Model Promotion Override

**Model:** {model.name} (ID: {model.id})
**Promoted to:** {promote_request.status.value}
**Risk Level:** {risk_level.upper()}

### Identified Risks
{risk_reason}

### Override Justification
{promote_request.override_reason}

---
*This entry was automatically created when a high-risk model was promoted with an override.*
""",
                )
                db.add(notebook_entry)
                logger.info(f"Created lab notebook entry for high-risk promotion override")

    # If promoting to production, demote any existing production model
    if promote_request.status == ModelStatus.PRODUCTION:
        existing_production = (
            db.query(ModelVersion)
            .filter(
                ModelVersion.project_id == model.project_id,
                ModelVersion.status == ModelStatus.PRODUCTION,
                ModelVersion.id != model_id,
            )
            .all()
        )
        for existing in existing_production:
            existing.status = ModelStatus.RETIRED

    model.status = promote_request.status
    db.commit()
    db.refresh(model)
    return model


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model_version(model_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Delete a model version."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    db.delete(model)
    db.commit()
    return None


@router.get("/models/{model_id}/risk-status")
def get_model_risk_status(
    model_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get the risk status for a model to inform promotion decisions.

    Returns risk level, whether override is required, and detailed risk information.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.models.experiment import Trial
    from app.services.risk_scoring import get_model_risk_status as compute_risk_status

    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Default response for models without risk data
    response = {
        "model_id": str(model_id),
        "model_name": model.name,
        "risk_level": "unknown",
        "requires_override": False,
        "risk_reason": "No robustness audit data available",
        "overfitting_risk": "unknown",
        "leakage_suspected": False,
        "time_split_suspicious": False,
        "too_good_to_be_true": False,
        "risk_adjusted_score": None,
    }

    # Check for risk data from associated trial
    if model.trial_id:
        trial = db.query(Trial).filter(Trial.id == model.trial_id).first()
        if trial and trial.metrics_json:
            metrics = trial.metrics_json
            overfitting_risk = metrics.get("overfitting_risk", "unknown")
            leakage_suspected = metrics.get("leakage_suspected", False)
            time_split_suspicious = metrics.get("time_split_suspicious", False)
            too_good_to_be_true = metrics.get("too_good_to_be_true", False)
            risk_adjusted_score = metrics.get("risk_adjusted_score")

            risk_level, requires_override, risk_reason = compute_risk_status(
                overfitting_risk=overfitting_risk,
                leakage_suspected=leakage_suspected,
                time_split_suspicious=time_split_suspicious,
                too_good_to_be_true=too_good_to_be_true,
            )

            response.update({
                "risk_level": risk_level,
                "requires_override": requires_override,
                "risk_reason": risk_reason,
                "overfitting_risk": overfitting_risk,
                "leakage_suspected": leakage_suspected,
                "time_split_suspicious": time_split_suspicious,
                "too_good_to_be_true": too_good_to_be_true,
                "risk_adjusted_score": risk_adjusted_score,
            })

    return response


@router.post(
    "/models/{model_id}/predict",
    response_model=PredictionResponse,
)
def predict(
    model_id: UUID,
    request: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Make predictions using a trained model.

    The features dict should contain values for all features the model expects.
    Feature names and types can be found in the model's serving_config_json.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Validate model has artifact
    if not model.artifact_location:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no artifact location",
        )

    # Validate model has serving config
    serving_config = model.serving_config_json or {}
    expected_features = serving_config.get("features", [])
    task_type = serving_config.get("task_type", "binary")

    if not expected_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no feature configuration in serving_config_json",
        )

    # Validate input features
    expected_names = {f["name"] for f in expected_features}
    provided_names = set(request.features.keys())

    missing = expected_names - provided_names
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing features: {list(missing)}",
        )

    extra = provided_names - expected_names
    if extra:
        logger.warning(f"Extra features provided (will be ignored): {list(extra)}")

    try:
        # Build DataFrame for prediction
        # Only include expected features in the right order
        feature_data = {f["name"]: request.features[f["name"]] for f in expected_features}
        df = pd.DataFrame([feature_data])

        # Get appropriate runner
        runner = get_runner_for_task(task_type)

        # Make prediction
        prediction = runner.predict(model.artifact_location, df)

        # Convert to Python native types
        pred_value = prediction.iloc[0]
        if hasattr(pred_value, 'item'):
            pred_value = pred_value.item()

        response = PredictionResponse(
            prediction=pred_value,
            probabilities=None,
            model_id=model.id,
            model_name=model.name,
        )

        # Get probabilities for classification tasks
        if task_type in ["binary", "multiclass", "classification"]:
            try:
                probas = runner.predict_proba(model.artifact_location, df)
                if probas is not None and not probas.empty:
                    proba_dict = probas.iloc[0].to_dict()
                    # Convert numpy types to Python types
                    proba_dict = {str(k): float(v) for k, v in proba_dict.items()}
                    response.probabilities = proba_dict
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")

        return response

    except Exception as e:
        logger.error(f"Prediction failed for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


def get_llm_provider_and_key(db: Session):
    """Get the LLM provider and API key from database or environment."""
    from app.models.api_key import ApiKey, LLMProvider
    from app.core.config import get_settings

    settings = get_settings()

    # Try providers in order of preference
    providers_to_try = [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]

    for provider in providers_to_try:
        # First check database
        api_key_record = (
            db.query(ApiKey)
            .filter(ApiKey.provider == provider)
            .first()
        )
        if api_key_record and api_key_record.api_key_encrypted:
            return provider, api_key_record.api_key_encrypted

        # Fallback to environment variables
        if provider == LLMProvider.ANTHROPIC and settings.anthropic_api_key:
            return provider, settings.anthropic_api_key
        elif provider == LLMProvider.OPENAI and settings.openai_api_key:
            return provider, settings.openai_api_key

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No LLM API key configured. Please add an API key in settings.",
    )


@router.post(
    "/models/{model_id}/explain",
    response_model=ModelExplainResponse,
)
async def explain_model(
    model_id: UUID,
    request: ModelExplainRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Ask questions about a model and get AI-powered explanations.

    The explainer agent has access to:
    - Model metrics and performance data
    - Feature importances
    - Training configuration

    Example questions:
    - "Which features are most important?"
    - "Why does the model perform well/poorly?"
    - "How can I improve the model?"
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.services.llm_client import get_llm_client

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Gather model context
    context_parts = [f"# Model: {model.name}"]
    context_parts.append(f"Type: {model.model_type or 'Unknown'}")
    context_parts.append(f"Status: {model.status.value}")

    # Add metrics
    if model.metrics_json:
        context_parts.append("\n## Performance Metrics")
        for key, value in model.metrics_json.items():
            if isinstance(value, float):
                context_parts.append(f"- {key}: {value:.4f}")
            else:
                context_parts.append(f"- {key}: {value}")

    # Add feature importances
    if model.feature_importances_json:
        context_parts.append("\n## Feature Importances")
        # Sort by importance
        sorted_features = sorted(
            model.feature_importances_json.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )
        for feat, importance in sorted_features[:15]:  # Top 15
            if isinstance(importance, float):
                context_parts.append(f"- {feat}: {importance:.4f}")
            else:
                context_parts.append(f"- {feat}: {importance}")

    # Add serving config info
    if model.serving_config_json:
        serving = model.serving_config_json
        context_parts.append(f"\n## Configuration")
        context_parts.append(f"- Task type: {serving.get('task_type', 'Unknown')}")
        context_parts.append(f"- Target column: {serving.get('target_column', 'Unknown')}")
        features = serving.get('features', [])
        context_parts.append(f"- Number of features: {len(features)}")
        if features:
            feature_types = {}
            for f in features:
                ftype = f.get('type', 'unknown')
                feature_types[ftype] = feature_types.get(ftype, 0) + 1
            context_parts.append(f"- Feature types: {feature_types}")

    model_context = "\n".join(context_parts)

    # Build prompt for the explainer agent
    system_prompt = """You are an expert ML model explainer. You help users understand their machine learning models.

Given the model information below, answer the user's question clearly and helpfully.
Use markdown formatting for better readability.
Be concise but thorough. If you don't have enough information to answer fully, say so.

Focus on:
- Explaining metrics in plain language
- Interpreting feature importances
- Suggesting potential improvements
- Identifying potential issues or biases"""

    user_prompt = f"""## Model Information

{model_context}

## User Question

{request.question}

Please provide a clear, helpful answer."""

    try:
        # Get LLM provider and key
        provider, api_key_str = get_llm_provider_and_key(db)
        client = get_llm_client(provider, api_key_str)

        # Build messages for chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Get response from LLM
        answer = await client.chat(messages)

        return ModelExplainResponse(
            answer=answer,
            model_id=model.id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model explanation failed for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}",
        )


# Validation Samples Endpoints

@router.get(
    "/models/{model_id}/validation-samples",
    response_model=ValidationSamplesListResponse,
)
def list_validation_samples(
    model_id: UUID,
    limit: int = Query(default=50, ge=1, le=500, description="Number of samples per page"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    sort: Literal["error_desc", "error_asc", "row_index", "random"] = Query(
        default="error_desc", description="Sort order"
    ),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List validation samples for a model.

    Returns paginated validation samples with various sort options:
    - error_desc: Highest absolute errors first (most interesting errors)
    - error_asc: Lowest absolute errors first
    - row_index: Original row order
    - random: Random sample
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify model exists
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Get total count
    total = db.query(ValidationSample).filter(
        ValidationSample.model_version_id == model_id
    ).count()

    # Build query with sorting
    query = db.query(ValidationSample).filter(ValidationSample.model_version_id == model_id)

    if sort == "error_desc":
        query = query.order_by(ValidationSample.absolute_error.desc().nullslast())
    elif sort == "error_asc":
        query = query.order_by(ValidationSample.absolute_error.asc().nullsfirst())
    elif sort == "row_index":
        query = query.order_by(ValidationSample.row_index.asc())
    elif sort == "random":
        from sqlalchemy.sql.expression import func
        query = query.order_by(func.random())

    # Apply pagination
    samples = query.offset(offset).limit(limit).all()

    return ValidationSamplesListResponse(
        model_id=model_id,
        total=total,
        limit=limit,
        offset=offset,
        samples=[ValidationSampleResponse.from_db_model(s) for s in samples],
    )


@router.get(
    "/validation-samples/{sample_id}",
    response_model=ValidationSampleResponse,
)
def get_validation_sample(
    sample_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a single validation sample by ID."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    sample = db.query(ValidationSample).filter(ValidationSample.id == sample_id).first()
    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Validation sample {sample_id} not found",
        )
    return ValidationSampleResponse.from_db_model(sample)


@router.post(
    "/models/{model_id}/what-if",
    response_model=WhatIfResponse,
)
def what_if_prediction(
    model_id: UUID,
    request: WhatIfRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Run a what-if prediction by modifying features from an existing validation sample.

    This allows exploring "what would happen if feature X was Y?" scenarios
    by taking an existing sample, modifying specific features, and re-running prediction.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify model exists and has artifact
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if not model.artifact_location:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no artifact location",
        )

    # Get the original validation sample
    sample = db.query(ValidationSample).filter(ValidationSample.id == request.sample_id).first()
    if not sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Validation sample {request.sample_id} not found",
        )

    # Verify sample belongs to this model
    if sample.model_version_id != model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation sample {request.sample_id} does not belong to model {model_id}",
        )

    # Validate model has serving config
    serving_config = model.serving_config_json or {}
    expected_features = serving_config.get("features", [])
    task_type = serving_config.get("task_type", "binary")

    if not expected_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no feature configuration in serving_config_json",
        )

    # Validate modified features are valid feature names
    expected_names = {f["name"] for f in expected_features}
    modified_names = set(request.modified_features.keys())
    invalid_features = modified_names - expected_names
    if invalid_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid feature names: {list(invalid_features)}. Valid features: {list(expected_names)}",
        )

    try:
        # Build merged feature dict (original + modifications)
        original_features = sample.features_json.copy()
        modified_features = {**original_features, **request.modified_features}

        # Build DataFrame for prediction
        feature_data = {f["name"]: modified_features.get(f["name"]) for f in expected_features}
        df = pd.DataFrame([feature_data])

        # Get appropriate runner
        runner = get_runner_for_task(task_type)

        # Make prediction with modified features
        prediction = runner.predict(model.artifact_location, df)
        modified_pred_value = prediction.iloc[0]
        if hasattr(modified_pred_value, 'item'):
            modified_pred_value = modified_pred_value.item()

        # Get original prediction value (stored in sample)
        original_pred_value = sample.predicted_value
        # Try to convert to numeric for delta calculation
        try:
            original_numeric = float(original_pred_value)
            modified_numeric = float(modified_pred_value)
            prediction_delta = modified_numeric - original_numeric
        except (ValueError, TypeError):
            prediction_delta = None

        # Build response
        response = WhatIfResponse(
            original_sample=ValidationSampleResponse.from_db_model(sample),
            modified_features=request.modified_features,
            original_prediction=original_pred_value,
            modified_prediction=modified_pred_value,
            prediction_delta=prediction_delta,
            original_probabilities=sample.prediction_probabilities_json,
            modified_probabilities=None,
        )

        # Get probabilities for classification tasks
        if task_type in ["binary", "multiclass", "classification"]:
            try:
                probas = runner.predict_proba(model.artifact_location, df)
                if probas is not None and not probas.empty:
                    proba_dict = probas.iloc[0].to_dict()
                    # Convert numpy types to Python types
                    proba_dict = {str(k): float(v) for k, v in proba_dict.items()}
                    response.modified_probabilities = proba_dict
            except Exception as e:
                logger.warning(f"Could not get probabilities for what-if: {e}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"What-if prediction failed for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"What-if prediction failed: {str(e)}",
        )


@router.get(
    "/models/{model_id}/testing-data",
    response_model=ModelTestingDataResponse,
)
def get_model_testing_data(
    model_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get feature statistics and sample data for model testing UI.

    Returns:
    - Feature statistics (min/max/median for numeric, categories for categorical)
    - Feature importance ranking
    - A random sample from validation data as starting point
    - Top important features for quick access
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    import random
    from statistics import mean, median

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Get serving config
    serving_config = model.serving_config_json or {}
    serving_features = serving_config.get("features", [])
    task_type = serving_config.get("task_type", "binary")
    target_column = serving_config.get("target_column")

    if not serving_features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no feature configuration",
        )

    # Get feature importances
    feature_importances = model.feature_importances_json or {}

    # Get validation samples for statistics
    samples = db.query(ValidationSample).filter(
        ValidationSample.model_version_id == model_id
    ).limit(1000).all()  # Limit to 1000 for performance

    sample_count = db.query(ValidationSample).filter(
        ValidationSample.model_version_id == model_id
    ).count()

    # Build feature statistics
    feature_stats_list = []
    for feat in serving_features:
        feat_name = feat["name"]
        feat_type = feat.get("type", "unknown")

        stats = FeatureStatistics(
            name=feat_name,
            type=feat_type,
            importance=feature_importances.get(feat_name),
        )

        # Calculate stats from validation samples
        if samples:
            values = [s.features_json.get(feat_name) for s in samples if s.features_json.get(feat_name) is not None]

            if feat_type == "numeric" and values:
                try:
                    numeric_values = [float(v) for v in values if v is not None]
                    if numeric_values:
                        stats.min_value = min(numeric_values)
                        stats.max_value = max(numeric_values)
                        stats.mean_value = mean(numeric_values)
                        stats.median_value = median(numeric_values)
                except (ValueError, TypeError):
                    pass

            elif feat_type in ["categorical", "string", "object"] and values:
                try:
                    str_values = [str(v) for v in values]
                    unique_values = list(set(str_values))
                    # Limit categories for UI
                    stats.categories = sorted(unique_values)[:50]
                    # Find most common
                    from collections import Counter
                    counter = Counter(str_values)
                    stats.most_common = counter.most_common(1)[0][0] if counter else None
                except Exception:
                    pass

            elif feat_type == "boolean":
                stats.categories = ["true", "false"]

        feature_stats_list.append(stats)

    # Sort features by importance for top_features list
    sorted_by_importance = sorted(
        feature_stats_list,
        key=lambda f: f.importance if f.importance is not None else -1,
        reverse=True
    )
    top_features = [f.name for f in sorted_by_importance[:10]]

    # Get a random sample for starting point
    sample_data = None
    if samples:
        random_sample = random.choice(samples)
        sample_data = random_sample.features_json

    return ModelTestingDataResponse(
        model_id=model.id,
        model_name=model.name,
        task_type=task_type,
        target_column=target_column,
        features=feature_stats_list,
        top_features=top_features,
        sample_data=sample_data,
        has_validation_samples=len(samples) > 0,
        validation_sample_count=sample_count,
    )


@router.get(
    "/models/{model_id}/random-sample",
    response_model=dict,
)
def get_random_sample(
    model_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a random sample from validation data for model testing.

    Returns just the feature values as a dict that can be used directly
    in the prediction form.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    import random

    # Verify model exists
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Get random validation sample
    samples = db.query(ValidationSample).filter(
        ValidationSample.model_version_id == model_id
    ).limit(100).all()

    if not samples:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No validation samples available for this model",
        )

    random_sample = random.choice(samples)

    return {
        "features": random_sample.features_json,
        "sample_id": str(random_sample.id),
        "actual_value": random_sample.target_value,
        "predicted_value": random_sample.predicted_value,
    }


# =============================================================================
# RAW DATA PREDICTION ENDPOINTS
# =============================================================================


@router.post(
    "/models/{model_id}/predict-raw",
    response_model=RawPredictionResponse,
)
def predict_raw(
    model_id: UUID,
    request: RawPredictionRequest,
    include_transformed: bool = Query(
        default=False, description="Include transformed features in response (for debugging)"
    ),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Make predictions using raw, untransformed data.

    This endpoint automatically applies any feature engineering transformations
    that were used during training before making predictions.

    Use this when you have original data (e.g., from a CSV) that needs to be
    transformed before the model can make predictions.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.services.feature_pipeline import get_pipeline_for_model

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Validate model has artifact
    if not model.artifact_location:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no artifact location - cannot make predictions",
        )

    serving_config = model.serving_config_json or {}
    task_type = serving_config.get("task_type", "binary")

    try:
        # Get the feature pipeline
        pipeline = get_pipeline_for_model(db, str(model_id))

        # Create DataFrame from raw input
        raw_df = pd.DataFrame([request.data])

        transformations_applied = False
        transformed_df = raw_df

        # Apply transformations if pipeline exists and has transformations
        if pipeline and pipeline.has_transformations():
            logger.info(f"Applying feature pipeline for model {model_id}")
            transformed_df = pipeline.transform(raw_df, create_target=False, strict=False)
            transformations_applied = True

        # Get expected features from serving config
        expected_features = serving_config.get("features", [])
        if expected_features:
            expected_names = [f["name"] for f in expected_features]
            # Keep only features the model expects
            available_expected = [c for c in expected_names if c in transformed_df.columns]
            if available_expected:
                transformed_df = transformed_df[available_expected]

        # Get appropriate runner
        runner = get_runner_for_task(task_type)

        # Make prediction
        prediction = runner.predict(model.artifact_location, transformed_df)

        # Convert to Python native types
        pred_value = prediction.iloc[0]
        if hasattr(pred_value, 'item'):
            pred_value = pred_value.item()

        response = RawPredictionResponse(
            prediction=pred_value,
            probabilities=None,
            model_id=model.id,
            model_name=model.name,
            transformations_applied=transformations_applied,
            transformed_features=transformed_df.iloc[0].to_dict() if include_transformed else None,
        )

        # Get probabilities for classification tasks
        if task_type in ["binary", "multiclass", "classification"]:
            try:
                probas = runner.predict_proba(model.artifact_location, transformed_df)
                if probas is not None and not probas.empty:
                    proba_dict = probas.iloc[0].to_dict()
                    proba_dict = {str(k): float(v) for k, v in proba_dict.items()}
                    response.probabilities = proba_dict
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")

        return response

    except Exception as e:
        logger.error(f"Raw prediction failed for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/models/{model_id}/predict-batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    model_id: UUID,
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Make batch predictions using raw data.

    Submit multiple records at once for efficient batch prediction.
    Feature transformations are applied automatically.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.services.feature_pipeline import get_pipeline_for_model

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if not model.artifact_location:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no artifact location",
        )

    serving_config = model.serving_config_json or {}
    task_type = serving_config.get("task_type", "binary")

    try:
        # Get the feature pipeline
        pipeline = get_pipeline_for_model(db, str(model_id))

        # Create DataFrame from batch input
        raw_df = pd.DataFrame(request.data)

        # Apply transformations if pipeline exists
        if pipeline and pipeline.has_transformations():
            raw_df = pipeline.transform(raw_df, create_target=False, strict=False)

        # Filter to expected features
        expected_features = serving_config.get("features", [])
        if expected_features:
            expected_names = [f["name"] for f in expected_features]
            available_expected = [c for c in expected_names if c in raw_df.columns]
            if available_expected:
                raw_df = raw_df[available_expected]

        # Get appropriate runner
        runner = get_runner_for_task(task_type)

        # Make predictions
        predictions = runner.predict(model.artifact_location, raw_df)
        pred_list = predictions.tolist()

        response = BatchPredictionResponse(
            predictions=pred_list,
            probabilities=None,
            model_id=model.id,
            model_name=model.name,
            count=len(pred_list),
        )

        # Get probabilities for classification
        if task_type in ["binary", "multiclass", "classification"]:
            try:
                probas = runner.predict_proba(model.artifact_location, raw_df)
                if probas is not None and not probas.empty:
                    proba_list = probas.to_dict(orient="records")
                    proba_list = [{str(k): float(v) for k, v in p.items()} for p in proba_list]
                    response.probabilities = proba_list
            except Exception as e:
                logger.warning(f"Could not get batch probabilities: {e}")

        return response

    except Exception as e:
        logger.error(f"Batch prediction failed for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


# =============================================================================
# FEATURE PIPELINE ENDPOINT
# =============================================================================


@router.get(
    "/models/{model_id}/pipeline",
    response_model=FeaturePipelineInfoResponse,
)
def get_model_pipeline(
    model_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get information about the feature transformation pipeline for a model.

    Returns details about what transformations are applied to raw data
    before predictions, including required input columns.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.services.feature_pipeline import get_pipeline_for_model

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Get pipeline
    pipeline = get_pipeline_for_model(db, str(model_id))

    if not pipeline:
        return FeaturePipelineInfoResponse(
            model_id=model.id,
            has_transformations=False,
            transformation_count=0,
            required_input_columns=[],
            output_columns=[],
            target_column=None,
            pipeline_config=None,
        )

    return FeaturePipelineInfoResponse(
        model_id=model.id,
        has_transformations=pipeline.has_transformations(),
        transformation_count=len(pipeline.config.feature_engineering),
        required_input_columns=pipeline.get_required_columns(),
        output_columns=pipeline.get_output_columns(),
        target_column=pipeline.config.target_column,
        pipeline_config=pipeline.config.to_dict(),
    )


# =============================================================================
# MODEL EXPORT ENDPOINTS
# =============================================================================


@router.get(
    "/models/{model_id}/export-info",
    response_model=ModelExportInfoResponse,
)
def get_export_info(
    model_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get information about exporting a model.

    Returns whether the model can be exported, estimated size,
    and required packages.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    import os
    from pathlib import Path

    from app.services.feature_pipeline import get_pipeline_for_model

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    serving_config = model.serving_config_json or {}
    task_type = serving_config.get("task_type", "unknown")

    # Check if model can be exported
    can_export = bool(model.artifact_location and os.path.exists(model.artifact_location))

    # Estimate size
    export_size_mb = None
    if can_export:
        try:
            total_size = 0
            artifact_path = Path(model.artifact_location)
            for f in artifact_path.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
            export_size_mb = total_size / (1024 * 1024)
        except Exception:
            pass

    # Check for pipeline
    pipeline = get_pipeline_for_model(db, str(model_id))
    has_pipeline = pipeline is not None and pipeline.has_transformations()

    # Determine required packages
    required_packages = ["pandas", "numpy"]
    model_type = model.model_type or "unknown"
    if "autogluon" in model_type.lower() or model_type in ["LightGBM", "CatBoost", "XGBoost", "NeuralNetTorch"]:
        required_packages.append("autogluon.tabular")
    if "lightgbm" in model_type.lower():
        required_packages.append("lightgbm")
    if "xgboost" in model_type.lower():
        required_packages.append("xgboost")
    if "catboost" in model_type.lower():
        required_packages.append("catboost")

    return ModelExportInfoResponse(
        model_id=model.id,
        model_name=model.name,
        model_type=model_type,
        task_type=task_type,
        can_export=can_export,
        export_size_mb=export_size_mb,
        has_pipeline=has_pipeline,
        required_packages=required_packages,
    )


@router.get(
    "/models/{model_id}/export",
)
def export_model(
    model_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Export a model as a downloadable ZIP file.

    The ZIP contains:
    - Model files (AutoGluon predictor)
    - pipeline_config.json (feature transformations)
    - predict.py (standalone prediction script)
    - requirements.txt (Python dependencies)
    - README.md (usage instructions)
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    import io
    import os
    import zipfile
    from pathlib import Path

    from fastapi.responses import StreamingResponse

    from app.services.feature_pipeline import get_pipeline_for_model, generate_predict_script

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if not model.artifact_location or not os.path.exists(model.artifact_location):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model artifacts not available for export",
        )

    serving_config = model.serving_config_json or {}
    task_type = serving_config.get("task_type", "binary")
    model_type = model.model_type or "autogluon"

    # Get pipeline
    pipeline = get_pipeline_for_model(db, str(model_id))
    pipeline_config = pipeline.config.to_dict() if pipeline else {}

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add model files
        artifact_path = Path(model.artifact_location)
        for file_path in artifact_path.rglob("*"):
            if file_path.is_file():
                arc_name = f"model/{file_path.relative_to(artifact_path)}"
                zf.write(file_path, arc_name)

        # Add pipeline config
        import json
        zf.writestr("pipeline_config.json", json.dumps(pipeline_config, indent=2))

        # Add predict script
        predict_script = generate_predict_script(pipeline_config, "autogluon")
        zf.writestr("predict.py", predict_script)

        # Add requirements.txt
        requirements = """# Model dependencies
pandas>=2.0.0
numpy>=1.24.0
autogluon.tabular>=1.0.0
"""
        zf.writestr("requirements.txt", requirements)

        # Add README
        readme = f"""# Exported Model: {model.name}

## Model Information
- Model ID: {model.id}
- Model Type: {model_type}
- Task Type: {task_type}

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make predictions:
   ```bash
   # Single prediction from JSON
   python predict.py --json '{{"feature1": value1, "feature2": value2}}' --model-path ./model

   # Batch prediction from CSV
   python predict.py input.csv output.csv --model-path ./model
   ```

## Files
- `model/` - AutoGluon model files
- `pipeline_config.json` - Feature transformation configuration
- `predict.py` - Standalone prediction script
- `requirements.txt` - Python dependencies

## Feature Transformations
{"This model includes feature transformations that are automatically applied." if pipeline and pipeline.has_transformations() else "This model does not require feature transformations."}

Required input columns: {pipeline.get_required_columns() if pipeline else "See model.serving_config for features"}

## Usage in Python

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load model
predictor = TabularPredictor.load("./model")

# Load and transform your data
df = pd.read_csv("your_data.csv")

# If using pipeline transformations, apply them first
# (see predict.py for the transformation code)

# Make predictions
predictions = predictor.predict(df)
```
"""
        zf.writestr("README.md", readme)

    # Prepare response
    zip_buffer.seek(0)

    # Create safe filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model.name)
    filename = f"{safe_name}_{model.id}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# =============================================================================
# REMOTE PREDICTION ENDPOINTS (Modal Cloud)
# =============================================================================


@router.get(
    "/models/{model_id}/remote-status",
    response_model=RemoteModelStatusResponse,
)
def get_remote_model_status(
    model_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Check if a model is available for remote predictions on Modal.

    This checks:
    1. Whether the model has an associated experiment ID
    2. Whether the model exists on the Modal Volume
    3. Whether local prediction is available
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    import os

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Check local availability
    can_predict_locally = bool(
        model.artifact_location and os.path.exists(model.artifact_location)
    )

    # Get experiment_id from model
    experiment_id = str(model.experiment_id) if model.experiment_id else None

    # Check remote availability by calling Modal
    exists_on_volume = False
    model_size_mb = None

    if experiment_id:
        try:
            import modal
            from modal_training_standalone import check_model_exists

            with modal.enable_output():
                result = check_model_exists.remote(experiment_id)

            exists_on_volume = result.get("exists", False)
            model_size_mb = result.get("model_size_mb")
        except Exception as e:
            logger.warning(f"Could not check remote model status: {e}")

    return RemoteModelStatusResponse(
        model_id=model.id,
        experiment_id=experiment_id,
        exists_on_volume=exists_on_volume,
        model_size_mb=model_size_mb,
        can_predict_locally=can_predict_locally,
        can_predict_remotely=exists_on_volume,
    )


@router.post(
    "/models/{model_id}/predict-remote",
    response_model=RemotePredictionResponse,
)
def predict_remote(
    model_id: UUID,
    request: RemotePredictionRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Make predictions using a model stored on Modal Volume.

    Use this endpoint when the model is too large to download locally.
    The prediction runs on Modal cloud infrastructure.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    import json

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Get experiment_id
    experiment_id = str(model.experiment_id) if model.experiment_id else None
    if not experiment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no experiment_id - cannot use remote prediction",
        )

    # Validate input features against serving config
    serving_config = model.serving_config_json or {}
    expected_features = serving_config.get("features", [])

    if expected_features:
        expected_names = {f["name"] for f in expected_features}
        provided_names = set(request.features.keys())

        missing = expected_names - provided_names
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing features: {list(missing)}",
            )

    try:
        import modal
        from modal_training_standalone import predict_remote as modal_predict

        # Serialize input data
        input_data = [request.features]
        input_json = json.dumps(input_data)

        logger.info(f"Making remote prediction for model {model_id} via Modal")

        with modal.enable_output():
            result = modal_predict.remote(experiment_id, input_json)

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Remote prediction failed: {result.get('error', 'Unknown error')}",
            )

        predictions = result.get("predictions", [])
        probabilities = result.get("probabilities")

        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Remote prediction returned no results",
            )

        return RemotePredictionResponse(
            prediction=predictions[0],
            probabilities=probabilities,
            model_id=model.id,
            model_name=model.name,
            is_remote=True,
            experiment_id=experiment_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remote prediction failed for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Remote prediction failed: {str(e)}",
        )


@router.post(
    "/models/{model_id}/predict-auto",
    response_model=PredictionResponse,
)
def predict_auto(
    model_id: UUID,
    request: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Make predictions using either local or remote model automatically.

    This endpoint will:
    1. Try local prediction first if model artifacts are available
    2. Fall back to remote prediction on Modal if local is unavailable

    Use this as the default prediction endpoint - it handles both cases.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    import os
    import json

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Check if local prediction is available
    can_predict_locally = bool(
        model.artifact_location and os.path.exists(model.artifact_location)
    )

    if can_predict_locally:
        # Use local prediction (existing logic from predict endpoint)
        serving_config = model.serving_config_json or {}
        expected_features = serving_config.get("features", [])
        task_type = serving_config.get("task_type", "binary")

        if not expected_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model has no feature configuration in serving_config_json",
            )

        expected_names = {f["name"] for f in expected_features}
        provided_names = set(request.features.keys())

        missing = expected_names - provided_names
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing features: {list(missing)}",
            )

        try:
            feature_data = {f["name"]: request.features[f["name"]] for f in expected_features}
            df = pd.DataFrame([feature_data])

            runner = get_runner_for_task(task_type)
            prediction = runner.predict(model.artifact_location, df)

            pred_value = prediction.iloc[0]
            if hasattr(pred_value, 'item'):
                pred_value = pred_value.item()

            response = PredictionResponse(
                prediction=pred_value,
                probabilities=None,
                model_id=model.id,
                model_name=model.name,
            )

            if task_type in ["binary", "multiclass", "classification"]:
                try:
                    probas = runner.predict_proba(model.artifact_location, df)
                    if probas is not None and not probas.empty:
                        proba_dict = probas.iloc[0].to_dict()
                        proba_dict = {str(k): float(v) for k, v in proba_dict.items()}
                        response.probabilities = proba_dict
                except Exception as e:
                    logger.warning(f"Could not get probabilities: {e}")

            return response

        except Exception as e:
            logger.error(f"Local prediction failed for model {model_id}: {e}")
            # Fall through to try remote
            pass

    # Try remote prediction
    experiment_id = str(model.experiment_id) if model.experiment_id else None
    if not experiment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not available locally and has no experiment_id for remote prediction",
        )

    try:
        import modal
        from modal_training_standalone import predict_remote as modal_predict

        input_data = [request.features]
        input_json = json.dumps(input_data)

        logger.info(f"Falling back to remote prediction for model {model_id}")

        with modal.enable_output():
            result = modal_predict.remote(experiment_id, input_json)

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Remote prediction failed: {result.get('error', 'Unknown error')}",
            )

        predictions = result.get("predictions", [])
        probabilities = result.get("probabilities")

        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Remote prediction returned no results",
            )

        # Convert probabilities format
        proba_dict = None
        if probabilities and len(probabilities) > 0:
            proba_dict = probabilities[0]

        return PredictionResponse(
            prediction=predictions[0],
            probabilities=proba_dict,
            model_id=model.id,
            model_name=model.name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remote prediction failed for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed (both local and remote): {str(e)}",
        )
