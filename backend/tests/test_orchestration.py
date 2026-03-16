"""Tests for Pipeline Orchestration: Project Manager, Debate System, and OpenAI Judge."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from app.services.agents.orchestration import (
    ORCHESTRABLE_AGENTS,
    AGENT_DESCRIPTIONS,
    CRITIQUE_SPECIALIZATIONS,
    AVAILABLE_JUDGE_MODELS,
    DEFAULT_JUDGE_MODEL,
    get_available_judge_models,
    validate_judge_model,
    DebateResult,
    DebateMessage,
)
from app.models.agent_run import PipelineOrchestrationMode, DebateMode


# ============================================
# Project Manager Constants Tests
# ============================================

class TestProjectManagerConstants:
    """Tests for Project Manager constants."""

    def test_orchestrable_agents_defined(self):
        """Test that orchestrable agents are defined."""
        assert len(ORCHESTRABLE_AGENTS) > 0
        # Should include key pipeline steps
        assert "data_analysis" in ORCHESTRABLE_AGENTS
        assert "problem_understanding" in ORCHESTRABLE_AGENTS

    def test_agent_descriptions_match_agents(self):
        """Test that descriptions exist for all orchestrable agents."""
        for agent in ORCHESTRABLE_AGENTS:
            assert agent in AGENT_DESCRIPTIONS, f"Missing description for {agent}"
            # Each description should be a non-empty string
            assert isinstance(AGENT_DESCRIPTIONS[agent], str)
            assert len(AGENT_DESCRIPTIONS[agent]) > 10


# ============================================
# Gemini Critique Constants Tests
# ============================================

class TestGeminiCritiqueConstants:
    """Tests for Gemini Critique Agent constants."""

    def test_critique_specializations_defined(self):
        """Test that critique specializations are defined."""
        assert len(CRITIQUE_SPECIALIZATIONS) > 0

    def test_critique_specializations_have_required_fields(self):
        """Test that each specialization has required fields."""
        for agent_type, spec in CRITIQUE_SPECIALIZATIONS.items():
            assert "name" in spec, f"Missing name for {agent_type}"
            assert "expertise" in spec, f"Missing expertise for {agent_type}"


# ============================================
# OpenAI Judge Constants Tests
# ============================================

class TestOpenAIJudgeConstants:
    """Tests for OpenAI Judge Agent constants."""

    def test_available_judge_models(self):
        """Test that judge models are defined."""
        models = get_available_judge_models()
        assert len(models) > 0
        assert DEFAULT_JUDGE_MODEL in models
        assert "gpt-4o" in models

    def test_validate_judge_model_valid(self):
        """Test validating a valid judge model."""
        assert validate_judge_model("gpt-4o") is True
        assert validate_judge_model("o3") is True
        assert validate_judge_model("gpt-5.1-thinking") is True
        assert validate_judge_model(DEFAULT_JUDGE_MODEL) is True

    def test_validate_judge_model_invalid(self):
        """Test validating an invalid judge model."""
        assert validate_judge_model("invalid-model") is False
        assert validate_judge_model("") is False
        assert validate_judge_model("claude-3") is False

    def test_available_models_list(self):
        """Test available models list."""
        assert isinstance(AVAILABLE_JUDGE_MODELS, list)
        # Should have multiple models
        assert len(AVAILABLE_JUDGE_MODELS) >= 3
        # All should be strings
        for model in AVAILABLE_JUDGE_MODELS:
            assert isinstance(model, str)


# ============================================
# Orchestration Mode Enum Tests
# ============================================

class TestOrchestrationModes:
    """Tests for pipeline orchestration mode enums."""

    def test_orchestration_mode_values(self):
        """Test PipelineOrchestrationMode enum values."""
        assert PipelineOrchestrationMode.SEQUENTIAL.value == "sequential"
        assert PipelineOrchestrationMode.PROJECT_MANAGER.value == "project_manager"

    def test_debate_mode_values(self):
        """Test DebateMode enum values."""
        assert DebateMode.DISABLED.value == "disabled"
        assert DebateMode.ENABLED.value == "enabled"

    def test_orchestration_mode_iteration(self):
        """Test that we can iterate over orchestration modes."""
        modes = list(PipelineOrchestrationMode)
        assert len(modes) == 2
        assert PipelineOrchestrationMode.SEQUENTIAL in modes
        assert PipelineOrchestrationMode.PROJECT_MANAGER in modes


# ============================================
# Debate Dataclass Tests
# ============================================

class TestDebateDataclasses:
    """Tests for debate dataclass structures."""

    def test_debate_message_structure(self):
        """Test DebateMessage dataclass structure."""
        msg = DebateMessage(
            role="main_agent",
            content="Test content",
            round=1,
            agrees=True,
            confidence=80,
            metadata={"key": "value"}
        )

        assert msg.role == "main_agent"
        assert msg.content == "Test content"
        assert msg.round == 1
        assert msg.agrees is True
        assert msg.confidence == 80
        assert msg.metadata == {"key": "value"}

    def test_debate_message_defaults(self):
        """Test DebateMessage default values."""
        msg = DebateMessage(
            role="critique_agent",
            content="Critique text",
            round=2,
        )

        assert msg.agrees is None
        assert msg.confidence is None
        assert msg.metadata == {}

    def test_debate_result_consensus(self):
        """Test DebateResult with consensus."""
        result = DebateResult(
            consensus_reached=True,
            final_output={"result": "test"},
            winner="consensus",
            transcript=[],
            total_rounds=2,
            judge_decision=None,
            summary="Agreement reached"
        )

        assert result.consensus_reached is True
        assert result.winner == "consensus"
        assert result.total_rounds == 2
        assert result.judge_decision is None

    def test_debate_result_no_consensus(self):
        """Test DebateResult when judge decides."""
        judge_decision = {
            "decision": "main_agent",
            "reasoning": "Main agent had better analysis",
            "confidence": 85,
        }

        result = DebateResult(
            consensus_reached=False,
            final_output={"result": "judged"},
            winner="main_agent",
            transcript=[
                DebateMessage(role="main_agent", content="Position 1", round=1),
                DebateMessage(role="critique_agent", content="Critique 1", round=1, agrees=False),
            ],
            total_rounds=3,
            judge_decision=judge_decision,
            summary="Judge decided in favor of main agent"
        )

        assert result.consensus_reached is False
        assert result.winner == "main_agent"
        assert result.total_rounds == 3
        assert result.judge_decision is not None
        assert result.judge_decision["decision"] == "main_agent"
        assert len(result.transcript) == 2


# ============================================
# API Endpoint Tests (using test client)
# ============================================

class TestOrchestrationEndpoints:
    """Tests for orchestration API endpoints."""

    @pytest.mark.asyncio
    async def test_get_orchestration_options_response_format(self):
        """Test that orchestration options have correct structure."""
        from app.api.agent import get_orchestration_options

        # Call the endpoint function directly (async)
        options = await get_orchestration_options()

        # Check attributes exist (Pydantic model, not dict)
        assert hasattr(options, "orchestration_modes")
        assert hasattr(options, "debate_modes")
        assert hasattr(options, "judge_models")
        assert hasattr(options, "default_judge_model")

        # Check types
        assert isinstance(options.orchestration_modes, list)
        assert isinstance(options.debate_modes, list)
        assert isinstance(options.judge_models, list)
        assert isinstance(options.default_judge_model, str)

        # Check values
        assert "sequential" in options.orchestration_modes
        assert "project_manager" in options.orchestration_modes
        assert "disabled" in options.debate_modes
        assert "enabled" in options.debate_modes
        assert options.default_judge_model in options.judge_models

    @pytest.mark.asyncio
    async def test_get_judge_models_response_format(self):
        """Test judge models endpoint response format."""
        from app.api.agent import get_judge_models

        result = await get_judge_models()

        # Check attributes exist (Pydantic model, not dict)
        assert hasattr(result, "models")
        assert hasattr(result, "default")
        assert isinstance(result.models, list)
        assert len(result.models) > 0
        assert result.default in result.models


# ============================================
# Database Model Tests
# ============================================

class TestOrchestrationDatabaseFields:
    """Tests for orchestration database model fields."""

    def test_agent_run_orchestration_fields_exist(self, db_session):
        """Test that AgentRun has orchestration fields."""
        from app.models import AgentRun

        # Create an agent run
        run = AgentRun(
            name="Test Orchestration Run",
            orchestration_mode=PipelineOrchestrationMode.PROJECT_MANAGER,
            debate_mode=DebateMode.ENABLED,
            judge_model="gpt-4o",
        )

        db_session.add(run)
        db_session.flush()

        assert run.orchestration_mode == PipelineOrchestrationMode.PROJECT_MANAGER
        assert run.debate_mode == DebateMode.ENABLED
        assert run.judge_model == "gpt-4o"

    def test_agent_run_default_values(self, db_session):
        """Test AgentRun default orchestration values."""
        from app.models import AgentRun

        run = AgentRun(name="Test Default Values")

        db_session.add(run)
        db_session.flush()

        assert run.orchestration_mode == PipelineOrchestrationMode.SEQUENTIAL
        assert run.debate_mode == DebateMode.DISABLED
        assert run.judge_model is None

    def test_agent_run_debate_transcript_json(self, db_session):
        """Test AgentRun debate_transcript_json field."""
        from app.models import AgentRun

        transcript = [
            {"role": "main_agent", "content": "Initial analysis", "round": 1},
            {"role": "critique_agent", "content": "I disagree", "round": 1, "agrees": False},
        ]

        run = AgentRun(
            name="Test With Transcript",
            debate_mode=DebateMode.ENABLED,
            debate_transcript_json=transcript,
        )

        db_session.add(run)
        db_session.flush()

        # Retrieve and verify
        db_session.refresh(run)
        assert run.debate_transcript_json == transcript
        assert len(run.debate_transcript_json) == 2


# ============================================
# Integration Validation Tests
# ============================================

class TestOrchestrationValidation:
    """Tests for orchestration validation logic."""

    def test_debate_requires_gemini_and_openai(self):
        """Test that debate mode conceptually requires both clients."""
        # This is a documentation/contract test
        # Debate system uses:
        # - Gemini for critique agent
        # - OpenAI for judge
        # - Main LLM for main agent responses
        from app.services.agents.orchestration.debate_manager import DebateManager

        # Verify the init signature requires these
        import inspect
        sig = inspect.signature(DebateManager.__init__)
        params = list(sig.parameters.keys())

        assert "gemini_client" in params
        assert "openai_client" in params
        assert "main_llm_client" in params

    def test_judge_model_validation_before_run(self):
        """Test that invalid judge models are caught."""
        # Before starting a pipeline with debate enabled,
        # we should validate the judge model
        invalid_model = "not-a-real-model"
        assert validate_judge_model(invalid_model) is False

        valid_model = "gpt-4o"
        assert validate_judge_model(valid_model) is True
