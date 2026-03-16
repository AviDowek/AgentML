"""Research cycle and lab notebook schemas."""
from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.research_cycle import ResearchCycleStatus, LabNotebookAuthorType


# ============================================
# Research Cycle Schemas
# ============================================

class ResearchCycleBase(BaseModel):
    """Base research cycle schema."""
    summary_title: Optional[str] = Field(None, max_length=500)


class ResearchCycleCreate(ResearchCycleBase):
    """Schema for creating a research cycle."""
    pass


class ResearchCycleUpdate(BaseModel):
    """Schema for updating a research cycle."""
    summary_title: Optional[str] = Field(None, max_length=500)
    status: Optional[ResearchCycleStatus] = None


class ExperimentSummary(BaseModel):
    """Summary of an experiment for cycle responses."""
    id: UUID
    name: str
    status: str
    best_metric: Optional[float] = None
    primary_metric: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class LabNotebookEntrySummary(BaseModel):
    """Summary of a lab notebook entry for cycle responses."""
    id: UUID
    title: str
    author_type: LabNotebookAuthorType
    created_at: datetime

    class Config:
        from_attributes = True


class ResearchCycleSummary(BaseModel):
    """Summary schema for listing research cycles."""
    id: UUID
    sequence_number: int
    status: ResearchCycleStatus
    summary_title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    experiment_count: int = 0

    class Config:
        from_attributes = True


class ResearchCycleResponse(BaseModel):
    """Full research cycle response with linked data."""
    id: UUID
    project_id: UUID
    sequence_number: int
    status: ResearchCycleStatus
    summary_title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    experiments: List[ExperimentSummary] = Field(default_factory=list)
    lab_notebook_entries: List[LabNotebookEntrySummary] = Field(default_factory=list)

    class Config:
        from_attributes = True


class ResearchCycleListResponse(BaseModel):
    """Response for listing research cycles."""
    cycles: List[ResearchCycleSummary]
    total: int


# ============================================
# Cycle Experiment Link Schemas
# ============================================

class CycleExperimentCreate(BaseModel):
    """Schema for linking an experiment to a cycle."""
    experiment_id: UUID


class CycleExperimentResponse(BaseModel):
    """Response for a cycle-experiment link."""
    id: UUID
    research_cycle_id: UUID
    experiment_id: UUID
    linked_at: datetime

    class Config:
        from_attributes = True


# ============================================
# Lab Notebook Entry Schemas
# ============================================

class LabNotebookEntryBase(BaseModel):
    """Base lab notebook entry schema."""
    title: str = Field(..., min_length=1, max_length=500)
    body_markdown: Optional[str] = None


class LabNotebookEntryCreate(LabNotebookEntryBase):
    """Schema for creating a lab notebook entry."""
    research_cycle_id: Optional[UUID] = None
    agent_step_id: Optional[UUID] = None
    author_type: LabNotebookAuthorType = LabNotebookAuthorType.HUMAN


class LabNotebookEntryUpdate(BaseModel):
    """Schema for updating a lab notebook entry."""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    body_markdown: Optional[str] = None


class LabNotebookEntryResponse(BaseModel):
    """Full lab notebook entry response."""
    id: UUID
    project_id: UUID
    research_cycle_id: Optional[UUID] = None
    agent_step_id: Optional[UUID] = None
    author_type: LabNotebookAuthorType
    title: str
    body_markdown: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LabNotebookEntryListResponse(BaseModel):
    """Response for listing lab notebook entries."""
    entries: List[LabNotebookEntryResponse]
    total: int
