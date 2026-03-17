"""Research cycles API endpoints for project-level research memory."""
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.core.security import get_current_user
from app.api.dependencies import check_project_access
from app.models import (
    Project,
    Experiment,
    User,
)
from app.models.research_cycle import (
    ResearchCycle,
    ResearchCycleStatus,
    CycleExperiment,
    LabNotebookEntry,
    LabNotebookAuthorType,
)
from app.schemas.research_cycle import (
    ResearchCycleCreate,
    ResearchCycleUpdate,
    ResearchCycleSummary,
    ResearchCycleResponse,
    ResearchCycleListResponse,
    CycleExperimentCreate,
    CycleExperimentResponse,
    LabNotebookEntryCreate,
    LabNotebookEntryUpdate,
    LabNotebookEntryResponse,
    LabNotebookEntryListResponse,
    ExperimentSummary,
    LabNotebookEntrySummary,
)

logger = logging.getLogger(__name__)

# Router for project-scoped research cycle endpoints
router = APIRouter(prefix="/projects/{project_id}/research-cycles", tags=["Research Cycles"])

# Router for top-level research cycle endpoints
cycles_router = APIRouter(tags=["Research Cycles"])


# ============================================
# Helper Functions
# ============================================

def get_project_or_404(
    project_id: UUID,
    db: Session,
    current_user: Optional[User] = None,
) -> Project:
    """Get project by ID or raise 404, and verify user access."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )
    return project


def get_cycle_or_404(cycle_id: UUID, db: Session) -> ResearchCycle:
    """Get research cycle by ID or raise 404."""
    cycle = db.query(ResearchCycle).filter(ResearchCycle.id == cycle_id).first()
    if not cycle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research cycle {cycle_id} not found",
        )
    return cycle


def create_research_cycle_for_project(
    db: Session,
    project_id: UUID,
    summary_title: Optional[str] = None,
) -> ResearchCycle:
    """Create a new research cycle with the next sequence number for this project.

    Args:
        db: Database session
        project_id: The project UUID
        summary_title: Optional title for the cycle

    Returns:
        The newly created ResearchCycle
    """
    # Get the next sequence number for this project
    max_sequence = db.query(func.max(ResearchCycle.sequence_number)).filter(
        ResearchCycle.project_id == project_id
    ).scalar()

    next_sequence = 1 if max_sequence is None else max_sequence + 1

    # Create the new cycle
    cycle = ResearchCycle(
        project_id=project_id,
        sequence_number=next_sequence,
        status=ResearchCycleStatus.PENDING,
        summary_title=summary_title,
    )
    db.add(cycle)
    db.commit()
    db.refresh(cycle)

    logger.info(f"Created research cycle {cycle.id} (sequence #{next_sequence}) for project {project_id}")

    return cycle


def link_experiment_to_cycle(
    db: Session,
    cycle_id: UUID,
    experiment_id: UUID,
) -> CycleExperiment:
    """Link an experiment to a research cycle.

    Args:
        db: Database session
        cycle_id: The research cycle UUID
        experiment_id: The experiment UUID

    Returns:
        The newly created CycleExperiment link
    """
    # Check if link already exists
    existing = db.query(CycleExperiment).filter(
        CycleExperiment.research_cycle_id == cycle_id,
        CycleExperiment.experiment_id == experiment_id,
    ).first()

    if existing:
        return existing

    link = CycleExperiment(
        research_cycle_id=cycle_id,
        experiment_id=experiment_id,
        linked_at=datetime.utcnow(),
    )
    db.add(link)
    db.commit()
    db.refresh(link)

    logger.info(f"Linked experiment {experiment_id} to research cycle {cycle_id}")

    return link


def create_lab_notebook_entry(
    db: Session,
    project_id: UUID,
    title: str,
    body_markdown: Optional[str] = None,
    research_cycle_id: Optional[UUID] = None,
    agent_step_id: Optional[UUID] = None,
    author_type: LabNotebookAuthorType = LabNotebookAuthorType.AGENT,
) -> LabNotebookEntry:
    """Create a lab notebook entry.

    Args:
        db: Database session
        project_id: The project UUID
        title: Entry title
        body_markdown: Entry content in markdown
        research_cycle_id: Optional cycle to link to
        agent_step_id: Optional agent step that created this entry
        author_type: Whether this was created by an agent or human

    Returns:
        The newly created LabNotebookEntry
    """
    entry = LabNotebookEntry(
        project_id=project_id,
        research_cycle_id=research_cycle_id,
        agent_step_id=agent_step_id,
        author_type=author_type,
        title=title,
        body_markdown=body_markdown,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    logger.info(f"Created lab notebook entry '{title}' for project {project_id}")

    return entry


# ============================================
# Project-Scoped Research Cycle Endpoints
# ============================================

@router.get("", response_model=ResearchCycleListResponse)
def list_research_cycles(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List all research cycles for a project.

    Returns cycles ordered by sequence number, with experiment counts.
    """
    project = get_project_or_404(project_id, db, current_user)

    cycles = db.query(ResearchCycle).filter(
        ResearchCycle.project_id == project_id
    ).order_by(ResearchCycle.sequence_number.desc()).all()

    # Build summaries with experiment counts
    summaries = []
    for cycle in cycles:
        exp_count = db.query(CycleExperiment).filter(
            CycleExperiment.research_cycle_id == cycle.id
        ).count()

        summaries.append(ResearchCycleSummary(
            id=cycle.id,
            sequence_number=cycle.sequence_number,
            status=cycle.status,
            summary_title=cycle.summary_title,
            created_at=cycle.created_at,
            updated_at=cycle.updated_at,
            experiment_count=exp_count,
        ))

    return ResearchCycleListResponse(
        cycles=summaries,
        total=len(summaries),
    )


@router.post("", response_model=ResearchCycleSummary, status_code=status.HTTP_201_CREATED)
def create_cycle(
    project_id: UUID,
    data: ResearchCycleCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new research cycle for a project.

    Automatically assigns the next sequence number.
    """
    project = get_project_or_404(project_id, db, current_user)

    cycle = create_research_cycle_for_project(
        db=db,
        project_id=project_id,
        summary_title=data.summary_title,
    )

    return ResearchCycleSummary(
        id=cycle.id,
        sequence_number=cycle.sequence_number,
        status=cycle.status,
        summary_title=cycle.summary_title,
        created_at=cycle.created_at,
        updated_at=cycle.updated_at,
        experiment_count=0,
    )


# ============================================
# Top-Level Research Cycle Endpoints
# ============================================

@cycles_router.get("/research-cycles/{cycle_id}", response_model=ResearchCycleResponse)
def get_research_cycle(
    cycle_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a research cycle with all its linked experiments and notebook entries."""
    cycle = get_cycle_or_404(cycle_id, db)

    # Verify user has access to the cycle's project
    project = db.query(Project).filter(Project.id == cycle.project_id).first()
    if project and not check_project_access(db, project, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    # Get linked experiments with their details
    cycle_exps = db.query(CycleExperiment).filter(
        CycleExperiment.research_cycle_id == cycle_id
    ).all()

    experiments = []
    for ce in cycle_exps:
        exp = db.query(Experiment).filter(Experiment.id == ce.experiment_id).first()
        if exp:
            # Get best metric from trials
            best_metric = None
            if exp.trials:
                for trial in exp.trials:
                    if trial.metrics_json and exp.primary_metric:
                        metric_val = trial.metrics_json.get(exp.primary_metric)
                        if metric_val is not None:
                            if best_metric is None:
                                best_metric = metric_val
                            else:
                                # Assume maximize for now
                                best_metric = max(best_metric, metric_val)

            experiments.append(ExperimentSummary(
                id=exp.id,
                name=exp.name,
                status=exp.status.value if exp.status else "unknown",
                best_metric=best_metric,
                primary_metric=exp.primary_metric,
                created_at=exp.created_at,
            ))

    # Get notebook entries for this cycle
    entries = db.query(LabNotebookEntry).filter(
        LabNotebookEntry.research_cycle_id == cycle_id
    ).order_by(LabNotebookEntry.created_at.desc()).all()

    entry_summaries = [
        LabNotebookEntrySummary(
            id=e.id,
            title=e.title,
            author_type=e.author_type,
            created_at=e.created_at,
        )
        for e in entries
    ]

    return ResearchCycleResponse(
        id=cycle.id,
        project_id=cycle.project_id,
        sequence_number=cycle.sequence_number,
        status=cycle.status,
        summary_title=cycle.summary_title,
        created_at=cycle.created_at,
        updated_at=cycle.updated_at,
        experiments=experiments,
        lab_notebook_entries=entry_summaries,
    )


@cycles_router.patch("/research-cycles/{cycle_id}", response_model=ResearchCycleSummary)
def update_research_cycle(
    cycle_id: UUID,
    data: ResearchCycleUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update a research cycle (title, status)."""
    cycle = get_cycle_or_404(cycle_id, db)

    # Verify user has write access to the cycle's project
    project = db.query(Project).filter(Project.id == cycle.project_id).first()
    if project and not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    if data.summary_title is not None:
        cycle.summary_title = data.summary_title
    if data.status is not None:
        cycle.status = data.status

    db.commit()
    db.refresh(cycle)

    exp_count = db.query(CycleExperiment).filter(
        CycleExperiment.research_cycle_id == cycle.id
    ).count()

    return ResearchCycleSummary(
        id=cycle.id,
        sequence_number=cycle.sequence_number,
        status=cycle.status,
        summary_title=cycle.summary_title,
        created_at=cycle.created_at,
        updated_at=cycle.updated_at,
        experiment_count=exp_count,
    )


@cycles_router.post("/research-cycles/{cycle_id}/experiments", response_model=CycleExperimentResponse, status_code=status.HTTP_201_CREATED)
def link_experiment(
    cycle_id: UUID,
    data: CycleExperimentCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Link an experiment to a research cycle."""
    cycle = get_cycle_or_404(cycle_id, db)

    # Verify user has write access to the cycle's project
    project = db.query(Project).filter(Project.id == cycle.project_id).first()
    if project and not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    # Verify experiment exists and belongs to same project
    experiment = db.query(Experiment).filter(Experiment.id == data.experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {data.experiment_id} not found",
        )

    if experiment.project_id != cycle.project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment must belong to the same project as the research cycle",
        )

    link = link_experiment_to_cycle(db, cycle_id, data.experiment_id)

    return CycleExperimentResponse(
        id=link.id,
        research_cycle_id=link.research_cycle_id,
        experiment_id=link.experiment_id,
        linked_at=link.linked_at,
    )


# ============================================
# Lab Notebook Endpoints
# ============================================

@router.get("/notebook", response_model=LabNotebookEntryListResponse)
def list_notebook_entries(
    project_id: UUID,
    cycle_id: Optional[UUID] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List lab notebook entries for a project.

    Optionally filter by research cycle ID.
    """
    project = get_project_or_404(project_id, db, current_user)

    query = db.query(LabNotebookEntry).filter(
        LabNotebookEntry.project_id == project_id
    )

    if cycle_id:
        query = query.filter(LabNotebookEntry.research_cycle_id == cycle_id)

    entries = query.order_by(LabNotebookEntry.created_at.desc()).all()

    return LabNotebookEntryListResponse(
        entries=[
            LabNotebookEntryResponse(
                id=e.id,
                project_id=e.project_id,
                research_cycle_id=e.research_cycle_id,
                agent_step_id=e.agent_step_id,
                author_type=e.author_type,
                title=e.title,
                body_markdown=e.body_markdown,
                created_at=e.created_at,
                updated_at=e.updated_at,
            )
            for e in entries
        ],
        total=len(entries),
    )


@router.post("/notebook", response_model=LabNotebookEntryResponse, status_code=status.HTTP_201_CREATED)
def create_notebook_entry(
    project_id: UUID,
    data: LabNotebookEntryCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new lab notebook entry."""
    project = get_project_or_404(project_id, db, current_user)

    # Verify cycle belongs to project if specified
    if data.research_cycle_id:
        cycle = db.query(ResearchCycle).filter(
            ResearchCycle.id == data.research_cycle_id,
            ResearchCycle.project_id == project_id,
        ).first()
        if not cycle:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research cycle {data.research_cycle_id} not found in this project",
            )

    entry = create_lab_notebook_entry(
        db=db,
        project_id=project_id,
        title=data.title,
        body_markdown=data.body_markdown,
        research_cycle_id=data.research_cycle_id,
        agent_step_id=data.agent_step_id,
        author_type=data.author_type,
    )

    return LabNotebookEntryResponse(
        id=entry.id,
        project_id=entry.project_id,
        research_cycle_id=entry.research_cycle_id,
        agent_step_id=entry.agent_step_id,
        author_type=entry.author_type,
        title=entry.title,
        body_markdown=entry.body_markdown,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@cycles_router.get("/notebook/{entry_id}", response_model=LabNotebookEntryResponse)
def get_notebook_entry(
    entry_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a single lab notebook entry by ID."""
    entry = db.query(LabNotebookEntry).filter(LabNotebookEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lab notebook entry {entry_id} not found",
        )

    # Verify user has access to the entry's project
    project = db.query(Project).filter(Project.id == entry.project_id).first()
    if project and not check_project_access(db, project, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    return LabNotebookEntryResponse(
        id=entry.id,
        project_id=entry.project_id,
        research_cycle_id=entry.research_cycle_id,
        agent_step_id=entry.agent_step_id,
        author_type=entry.author_type,
        title=entry.title,
        body_markdown=entry.body_markdown,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@cycles_router.patch("/notebook/{entry_id}", response_model=LabNotebookEntryResponse)
def update_notebook_entry(
    entry_id: UUID,
    data: LabNotebookEntryUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update a lab notebook entry."""
    entry = db.query(LabNotebookEntry).filter(LabNotebookEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lab notebook entry {entry_id} not found",
        )

    # Verify user has write access to the entry's project
    project = db.query(Project).filter(Project.id == entry.project_id).first()
    if project and not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    if data.title is not None:
        entry.title = data.title
    if data.body_markdown is not None:
        entry.body_markdown = data.body_markdown

    db.commit()
    db.refresh(entry)

    return LabNotebookEntryResponse(
        id=entry.id,
        project_id=entry.project_id,
        research_cycle_id=entry.research_cycle_id,
        agent_step_id=entry.agent_step_id,
        author_type=entry.author_type,
        title=entry.title,
        body_markdown=entry.body_markdown,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@cycles_router.delete("/notebook/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_notebook_entry(
    entry_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Delete a lab notebook entry."""
    entry = db.query(LabNotebookEntry).filter(LabNotebookEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lab notebook entry {entry_id} not found",
        )

    # Verify user has write access to the entry's project
    project = db.query(Project).filter(Project.id == entry.project_id).first()
    if project and not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    db.delete(entry)
    db.commit()

    return None
