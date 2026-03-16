"""Tests for research cycles and lab notebook API endpoints."""
import pytest


class TestResearchCyclesCRUD:
    """Test research cycle CRUD operations."""

    def _create_project(self, client) -> str:
        """Helper to create a test project."""
        response = client.post(
            "/projects",
            json={"name": "Research Test Project", "task_type": "classification"},
        )
        assert response.status_code == 201
        return response.json()["id"]

    def test_create_research_cycle(self, client):
        """Test creating a new research cycle."""
        project_id = self._create_project(client)

        response = client.post(
            f"/projects/{project_id}/research-cycles",
            json={"summary_title": "Initial Exploration"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["sequence_number"] == 1
        assert data["status"] == "pending"
        assert data["summary_title"] == "Initial Exploration"
        assert data["experiment_count"] == 0
        assert "id" in data
        assert "created_at" in data

    def test_create_research_cycle_auto_sequence(self, client):
        """Test that sequence numbers auto-increment."""
        project_id = self._create_project(client)

        # Create first cycle
        response1 = client.post(
            f"/projects/{project_id}/research-cycles",
            json={"summary_title": "Cycle 1"},
        )
        assert response1.status_code == 201
        assert response1.json()["sequence_number"] == 1

        # Create second cycle
        response2 = client.post(
            f"/projects/{project_id}/research-cycles",
            json={"summary_title": "Cycle 2"},
        )
        assert response2.status_code == 201
        assert response2.json()["sequence_number"] == 2

        # Create third cycle
        response3 = client.post(
            f"/projects/{project_id}/research-cycles",
            json={},
        )
        assert response3.status_code == 201
        assert response3.json()["sequence_number"] == 3

    def test_list_research_cycles_empty(self, client):
        """Test listing research cycles when none exist."""
        project_id = self._create_project(client)

        response = client.get(f"/projects/{project_id}/research-cycles")
        assert response.status_code == 200
        data = response.json()
        assert data["cycles"] == []
        assert data["total"] == 0

    def test_list_research_cycles(self, client):
        """Test listing research cycles."""
        project_id = self._create_project(client)

        # Create two cycles
        client.post(
            f"/projects/{project_id}/research-cycles",
            json={"summary_title": "First Cycle"},
        )
        client.post(
            f"/projects/{project_id}/research-cycles",
            json={"summary_title": "Second Cycle"},
        )

        response = client.get(f"/projects/{project_id}/research-cycles")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["cycles"]) == 2
        # Should be ordered by sequence_number desc (newest first)
        assert data["cycles"][0]["sequence_number"] == 2
        assert data["cycles"][1]["sequence_number"] == 1

    def test_get_research_cycle(self, client):
        """Test getting a research cycle by ID."""
        project_id = self._create_project(client)

        # Create a cycle
        create_response = client.post(
            f"/projects/{project_id}/research-cycles",
            json={"summary_title": "Get Test Cycle"},
        )
        cycle_id = create_response.json()["id"]

        # Get the cycle
        response = client.get(f"/research-cycles/{cycle_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == cycle_id
        assert data["project_id"] == project_id
        assert data["summary_title"] == "Get Test Cycle"
        assert data["experiments"] == []
        assert data["lab_notebook_entries"] == []

    def test_get_research_cycle_not_found(self, client):
        """Test getting a non-existent research cycle."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/research-cycles/{fake_id}")
        assert response.status_code == 404

    def test_update_research_cycle(self, client):
        """Test updating a research cycle."""
        project_id = self._create_project(client)

        # Create a cycle
        create_response = client.post(
            f"/projects/{project_id}/research-cycles",
            json={"summary_title": "Original Title"},
        )
        cycle_id = create_response.json()["id"]

        # Update the cycle
        response = client.patch(
            f"/research-cycles/{cycle_id}",
            json={
                "summary_title": "Updated Title",
                "status": "running",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary_title"] == "Updated Title"
        assert data["status"] == "running"


class TestLabNotebookEntries:
    """Test lab notebook entry CRUD operations."""

    def _create_project(self, client) -> str:
        """Helper to create a test project."""
        response = client.post(
            "/projects",
            json={"name": "Notebook Test Project", "task_type": "regression"},
        )
        assert response.status_code == 201
        return response.json()["id"]

    def _create_cycle(self, client, project_id: str) -> str:
        """Helper to create a test research cycle."""
        response = client.post(
            f"/projects/{project_id}/research-cycles",
            json={"summary_title": "Test Cycle"},
        )
        assert response.status_code == 201
        return response.json()["id"]

    def test_create_notebook_entry(self, client):
        """Test creating a new lab notebook entry."""
        project_id = self._create_project(client)

        response = client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={
                "title": "Initial Hypothesis",
                "body_markdown": "The target variable appears to be correlated with feature X.",
                "author_type": "human",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Initial Hypothesis"
        assert data["body_markdown"] == "The target variable appears to be correlated with feature X."
        assert data["author_type"] == "human"
        assert data["project_id"] == project_id
        assert data["research_cycle_id"] is None
        assert "id" in data
        assert "created_at" in data

    def test_create_notebook_entry_with_cycle(self, client):
        """Test creating a notebook entry linked to a cycle."""
        project_id = self._create_project(client)
        cycle_id = self._create_cycle(client, project_id)

        response = client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={
                "title": "Cycle Observation",
                "body_markdown": "Observed pattern in data.",
                "research_cycle_id": cycle_id,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["research_cycle_id"] == cycle_id

    def test_create_notebook_entry_agent_author(self, client):
        """Test creating a notebook entry with agent author."""
        project_id = self._create_project(client)

        response = client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={
                "title": "Agent Analysis",
                "body_markdown": "Automated analysis results.",
                "author_type": "agent",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["author_type"] == "agent"

    def test_list_notebook_entries_empty(self, client):
        """Test listing notebook entries when none exist."""
        project_id = self._create_project(client)

        response = client.get(f"/projects/{project_id}/research-cycles/notebook")
        assert response.status_code == 200
        data = response.json()
        assert data["entries"] == []
        assert data["total"] == 0

    def test_list_notebook_entries(self, client):
        """Test listing notebook entries."""
        project_id = self._create_project(client)

        # Create two entries
        client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={"title": "Entry 1", "author_type": "human"},
        )
        client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={"title": "Entry 2", "author_type": "agent"},
        )

        response = client.get(f"/projects/{project_id}/research-cycles/notebook")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["entries"]) == 2

    def test_list_notebook_entries_filter_by_cycle(self, client):
        """Test filtering notebook entries by cycle."""
        project_id = self._create_project(client)
        cycle_id = self._create_cycle(client, project_id)

        # Create entry without cycle
        client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={"title": "General Entry", "author_type": "human"},
        )

        # Create entry with cycle
        client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={
                "title": "Cycle Entry",
                "research_cycle_id": cycle_id,
                "author_type": "human",
            },
        )

        # List all entries
        response_all = client.get(f"/projects/{project_id}/research-cycles/notebook")
        assert response_all.json()["total"] == 2

        # Filter by cycle
        response_filtered = client.get(
            f"/projects/{project_id}/research-cycles/notebook?cycle_id={cycle_id}"
        )
        data = response_filtered.json()
        assert data["total"] == 1
        assert data["entries"][0]["title"] == "Cycle Entry"

    def test_get_notebook_entry(self, client):
        """Test getting a notebook entry by ID."""
        project_id = self._create_project(client)

        # Create an entry
        create_response = client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={
                "title": "Get Test Entry",
                "body_markdown": "Test content",
                "author_type": "human",
            },
        )
        entry_id = create_response.json()["id"]

        # Get the entry
        response = client.get(f"/notebook/{entry_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == entry_id
        assert data["title"] == "Get Test Entry"
        assert data["body_markdown"] == "Test content"

    def test_get_notebook_entry_not_found(self, client):
        """Test getting a non-existent notebook entry."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/notebook/{fake_id}")
        assert response.status_code == 404

    def test_update_notebook_entry(self, client):
        """Test updating a notebook entry."""
        project_id = self._create_project(client)

        # Create an entry
        create_response = client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={
                "title": "Original Title",
                "body_markdown": "Original content",
                "author_type": "human",
            },
        )
        entry_id = create_response.json()["id"]

        # Update the entry
        response = client.patch(
            f"/notebook/{entry_id}",
            json={
                "title": "Updated Title",
                "body_markdown": "Updated content with new findings.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["body_markdown"] == "Updated content with new findings."

    def test_delete_notebook_entry(self, client):
        """Test deleting a notebook entry."""
        project_id = self._create_project(client)

        # Create an entry
        create_response = client.post(
            f"/projects/{project_id}/research-cycles/notebook",
            json={"title": "To Delete", "author_type": "human"},
        )
        entry_id = create_response.json()["id"]

        # Delete the entry
        response = client.delete(f"/notebook/{entry_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_response = client.get(f"/notebook/{entry_id}")
        assert get_response.status_code == 404


class TestCycleExperimentLinks:
    """Test linking experiments to research cycles."""

    def _create_project(self, client) -> str:
        """Helper to create a test project."""
        response = client.post(
            "/projects",
            json={"name": "Link Test Project", "task_type": "binary"},
        )
        assert response.status_code == 201
        return response.json()["id"]

    def _create_cycle(self, client, project_id: str) -> str:
        """Helper to create a test research cycle."""
        response = client.post(
            f"/projects/{project_id}/research-cycles",
            json={"summary_title": "Test Cycle"},
        )
        assert response.status_code == 201
        return response.json()["id"]

    def _create_experiment(self, client, project_id: str) -> str:
        """Helper to create a test experiment."""
        response = client.post(
            f"/projects/{project_id}/experiments",
            json={"name": "Test Experiment"},
        )
        assert response.status_code == 201
        return response.json()["id"]

    def test_link_experiment_to_cycle(self, client):
        """Test linking an experiment to a research cycle."""
        project_id = self._create_project(client)
        cycle_id = self._create_cycle(client, project_id)
        experiment_id = self._create_experiment(client, project_id)

        response = client.post(
            f"/research-cycles/{cycle_id}/experiments",
            json={"experiment_id": experiment_id},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["research_cycle_id"] == cycle_id
        assert data["experiment_id"] == experiment_id
        assert "linked_at" in data

    def test_link_experiment_idempotent(self, client):
        """Test that linking the same experiment twice is idempotent."""
        project_id = self._create_project(client)
        cycle_id = self._create_cycle(client, project_id)
        experiment_id = self._create_experiment(client, project_id)

        # Link first time
        response1 = client.post(
            f"/research-cycles/{cycle_id}/experiments",
            json={"experiment_id": experiment_id},
        )
        assert response1.status_code == 201

        # Link second time should succeed (return existing link)
        response2 = client.post(
            f"/research-cycles/{cycle_id}/experiments",
            json={"experiment_id": experiment_id},
        )
        assert response2.status_code == 201
        assert response2.json()["id"] == response1.json()["id"]

    def test_link_experiment_wrong_project(self, client):
        """Test that linking an experiment from a different project fails."""
        project_id_1 = self._create_project(client)
        project_id_2 = self._create_project(client)

        cycle_id = self._create_cycle(client, project_id_1)
        experiment_id = self._create_experiment(client, project_id_2)

        response = client.post(
            f"/research-cycles/{cycle_id}/experiments",
            json={"experiment_id": experiment_id},
        )
        assert response.status_code == 400
        assert "same project" in response.json()["detail"].lower()

    def test_cycle_includes_linked_experiments(self, client):
        """Test that getting a cycle includes linked experiments."""
        project_id = self._create_project(client)
        cycle_id = self._create_cycle(client, project_id)
        experiment_id = self._create_experiment(client, project_id)

        # Link the experiment
        client.post(
            f"/research-cycles/{cycle_id}/experiments",
            json={"experiment_id": experiment_id},
        )

        # Get the cycle
        response = client.get(f"/research-cycles/{cycle_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["experiments"]) == 1
        assert data["experiments"][0]["id"] == experiment_id

    def test_cycle_experiment_count_updates(self, client):
        """Test that cycle list shows correct experiment count."""
        project_id = self._create_project(client)
        cycle_id = self._create_cycle(client, project_id)

        # Check initial count
        list_response = client.get(f"/projects/{project_id}/research-cycles")
        assert list_response.json()["cycles"][0]["experiment_count"] == 0

        # Link an experiment
        experiment_id = self._create_experiment(client, project_id)
        client.post(
            f"/research-cycles/{cycle_id}/experiments",
            json={"experiment_id": experiment_id},
        )

        # Check updated count
        list_response = client.get(f"/projects/{project_id}/research-cycles")
        assert list_response.json()["cycles"][0]["experiment_count"] == 1
