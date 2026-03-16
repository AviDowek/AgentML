"""Tests for project API endpoints."""
import pytest


class TestProjectsCRUD:
    """Test project CRUD operations."""

    def test_create_project(self, client):
        """Test creating a new project."""
        response = client.post(
            "/projects",
            json={
                "name": "Test Project",
                "description": "A test ML project",
                "task_type": "classification",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["description"] == "A test ML project"
        assert data["task_type"] == "classification"
        assert data["status"] == "draft"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_project_minimal(self, client):
        """Test creating a project with only required fields."""
        response = client.post("/projects", json={"name": "Minimal Project"})
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal Project"
        assert data["description"] is None
        assert data["task_type"] is None

    def test_list_projects_empty(self, client):
        """Test listing projects when none exist."""
        response = client.get("/projects")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_projects(self, client):
        """Test listing projects."""
        # Create two projects
        client.post("/projects", json={"name": "Project 1"})
        client.post("/projects", json={"name": "Project 2"})

        response = client.get("/projects")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2

    def test_get_project(self, client):
        """Test getting a project by ID."""
        # Create a project
        create_response = client.post(
            "/projects",
            json={"name": "Get Test", "description": "Test get endpoint"},
        )
        project_id = create_response.json()["id"]

        # Get the project
        response = client.get(f"/projects/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == project_id
        assert data["name"] == "Get Test"

    def test_get_project_not_found(self, client):
        """Test getting a non-existent project."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/projects/{fake_id}")
        assert response.status_code == 404

    def test_update_project(self, client):
        """Test updating a project."""
        # Create a project
        create_response = client.post("/projects", json={"name": "Original Name"})
        project_id = create_response.json()["id"]

        # Update the project
        response = client.put(
            f"/projects/{project_id}",
            json={
                "name": "Updated Name",
                "description": "New description",
                "status": "active",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "New description"
        assert data["status"] == "active"

    def test_delete_project(self, client):
        """Test deleting a project."""
        # Create a project
        create_response = client.post("/projects", json={"name": "To Delete"})
        project_id = create_response.json()["id"]

        # Delete the project
        response = client.delete(f"/projects/{project_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_response = client.get(f"/projects/{project_id}")
        assert get_response.status_code == 404
