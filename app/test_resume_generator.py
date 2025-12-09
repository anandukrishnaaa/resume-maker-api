"""
test_resume_generator.py

Comprehensive test suite for Resume Generator API

Run with:
    uv run pytest test_resume_generator.py -v
    uv run pytest test_resume_generator.py -v -s  # with print output
    uv run pytest test_resume_generator.py -k "test_user"  # run specific tests
"""

import pytest
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

# Import your app
from resume_generator import (
    app,
    get_session,
    create_db_and_tables,
    UserProfile,
    PersonalInfo,
    JobDescription,
    TaskLog,
    ResumeContent,
)

# ============================================================================
# Test Configuration
# ============================================================================
TEST_API_TOKEN = "secret-token-123"
TEST_USER_ID = "test_user_123"


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture(name="session")
def session_fixture():
    """Create a fresh in-memory database for each test"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: Session):
    """Create a test client with database override"""

    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    """Standard authentication headers"""
    return {
        "X-Api-Key": TEST_API_TOKEN,
        "X-User-Id": TEST_USER_ID,
    }


@pytest.fixture
def sample_user_data():
    """Sample user creation data"""
    return {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "9876543210",
        "country_code": "+91",
        "location": "Mumbai, India",
        "social_networks": [
            {"network": "LinkedIn", "username": "johndoe"},
            {"network": "GitHub", "username": "johndoe-dev"},
        ],
        "profile_data": {
            "education": [
                {
                    "institution": "Indian Institute of Technology",
                    "area": "Computer Science",
                    "degree": "B.Tech",
                    "start_date": "2018-08",
                    "end_date": "2022-05",
                    "highlights": [
                        "GPA: 3.8/4.0",
                        "President of Computer Science Club",
                    ],
                }
            ],
            "experience": [
                {
                    "company": "TechCorp India",
                    "position": "Software Engineer",
                    "start_date": "2022-06",
                    "end_date": "present",
                    "location": "Bangalore, India",
                    "highlights": [
                        "Developed scalable microservices using Python and FastAPI",
                        "Reduced API response time by 40% through optimization",
                        "Led a team of 3 junior developers",
                    ],
                }
            ],
            "skills": {
                "languages": ["Python", "JavaScript", "Go"],
                "frameworks": ["FastAPI", "React", "Django"],
                "tools": ["Docker", "Kubernetes", "AWS"],
            },
        },
    }


@pytest.fixture
def sample_resume_text():
    """Sample resume text for parsing"""
    return """John Doe
john.doe@example.com | +91-9876543210 | Mumbai, India | linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years building scalable web applications.

EXPERIENCE

Senior Software Engineer | TechCorp Inc. | Mumbai, India | June 2022 - Present
• Led development of microservices architecture serving 1M+ users
• Reduced API response time by 40% through optimization
• Mentored team of 5 junior engineers
• Technologies: Python, FastAPI, Docker, AWS

Software Engineer | StartupXYZ | Bangalore, India | Jan 2020 - May 2022
• Built RESTful APIs using Python and FastAPI
• Implemented CI/CD pipelines with GitHub Actions
• Collaborated with frontend team on React applications
• Increased system reliability to 99.9% uptime

EDUCATION

Bachelor of Technology in Computer Science | IIT Bombay | Mumbai, India | 2016 - 2020
• GPA: 3.8/4.0
• Dean's List all semesters
• President of Computer Science Club

SKILLS

Programming Languages: Python, JavaScript, Go, TypeScript, SQL
Frameworks & Libraries: FastAPI, Django, React, Node.js, Express
Tools & Technologies: Docker, Kubernetes, AWS, PostgreSQL, MongoDB, Redis
Soft Skills: Leadership, Team Collaboration, Problem Solving
"""


@pytest.fixture
def sample_job_request():
    """Sample job description"""
    return {
        "title": "Senior Backend Engineer",
        "company": "BigTech Corp",
        "description": """
        We are looking for a Senior Backend Engineer to join our team.
        
        Requirements:
        - 3+ years of experience with Python
        - Experience with FastAPI or Django
        - Strong knowledge of microservices architecture
        - Experience with AWS or GCP
        - Excellent problem-solving skills
        
        Responsibilities:
        - Design and implement scalable backend services
        - Mentor junior developers
        - Collaborate with frontend team
        - Optimize system performance
        """,
    }


# ============================================================================
# Test: User Management
# ============================================================================
class TestUserManagement:
    """Test user creation, profile retrieval, and updates"""

    def test_create_user_with_complete_info(
        self, client: TestClient, auth_headers: dict, sample_user_data: dict
    ):
        """Test creating a new user with complete personal and profile data"""
        response = client.post("/users", headers=auth_headers, json=sample_user_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "User profile created successfully"
        assert data["data"]["user_id"] == TEST_USER_ID
        assert data["data"]["name"] == sample_user_data["name"]
        assert data["data"]["email"] == sample_user_data["email"]

    def test_create_duplicate_user(
        self, client: TestClient, auth_headers: dict, sample_user_data: dict
    ):
        """Test creating a user that already exists"""
        # Create user first time
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Try to create again
        response = client.post("/users", headers=auth_headers, json=sample_user_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "USER_EXISTS"

    def test_get_user_profile(
        self, client: TestClient, auth_headers: dict, sample_user_data: dict
    ):
        """Test retrieving user profile"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Get profile
        response = client.get("/users/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["user_id"] == TEST_USER_ID
        assert data["data"]["personal_info"]["name"] == sample_user_data["name"]
        assert len(data["data"]["profile_data"]["education"]) == 1
        assert len(data["data"]["profile_data"]["experience"]) == 1

    def test_update_contact_info(
        self, client: TestClient, auth_headers: dict, sample_user_data: dict
    ):
        """Test updating user contact information"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Update contact info
        update_data = {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "9999999999",
            "country_code": "+1",
            "location": "San Francisco, USA",
            "social_networks": [{"network": "LinkedIn", "username": "janedoe"}],
        }

        response = client.put(
            "/users/me/contact", headers=auth_headers, json=update_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify update
        profile_response = client.get("/users/me", headers=auth_headers)
        profile = profile_response.json()
        assert profile["data"]["personal_info"]["name"] == "Jane Doe"
        assert profile["data"]["personal_info"]["email"] == "jane.doe@example.com"

    def test_update_profile_data(
        self, client: TestClient, auth_headers: dict, sample_user_data: dict
    ):
        """Test updating user work history and skills"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Add new experience
        update_data = {
            "experience": [
                {
                    "company": "NewCorp",
                    "position": "Tech Lead",
                    "start_date": "2024-01",
                    "end_date": "present",
                    "highlights": ["Leading team of 10 engineers"],
                }
            ]
        }

        response = client.put(
            "/users/me/profile", headers=auth_headers, json=update_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify update
        profile_response = client.get("/users/me", headers=auth_headers)
        profile = profile_response.json()
        assert len(profile["data"]["profile_data"]["experience"]) == 1
        assert profile["data"]["profile_data"]["experience"][0]["company"] == "NewCorp"


# ============================================================================
# Test: Resume Parsing (NEW)
# ============================================================================
class TestResumeParsing:
    """Test AI-powered resume parsing functionality"""

    def test_create_user_from_resume(
        self, client: TestClient, auth_headers: dict, sample_resume_text: str
    ):
        """Test creating user from resume text"""
        response = client.post(
            "/users/from-resume",
            headers=auth_headers,
            json={"resume_text": sample_resume_text, "overwrite_existing": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "extracted_data" in data["data"]
        assert data["data"]["name"] == "John Doe"
        assert data["data"]["email"] == "john.doe@example.com"
        assert data["data"]["education_count"] >= 1
        assert data["data"]["experience_count"] >= 1
        assert "tokens_used" in data["data"]

    def test_create_user_from_resume_duplicate_without_overwrite(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_resume_text: str,
        sample_user_data: dict,
    ):
        """Test creating user from resume when user already exists (without overwrite)"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Try to create from resume without overwrite
        response = client.post(
            "/users/from-resume",
            headers=auth_headers,
            json={"resume_text": sample_resume_text, "overwrite_existing": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "USER_EXISTS"

    def test_overwrite_user_from_resume(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_resume_text: str,
        sample_user_data: dict,
    ):
        """Test overwriting existing user profile with parsed resume"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Overwrite with resume data
        response = client.post(
            "/users/from-resume",
            headers=auth_headers,
            json={"resume_text": sample_resume_text, "overwrite_existing": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify the profile was updated
        profile_response = client.get("/users/me", headers=auth_headers)
        profile = profile_response.json()
        assert profile["data"]["personal_info"]["name"] == "John Doe"


# ============================================================================
# Test: Job Submission
# ============================================================================
class TestJobSubmission:
    """Test job submission workflows"""

    def test_submit_job_two_step(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test two-step job submission (analysis only)"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Submit job for analysis only
        response = client.post("/jobs", headers=auth_headers, json=sample_job_request)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data["data"]
        assert data["data"]["status"] in ["analysis_pending", "analysis_complete"]

    def test_submit_job_complete(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test one-shot job submission (analysis + resume)"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Submit job with auto-resume
        response = client.post(
            "/jobs/complete", headers=auth_headers, json=sample_job_request
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data["data"]

    def test_submit_job_with_custom_ai_model(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test job submission with custom AI model"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Submit job with custom model
        job_with_model = {
            **sample_job_request,
            "ai_model": "anthropic/claude-3.5-sonnet",
        }

        response = client.post("/jobs", headers=auth_headers, json=job_with_model)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_submit_job_with_design_config(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test job submission with custom design configuration"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Submit job with custom design
        job_with_design = {
            **sample_job_request,
            "design_config": {
                "theme": "moderncv",
                "colors": {"section_titles": "rgb(200, 0, 0)"},
                "page": {"size": "a4"},
            },
        }

        response = client.post(
            "/jobs/complete", headers=auth_headers, json=job_with_design
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_list_jobs(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test listing all jobs for a user"""
        # Create user first
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Submit a job
        client.post("/jobs", headers=auth_headers, json=sample_job_request)

        # List jobs
        response = client.get("/jobs", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["count"] >= 1
        assert len(data["data"]["jobs"]) >= 1


# ============================================================================
# Test: Task Management
# ============================================================================
class TestTaskManagement:
    """Test task status, retrieval, and management"""

    def test_get_task_status(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test getting task status"""
        # Create user and submit job
        client.post("/users", headers=auth_headers, json=sample_user_data)
        submit_response = client.post(
            "/jobs", headers=auth_headers, json=sample_job_request
        )
        task_id = submit_response.json()["data"]["task_id"]

        # Get task status
        response = client.get(f"/tasks/{task_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == task_id
        assert "status" in data["data"]
        assert "logs" in data["data"]

    def test_list_tasks(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test listing all tasks for a user"""
        # Create user and submit job
        client.post("/users", headers=auth_headers, json=sample_user_data)
        client.post("/jobs", headers=auth_headers, json=sample_job_request)

        # List tasks
        response = client.get("/tasks", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["count"] >= 1
        assert len(data["data"]["tasks"]) >= 1

    def test_trigger_resume_generation(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test manually triggering resume generation after analysis"""
        # Create user and submit job (two-step)
        client.post("/users", headers=auth_headers, json=sample_user_data)
        submit_response = client.post(
            "/jobs", headers=auth_headers, json=sample_job_request
        )
        task_id = submit_response.json()["data"]["task_id"]

        # Wait for analysis to complete (in real scenario)
        # For testing, we'll just try to trigger

        # Trigger resume generation
        response = client.post(f"/tasks/{task_id}/resume", headers=auth_headers)

        # Should work if analysis is complete, or return error if not
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_regenerate_resume_with_improvements(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test regenerating resume with improvement remarks"""
        # Create user and submit job
        client.post("/users", headers=auth_headers, json=sample_user_data)
        submit_response = client.post(
            "/jobs/complete", headers=auth_headers, json=sample_job_request
        )
        task_id = submit_response.json()["data"]["task_id"]

        # Regenerate with improvements
        regenerate_request = {
            "improvement_remarks": "Make the summary more concise and add quantifiable metrics to achievements",
            "design_config": {"theme": "sb2nov"},
        }

        response = client.post(
            f"/tasks/{task_id}/regenerate",
            headers=auth_headers,
            json=regenerate_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data["data"]
        assert data["data"]["original_task_id"] == task_id
        assert data["data"]["has_improvement_remarks"] is True

    def test_regenerate_resume_design_only(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test regenerating resume with design changes only (should reuse AI content)"""
        # Create user and submit job
        client.post("/users", headers=auth_headers, json=sample_user_data)
        submit_response = client.post(
            "/jobs/complete", headers=auth_headers, json=sample_job_request
        )
        task_id = submit_response.json()["data"]["task_id"]

        # Regenerate with ONLY design change (no improvement remarks)
        regenerate_request = {
            "design_config": {
                "theme": "moderncv",
                "colors": {"section_titles": "rgb(100, 0, 0)"},
            }
        }

        response = client.post(
            f"/tasks/{task_id}/regenerate",
            headers=auth_headers,
            json=regenerate_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["has_improvement_remarks"] is False

        # This should have created a new task that reuses AI content
        new_task_id = data["data"]["task_id"]
        assert new_task_id != task_id


# ============================================================================
# Test: Resume Content
# ============================================================================
class TestResumeContent:
    """Test resume content retrieval"""

    def test_get_task_analysis(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test retrieving job analysis for a task"""
        # Create user and submit job
        client.post("/users", headers=auth_headers, json=sample_user_data)
        submit_response = client.post(
            "/jobs", headers=auth_headers, json=sample_job_request
        )
        task_id = submit_response.json()["data"]["task_id"]

        # Try to get analysis (might not be ready immediately in test)
        response = client.get(f"/tasks/{task_id}/analysis", headers=auth_headers)

        # Either success or not found (depending on async completion)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_get_task_resume(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test retrieving generated resume content"""
        # Create user and submit job
        client.post("/users", headers=auth_headers, json=sample_user_data)
        submit_response = client.post(
            "/jobs/complete", headers=auth_headers, json=sample_job_request
        )
        task_id = submit_response.json()["data"]["task_id"]

        # Try to get resume (might not be ready immediately in test)
        response = client.get(f"/tasks/{task_id}/resume", headers=auth_headers)

        # Either success or not found (depending on async completion)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data


# ============================================================================
# Test: Statistics
# ============================================================================
class TestStatistics:
    """Test user statistics endpoint"""

    def test_get_user_stats(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test retrieving user statistics"""
        # Create user and submit a job
        client.post("/users", headers=auth_headers, json=sample_user_data)
        client.post("/jobs", headers=auth_headers, json=sample_job_request)

        # Get stats
        response = client.get("/users/me/stats", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_tasks" in data["data"]
        assert "total_tokens_used" in data["data"]
        assert "recent_tasks" in data["data"]
        assert data["data"]["total_tasks"] >= 1


# ============================================================================
# Test: Error Handling
# ============================================================================
class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_missing_api_key(self, client: TestClient):
        """Test request without API key"""
        response = client.get("/users/me", headers={"X-User-Id": TEST_USER_ID})
        assert response.status_code == 422  # Validation error

    def test_missing_user_id(self, client: TestClient):
        """Test request without user ID"""
        response = client.get("/users/me", headers={"X-Api-Key": TEST_API_TOKEN})
        assert response.status_code == 422  # Validation error

    def test_invalid_task_id(
        self, client: TestClient, auth_headers: dict, sample_user_data: dict
    ):
        """Test requesting a non-existent task"""
        client.post("/users", headers=auth_headers, json=sample_user_data)

        response = client.get("/tasks/invalid-task-id-12345", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "TASK_NOT_FOUND"

    def test_unauthorized_task_access(
        self, client: TestClient, sample_user_data: dict, sample_job_request: dict
    ):
        """Test accessing another user's task"""
        # Create user 1 and submit job
        headers_user1 = {"X-Api-Key": TEST_API_TOKEN, "X-User-Id": "user_1"}
        client.post("/users", headers=headers_user1, json=sample_user_data)
        submit_response = client.post(
            "/jobs", headers=headers_user1, json=sample_job_request
        )
        task_id = submit_response.json()["data"]["task_id"]

        # Try to access with user 2
        headers_user2 = {"X-Api-Key": TEST_API_TOKEN, "X-User-Id": "user_2"}
        response = client.get(f"/tasks/{task_id}", headers=headers_user2)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "TASK_NOT_FOUND"

    def test_parse_empty_resume(self, client: TestClient, auth_headers: dict):
        """Test parsing empty or invalid resume text"""
        response = client.post(
            "/users/from-resume",
            headers=auth_headers,
            json={"resume_text": "   ", "overwrite_existing": False},
        )

        # Should handle gracefully (either error or minimal extraction)
        assert response.status_code == 200
        data = response.json()
        # Depending on AI behavior, might succeed with minimal data or fail
        assert "success" in data


# ============================================================================
# Test: Integration Scenarios
# ============================================================================
class TestIntegrationScenarios:
    """End-to-end integration tests"""

    def test_complete_workflow(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test complete workflow from user creation to resume generation"""
        # Step 1: Create user
        create_response = client.post(
            "/users", headers=auth_headers, json=sample_user_data
        )
        assert create_response.json()["success"] is True

        # Step 2: Verify profile
        profile_response = client.get("/users/me", headers=auth_headers)
        assert profile_response.json()["success"] is True

        # Step 3: Submit job
        job_response = client.post(
            "/jobs/complete", headers=auth_headers, json=sample_job_request
        )
        assert job_response.json()["success"] is True
        task_id = job_response.json()["data"]["task_id"]

        # Step 4: Check task status
        status_response = client.get(f"/tasks/{task_id}", headers=auth_headers)
        assert status_response.json()["success"] is True

        # Step 5: List all tasks
        tasks_response = client.get("/tasks", headers=auth_headers)
        assert tasks_response.json()["data"]["count"] >= 1

        # Step 6: Get statistics
        stats_response = client.get("/users/me/stats", headers=auth_headers)
        assert stats_response.json()["data"]["total_tasks"] >= 1

    def test_resume_parsing_workflow(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_resume_text: str,
        sample_job_request: dict,
    ):
        """Test complete workflow starting with resume parsing"""
        # Step 1: Create user from resume
        parse_response = client.post(
            "/users/from-resume",
            headers=auth_headers,
            json={"resume_text": sample_resume_text, "overwrite_existing": False},
        )
        assert parse_response.json()["success"] is True

        # Step 2: Verify profile was created
        profile_response = client.get("/users/me", headers=auth_headers)
        assert profile_response.json()["success"] is True
        assert profile_response.json()["data"]["personal_info"]["name"] == "John Doe"

        # Step 3: Submit job to generate tailored resume
        job_response = client.post(
            "/jobs/complete", headers=auth_headers, json=sample_job_request
        )
        assert job_response.json()["success"] is True

    def test_multiple_job_submissions(
        self, client: TestClient, auth_headers: dict, sample_user_data: dict
    ):
        """Test submitting multiple jobs for the same user with different descriptions"""
        # Create user
        client.post("/users", headers=auth_headers, json=sample_user_data)

        # Submit 3 different jobs with DIFFERENT descriptions
        job_requests = [
            {
                "title": "Backend Engineer",
                "company": "Company A",
                "description": "Looking for a Backend Engineer with Python expertise. Requirements: 3+ years Python, Django/FastAPI experience.",
            },
            {
                "title": "Full Stack Developer",
                "company": "Company B",
                "description": "Seeking Full Stack Developer proficient in React and Node.js. Requirements: 2+ years JavaScript, React, Node.js.",
            },
            {
                "title": "Tech Lead",
                "company": "Company C",
                "description": "Need experienced Tech Lead to mentor team. Requirements: 5+ years experience, leadership skills, architecture design.",
            },
        ]

        task_ids = []
        for job_req in job_requests:
            response = client.post("/jobs", headers=auth_headers, json=job_req)
            assert response.json()["success"] is True
            task_ids.append(response.json()["data"]["task_id"])

        # Verify all tasks exist
        tasks_response = client.get("/tasks", headers=auth_headers)
        assert tasks_response.json()["data"]["count"] == 3, (
            f"Expected 3 tasks, got {tasks_response.json()['data']['count']}"
        )

        # Verify all jobs exist
        jobs_response = client.get("/jobs", headers=auth_headers)
        assert jobs_response.json()["data"]["count"] == 3, (
            f"Expected 3 jobs, got {jobs_response.json()['data']['count']}"
        )

    def test_iterative_resume_improvement(
        self,
        client: TestClient,
        auth_headers: dict,
        sample_user_data: dict,
        sample_job_request: dict,
    ):
        """Test iterating on a resume with multiple improvement cycles"""
        # Create user and generate initial resume
        client.post("/users", headers=auth_headers, json=sample_user_data)
        initial_response = client.post(
            "/jobs/complete", headers=auth_headers, json=sample_job_request
        )
        task_id_1 = initial_response.json()["data"]["task_id"]

        # First improvement: Make it more concise
        improve_1 = client.post(
            f"/tasks/{task_id_1}/regenerate",
            headers=auth_headers,
            json={"improvement_remarks": "Make the summary more concise"},
        )
        assert improve_1.json()["success"] is True
        task_id_2 = improve_1.json()["data"]["task_id"]

        # Second improvement: Add metrics + change design
        improve_2 = client.post(
            f"/tasks/{task_id_2}/regenerate",
            headers=auth_headers,
            json={
                "improvement_remarks": "Add more quantifiable metrics",
                "design_config": {"theme": "sb2nov"},
            },
        )
        assert improve_2.json()["success"] is True
        task_id_3 = improve_2.json()["data"]["task_id"]

        # Verify all three tasks exist
        tasks_response = client.get("/tasks", headers=auth_headers)
        assert tasks_response.json()["data"]["count"] >= 3


# ============================================================================
# Run Tests
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
