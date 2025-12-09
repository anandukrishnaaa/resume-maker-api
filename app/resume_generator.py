"""
AI-Powered Resume Generator API with Multi-User Support and REST-Compliant Endpoints
"""

import os
import json
import hashlib
import uuid
import yaml
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import (
    SQLModel,
    Field as SQLField,
    create_engine,
    Session,
    select,
    JSON,
    Column,
)

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb

from rendercv.data.models import RenderCVDataModel
import rendercv.renderer as renderer

from environs import Env
from huey import SqliteHuey

# ============================================================================
# Logging Setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Environment & Config
# ============================================================================
env = Env()
env.read_env()
api_key = env.str("OPENROUTER_API_KEY", "your-key-here")
API_TOKEN = "secret-token-123"

app = FastAPI(title="Resume Generator AI", version="1.0.0")

# ============================================================================
# Huey Configuration (Task Queue)
# ============================================================================
huey = SqliteHuey(filename="tmp/huey_queue.db")


# ============================================================================
# Pydantic Models for Nested Structures
# ============================================================================
class SocialNetworkInput(BaseModel):
    """Social network profile"""

    network: str = Field(..., example="LinkedIn")
    username: str = Field(..., example="johndoe")


class EducationInput(BaseModel):
    """Education entry for profile"""

    institution: str = Field(..., example="Indian Institute of Technology")
    area: str = Field(..., example="Computer Science")
    degree: str = Field(..., example="B.Tech")
    start_date: str = Field(..., example="2018-08")
    end_date: str = Field(..., example="2022-05")
    highlights: Optional[List[str]] = Field(
        default=None, example=["GPA: 3.8/4.0", "President of Computer Science Club"]
    )
    location: Optional[str] = Field(default=None, example="Mumbai, India")


class ExperienceInput(BaseModel):
    """Experience entry for profile"""

    company: str = Field(..., example="TechCorp India")
    position: str = Field(..., example="Software Engineer")
    start_date: str = Field(..., example="2022-06")
    end_date: str = Field(..., example="present")
    location: Optional[str] = Field(default=None, example="Bangalore, India")
    highlights: Optional[List[str]] = Field(
        default=None,
        example=[
            "Developed scalable microservices using Python and FastAPI",
            "Reduced API response time by 40%",
        ],
    )


class SkillsInput(BaseModel):
    """Skills organized by category"""

    languages: Optional[List[str]] = Field(
        default=None, example=["Python", "JavaScript", "Go"]
    )
    frameworks: Optional[List[str]] = Field(
        default=None, example=["FastAPI", "React", "Django"]
    )
    tools: Optional[List[str]] = Field(
        default=None, example=["Docker", "Kubernetes", "AWS"]
    )
    databases: Optional[List[str]] = Field(
        default=None, example=["PostgreSQL", "MongoDB", "Redis"]
    )

    class Config:
        # Allow additional fields for custom categories
        extra = "allow"


class ProfileDataInput(BaseModel):
    """Complete profile data structure"""

    education: Optional[List[EducationInput]] = Field(
        default=None,
        example=[
            {
                "institution": "Indian Institute of Technology",
                "area": "Computer Science",
                "degree": "B.Tech",
                "start_date": "2018-08",
                "end_date": "2022-05",
                "highlights": ["GPA: 3.8/4.0"],
            }
        ],
    )
    experience: Optional[List[ExperienceInput]] = Field(
        default=None,
        example=[
            {
                "company": "TechCorp",
                "position": "Software Engineer",
                "start_date": "2022-06",
                "end_date": "present",
                "highlights": ["Built scalable APIs"],
            }
        ],
    )
    skills: Optional[SkillsInput] = Field(
        default=None,
        example={
            "languages": ["Python", "JavaScript"],
            "frameworks": ["FastAPI", "React"],
        },
    )


# ============================================================================
# Resume Parsing Models (AI Output)
# ============================================================================
class ExtractedContactInfo(BaseModel):
    """Contact information extracted from resume"""

    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    phone: str = Field(..., description="Phone number without country code")
    country_code: str = Field(default="+1", description="Country code")
    location: str = Field(
        ..., description="Full location (City, State/Province, Country)"
    )
    social_networks: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of social networks like [{'network': 'LinkedIn', 'username': 'johndoe'}]",
    )


class ExtractedEducation(BaseModel):
    """Education entry extracted from resume"""

    institution: str
    area: str = Field(..., description="Field of study")
    degree: str
    start_date: str = Field(..., description="Start date in YYYY-MM format")
    end_date: str = Field(..., description="End date in YYYY-MM format or 'present'")
    highlights: Optional[List[str]] = Field(default=None)
    location: Optional[str] = None


class ExtractedExperience(BaseModel):
    """Work experience extracted from resume"""

    company: str
    position: str
    start_date: str = Field(..., description="Start date in YYYY-MM format")
    end_date: str = Field(..., description="End date in YYYY-MM format or 'present'")
    location: Optional[str] = None
    highlights: List[str] = Field(
        default_factory=list, description="List of achievements and responsibilities"
    )


class ExtractedSkills(BaseModel):
    """Skills extracted from resume, organized by category"""

    languages: Optional[List[str]] = None
    frameworks: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    databases: Optional[List[str]] = None
    soft_skills: Optional[List[str]] = None
    certifications: Optional[List[str]] = None

    class Config:
        extra = "allow"  # Allow custom skill categories


class ExtractedProfile(BaseModel):
    """Complete profile extracted from resume"""

    contact_info: ExtractedContactInfo
    education: List[ExtractedEducation]
    experience: List[ExtractedExperience]
    skills: ExtractedSkills


# ============================================================================
# Pydantic Output Models (For Structured AI Response)
# ============================================================================
class EducationEntry(BaseModel):
    """Education entry matching RenderCV format"""

    institution: str
    area: str
    degree: str
    start_date: str  # YYYY-MM format
    end_date: str  # YYYY-MM or "present"
    highlights: List[str] = Field(default_factory=list)
    location: Optional[str] = None


class ExperienceEntry(BaseModel):
    """Experience entry matching RenderCV format"""

    company: str
    position: str
    start_date: str  # YYYY-MM format
    end_date: str  # YYYY-MM or "present"
    location: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)


class ProjectEntry(BaseModel):
    """Project entry matching RenderCV format"""

    name: str
    date: str
    highlights: List[str] = Field(default_factory=list)


class SkillEntry(BaseModel):
    """Skill entry matching RenderCV OneLineEntry format"""

    label: str
    details: str


class ResumeSections(BaseModel):
    """The main sections of the resume generated by AI"""

    summary: List[str] = Field(
        description="A brief professional summary as a list of 2-4 strings"
    )
    education: List[EducationEntry]
    experience: List[ExperienceEntry]
    projects: Optional[List[ProjectEntry]] = None
    skills: List[SkillEntry]


# ============================================================================
# Standardized API Response Models
# ============================================================================
class APIResponse(BaseModel):
    """Standard API response wrapper"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ============================================================================
# Database Models with Multi-User Support
# ============================================================================
class UserProfile(SQLModel, table=True):
    """Stores user-specific profile information"""

    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: str = SQLField(index=True, unique=True)
    profile_data: str  # JSON string of work history, skills, etc.
    created_at: datetime = SQLField(default_factory=datetime.now)
    updated_at: datetime = SQLField(default_factory=datetime.now)


class PersonalInfo(SQLModel, table=True):
    """Stores personal contact information per user"""

    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: str = SQLField(foreign_key="userprofile.user_id", index=True, unique=True)
    name: str
    email: str
    phone: str
    country_code: str = SQLField(default="+91")
    location: str
    social_networks: str  # JSON string
    updated_at: datetime = SQLField(default_factory=datetime.now)


class JobDescription(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: str = SQLField(foreign_key="userprofile.user_id", index=True)
    title: str
    company: str
    description: str
    content_hash: str = SQLField(index=True)
    created_at: datetime = SQLField(default_factory=datetime.now)


class JobAnalysis(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: str = SQLField(foreign_key="userprofile.user_id", index=True)
    job_id: int = SQLField(foreign_key="jobdescription.id")
    analysis_text: str
    ai_model: Optional[str] = None
    created_at: datetime = SQLField(default_factory=datetime.now)


class ResumeContent(SQLModel, table=True):
    """Stores the AI-generated resume JSON"""

    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: str = SQLField(foreign_key="userprofile.user_id", index=True)
    task_id: str = SQLField(foreign_key="tasklog.task_id", index=True)
    job_id: int = SQLField(foreign_key="jobdescription.id")
    content: str  # JSON string of ResumeSections
    design_config: Optional[str] = None  # JSON string of design overrides
    locale_config: Optional[str] = None  # JSON string of locale overrides
    rendercv_settings: Optional[str] = None  # JSON string of rendercv settings
    ai_model: Optional[str] = None
    improvement_remarks: Optional[str] = None
    created_at: datetime = SQLField(default_factory=datetime.now)


class TaskLog(SQLModel, table=True):
    """Tracks the lifecycle of a request via UUID"""

    task_id: str = SQLField(primary_key=True, index=True)
    user_id: str = SQLField(foreign_key="userprofile.user_id", index=True)
    job_id: Optional[int] = SQLField(default=None, foreign_key="jobdescription.id")
    status: str = SQLField(default="pending")
    total_tokens: int = SQLField(default=0)
    logs: List[str] = SQLField(default=[], sa_column=Column(JSON))
    output_folder: Optional[str] = None
    pdf_path: Optional[str] = None
    ai_model: Optional[str] = None
    created_at: datetime = SQLField(default_factory=datetime.now)
    updated_at: datetime = SQLField(default_factory=datetime.now)


# ============================================================================
# Database & Security
# ============================================================================
DATABASE_URL = "sqlite:///./resume_generator.db"
engine = create_engine(
    DATABASE_URL, echo=False, connect_args={"check_same_thread": False}
)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API Key")


async def get_user_id(x_user_id: str = Header(...)) -> str:
    """Extract user_id from request header"""
    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header required")
    return x_user_id


# ============================================================================
# API Input Models
# ============================================================================
class CreateUserRequest(BaseModel):
    """Request to create a new user with personal info"""

    name: str = Field(..., example="John Doe")
    email: str = Field(..., example="john.doe@example.com")
    phone: str = Field(..., example="9876543210")
    country_code: str = Field(default="+91", example="+91")
    location: str = Field(..., example="Mumbai, India")
    social_networks: Optional[List[SocialNetworkInput]] = Field(
        default=None,
        example=[
            {"network": "LinkedIn", "username": "johndoe"},
            {"network": "GitHub", "username": "johndoe-dev"},
        ],
    )
    profile_data: Optional[ProfileDataInput] = Field(
        default=None,
        example={
            "education": [
                {
                    "institution": "IIT Bombay",
                    "area": "Computer Science",
                    "degree": "B.Tech",
                    "start_date": "2018-08",
                    "end_date": "2022-05",
                }
            ],
            "experience": [
                {
                    "company": "TechCorp",
                    "position": "Software Engineer",
                    "start_date": "2022-06",
                    "end_date": "present",
                    "highlights": ["Built scalable APIs"],
                }
            ],
            "skills": {
                "languages": ["Python", "JavaScript"],
                "frameworks": ["FastAPI", "React"],
            },
        },
    )

    class Config:
        schema_extra = {
            "example": {
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
                            "highlights": ["GPA: 3.8/4.0"],
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
                                "Developed scalable microservices using Python",
                                "Reduced API response time by 40%",
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
        }


class ResumeParseRequest(BaseModel):
    """Request to parse resume text"""

    resume_text: str = Field(
        ...,
        example="""John Doe
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
""",
        description="Complete resume text to parse",
    )
    overwrite_existing: bool = Field(
        default=False,
        description="If true, overwrites existing profile. If false, returns error if profile exists.",
    )


class JobRequest(BaseModel):
    title: str = Field(..., example="Senior Backend Engineer")
    company: str = Field(..., example="BigTech Corp")
    description: str = Field(
        ...,
        example="""We are looking for a Senior Backend Engineer to join our team.

Requirements:
- 3+ years of experience with Python
- Experience with FastAPI or Django
- Strong knowledge of microservices architecture
- Experience with AWS or GCP

Responsibilities:
- Design and implement scalable backend services
- Mentor junior developers
- Collaborate with frontend team""",
    )
    ai_model: Optional[str] = Field(
        default=None,
        example="google/gemini-2.5-flash-lite",
        description="AI model to use (e.g., anthropic/claude-3.5-sonnet, openai/gpt-4)",
    )
    design_config: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"theme": "moderncv", "colors": {"section_titles": "rgb(200, 0, 0)"}},
        description="Custom design configuration (overrides theme defaults)",
    )
    locale_config: Optional[Dict[str, Any]] = Field(default=None)
    rendercv_settings: Optional[Dict[str, Any]] = Field(default=None)


class RegenerateResumeRequest(BaseModel):
    """Request to regenerate resume from existing task with new configs"""

    ai_model: Optional[str] = Field(default=None, example="anthropic/claude-3.5-sonnet")
    design_config: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"theme": "sb2nov", "colors": {"name": "rgb(0, 100, 200)"}},
    )
    locale_config: Optional[Dict[str, Any]] = None
    rendercv_settings: Optional[Dict[str, Any]] = None
    improvement_remarks: Optional[str] = Field(
        default=None,
        example="Make the summary more concise and add quantifiable metrics to achievements. Emphasize leadership experience.",
    )


class PersonalInfoUpdate(BaseModel):
    name: str = Field(..., example="John Doe")
    email: str = Field(..., example="john.doe@example.com")
    phone: str = Field(..., example="9876543210")
    country_code: str = Field(..., example="+91")
    location: str = Field(..., example="Mumbai, India")
    social_networks: Optional[List[SocialNetworkInput]] = Field(
        default=None, example=[{"network": "LinkedIn", "username": "johndoe"}]
    )


class ProfileDataUpdate(BaseModel):
    """Update user's work history, education, skills"""

    education: Optional[List[EducationInput]] = None
    experience: Optional[List[ExperienceInput]] = None
    skills: Optional[SkillsInput] = None


# ============================================================================
# AI Agents & Helpers
# ============================================================================
def get_openrouter_model(model_id: Optional[str] = None):
    """Get OpenRouter model with optional custom model ID"""
    default_model = "google/gemini-2.5-flash-lite"
    return OpenAIChat(
        id=model_id or default_model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def create_jd_analyzer_agent(model_id: Optional[str] = None) -> Agent:
    db = SqliteDb(db_file="tmp/agents.db", session_table="jd_analyzer_sessions")
    return Agent(
        name="JD Analyzer",
        model=get_openrouter_model(model_id),
        instructions=[
            "You are an expert job description analyzer.",
            "Analyze the provided job description and extract key insights.",
            "Format your response as structured text with clear sections.",
            "Focus on: 1. Required Skills, 2. Soft Skills, 3. Experience, 4. Keywords, 5. Responsibilities.",
        ],
        db=db,
        markdown=True,
    )


def create_resume_generator_agent(model_id: Optional[str] = None) -> Agent:
    db = SqliteDb(db_file="tmp/agents.db", session_table="resume_generator_sessions")
    return Agent(
        name="Resume Generator",
        model=get_openrouter_model(model_id),
        instructions=[
            "You are an expert resume writer.",
            "You will be given a Job Description, an AI Analysis of that job, and a User Profile.",
            "Your task is to select and adapt the user's experience to fit the job.",
            "Do NOT invent false information. Use the user's actual profile but rephrase highlights to match keywords.",
            "For dates, use YYYY-MM format or 'present'.",
            "",
            "IMPORTANT FORMATTING RULES:",
            "- For skills: label is the category (e.g., 'Programming Languages'), details is a comma-separated string (e.g., 'Python, JavaScript, Go')",
            "- For projects: Only include if the user has relevant projects. If no projects, return empty list or null.",
            "- For experience highlights: Use action verbs and quantify achievements when possible",
            "- Keep summary to 2-4 concise bullet points",
        ],
        output_schema=ResumeSections,
        db=db,
        markdown=False,
    )


def create_resume_parser_agent(model_id: Optional[str] = None) -> Agent:
    """Create an AI agent that parses resume text into structured data"""
    db = SqliteDb(db_file="tmp/agents.db", session_table="resume_parser_sessions")
    return Agent(
        name="Resume Parser",
        model=get_openrouter_model(model_id),
        instructions=[
            "You are an expert resume parser that extracts structured information from resumes.",
            "Extract contact information, education, work experience, and skills.",
            "",
            "IMPORTANT FORMATTING RULES:",
            "1. For dates: Use YYYY-MM format. If only year is given, use YYYY-01 for start and YYYY-12 for end.",
            "2. If currently employed, use 'present' as end_date",
            "3. Extract phone numbers without special characters (only digits)",
            "4. For country code: Extract if present (e.g., +91, +1), otherwise use +1 as default",
            "5. For location: Include city, state/province, and country if available",
            "6. For social networks: Extract LinkedIn, GitHub, Twitter, etc. Format as {'network': 'LinkedIn', 'username': 'johndoe'}",
            "7. For skills: Categorize into languages, frameworks, tools, databases, soft_skills, certifications",
            "8. For highlights: Extract bullet points as a list of achievements/responsibilities",
            "9. Be thorough - extract ALL information from the resume",
            "10. If information is missing or unclear, make reasonable inferences based on context",
            "",
            "Output the extracted information in the specified JSON structure.",
        ],
        output_schema=ExtractedProfile,
        db=db,
        markdown=False,
    )


def get_or_create_user_profile(session: Session, user_id: str):
    """Get or create profile for specific user"""
    user = session.exec(
        select(UserProfile).where(UserProfile.user_id == user_id)
    ).first()

    personal = session.exec(
        select(PersonalInfo).where(PersonalInfo.user_id == user_id)
    ).first()

    if not user:
        default_profile = {
            "education": [
                {
                    "institution": "University of Tech",
                    "area": "Computer Science",
                    "degree": "BS",
                    "start_date": "2020-01",
                    "end_date": "2024-01",
                }
            ],
            "experience": [
                {
                    "company": "Tech Corp",
                    "position": "Software Developer",
                    "start_date": "2024-02",
                    "end_date": "present",
                    "highlights": [
                        "Developed scalable APIs using Python and FastAPI",
                        "Optimized database queries reducing response time by 40%",
                    ],
                }
            ],
            "skills": {
                "languages": ["Python", "JavaScript"],
                "tools": ["Git", "Docker", "AWS"],
            },
        }
        user = UserProfile(user_id=user_id, profile_data=json.dumps(default_profile))
        session.add(user)
        session.commit()
        session.refresh(user)

    if not personal:
        personal = PersonalInfo(
            user_id=user_id,
            name="John Doe",
            email="john@example.com",
            phone="9876543210",
            country_code="+91",
            location="Thiruvananthapuram, Kerala, India",
            social_networks=json.dumps(
                [{"network": "LinkedIn", "username": "johndoe"}]
            ),
        )
        session.add(personal)
        session.commit()
        session.refresh(personal)

    return user, personal


# ============================================================================
# HUEY TASKS (Background Workers)
# ============================================================================
@huey.task()
def process_analysis_task(
    task_id: str,
    user_id: str,
    title: str,
    company: str,
    description: str,
    job_id: int,
    auto_resume: bool = False,
    ai_model: Optional[str] = None,
):
    """Background task: Runs AI analysis. Can optionally trigger resume generation."""
    logger.info(
        f"--- Starting Analysis Task: {task_id} (User: {user_id}, Model: {ai_model or 'default'}, Auto-Resume: {auto_resume}) ---"
    )

    with Session(engine) as session:
        task = session.get(TaskLog, task_id)
        if not task:
            return

        try:
            task.status = "analysis_in_progress"
            task.ai_model = ai_model
            task.logs.append(
                f"[{datetime.now()}] Analysis started (Worker, Model: {ai_model or 'default'})"
            )
            session.add(task)
            session.commit()

            analyzer = create_jd_analyzer_agent(ai_model)
            prompt = f"Analyze:\nTitle: {title}\nCompany: {company}\n\n{description}"
            response = analyzer.run(prompt)

            analysis = JobAnalysis(
                user_id=user_id,
                job_id=job_id,
                analysis_text=response.content,
                ai_model=ai_model,
            )
            session.add(analysis)

            # Safe Token Logging
            tokens = 0
            if response.metrics:
                try:
                    tokens = response.metrics.to_dict().get("total_tokens", 0)
                except:
                    tokens = getattr(response.metrics, "total_tokens", 0)

            task.total_tokens += int(tokens)
            task.logs.append(f"[{datetime.now()}] Analysis complete. Tokens: {tokens}")
            task.status = "analysis_complete"
            session.add(task)
            session.commit()

            if auto_resume:
                process_resume_task(task_id)

        except Exception as e:
            logger.error(f"Analysis Error: {e}", exc_info=True)
            task.status = "error"
            task.logs.append(f"[{datetime.now()}] Error: {str(e)}")
            session.add(task)
            session.commit()


@huey.task()
def process_resume_task(
    task_id: str,
    design_override: Optional[Dict] = None,
    locale_override: Optional[Dict] = None,
    rendercv_settings_override: Optional[Dict] = None,
    ai_model_override: Optional[str] = None,
    improvement_remarks: Optional[str] = None,
):
    """Background task: Generates JSON -> YAML -> PDF in dedicated folder per task"""
    logger.info(f"--- Starting Resume Task: {task_id} ---")

    with Session(engine) as session:
        task = session.get(TaskLog, task_id)
        if not task or not task.job_id:
            return

        try:
            task.status = "resume_in_progress"
            task.logs.append(f"[{datetime.now()}] Resume generation started (Worker)")

            # Update AI model if provided
            if ai_model_override:
                task.ai_model = ai_model_override

            session.add(task)
            session.commit()

            # Fetch Data
            jd = session.get(JobDescription, task.job_id)
            analysis = session.exec(
                select(JobAnalysis)
                .where(JobAnalysis.job_id == task.job_id)
                .where(JobAnalysis.user_id == task.user_id)
                .order_by(JobAnalysis.id.desc())
            ).first()
            user, personal = get_or_create_user_profile(session, task.user_id)

            if not analysis:
                raise ValueError("No analysis found")

            # Check if we can reuse existing content (optimization)
            existing_resume = session.exec(
                select(ResumeContent)
                .where(ResumeContent.job_id == task.job_id)
                .where(ResumeContent.user_id == task.user_id)
                .order_by(ResumeContent.created_at.desc())
            ).first()

            sections_dict = None
            tokens_used = 0

            # OPTIMIZATION: Only call AI if we have improvement remarks OR no existing content
            if improvement_remarks or not existing_resume:
                task.logs.append(
                    f"[{datetime.now()}] Calling AI to generate/improve content"
                )

                # 1. AI Generation
                generator = create_resume_generator_agent(
                    ai_model_override or task.ai_model
                )

                # Build prompt with optional improvement remarks
                prompt = f"Generate Resume Content for:\nJOB: {jd.title}\nANALYSIS: {analysis.analysis_text}\nUSER: {user.profile_data}"

                if improvement_remarks and existing_resume:
                    prompt += f"\n\nPREVIOUS RESUME CONTENT:\n{existing_resume.content}"
                    prompt += f"\n\nIMPROVEMENT INSTRUCTIONS:\n{improvement_remarks}\n\nPlease improve the resume based on the above instructions."
                    task.logs.append(
                        f"[{datetime.now()}] Regenerating with improvement remarks"
                    )

                response = generator.run(prompt)
                resume_sections_obj = response.content

                # Metrics
                if response.metrics:
                    try:
                        tokens_used = response.metrics.to_dict().get("total_tokens", 0)
                    except:
                        tokens_used = getattr(response.metrics, "total_tokens", 0)

                task.total_tokens += int(tokens_used)
                task.logs.append(
                    f"[{datetime.now()}] Structured content generated. Tokens: {tokens_used}"
                )

                # Convert to dict
                sections_dict = resume_sections_obj.model_dump(exclude_none=True)
            else:
                # REUSE existing AI-generated content
                task.logs.append(
                    f"[{datetime.now()}] Reusing existing AI-generated content (design-only change)"
                )
                sections_dict = json.loads(existing_resume.content)
                task.logs.append(
                    f"[{datetime.now()}] No AI call needed - saved tokens!"
                )

            # 2. Store resume content with configs
            resume_content = ResumeContent(
                user_id=task.user_id,
                task_id=task_id,
                job_id=task.job_id,
                content=json.dumps(sections_dict),
                design_config=json.dumps(design_override) if design_override else None,
                locale_config=json.dumps(locale_override) if locale_override else None,
                rendercv_settings=json.dumps(rendercv_settings_override)
                if rendercv_settings_override
                else None,
                ai_model=ai_model_override or task.ai_model,
                improvement_remarks=improvement_remarks,
            )
            session.add(resume_content)
            session.commit()

            # 3. Fix sections format for RenderCV
            if "projects" in sections_dict and not sections_dict["projects"]:
                del sections_dict["projects"]

            if "skills" in sections_dict:
                skills_formatted = []
                for skill in sections_dict["skills"]:
                    if "label" in skill and "details" in skill:
                        skills_formatted.append(skill)
                sections_dict["skills"] = skills_formatted

            # 4. Format phone with country code
            full_phone = personal.country_code + personal.phone.replace(
                "-", ""
            ).replace(" ", "")

            # Assemble Full Resume Data - ONLY add design config if provided (don't merge with defaults)
            full_cv_dict = {
                "cv": {
                    "name": personal.name,
                    "location": personal.location,
                    "email": personal.email,
                    "phone": full_phone,
                    "sections": sections_dict,
                }
            }

            # ONLY add design config if user provided overrides
            if design_override:
                full_cv_dict["design"] = design_override
            # Otherwise let RenderCV use the theme's default design

            # Add locale config if provided
            if locale_override:
                full_cv_dict["locale"] = locale_override

            # Add rendercv settings if provided
            if rendercv_settings_override:
                full_cv_dict["rendercv_settings"] = rendercv_settings_override

            # Add social_networks only if not empty
            social_networks = json.loads(personal.social_networks)
            if social_networks:
                full_cv_dict["cv"]["social_networks"] = social_networks

            # 5. Render PDF in dedicated task folder
            output_dir = Path("output") / task_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Store folder path in task
            task.output_folder = str(output_dir)

            # Create the RenderCV data model from the dictionary
            cv_data_model = RenderCVDataModel(**full_cv_dict)

            task.logs.append(
                f"[{datetime.now()}] Data model created, generating Typst file..."
            )
            session.add(task)
            session.commit()

            # Generate Typst file and copy theme files
            typst_file_path = renderer.create_a_typst_file_and_copy_theme_files(
                cv_data_model, output_dir
            )

            task.logs.append(
                f"[{datetime.now()}] Typst file generated at {typst_file_path}"
            )
            session.add(task)
            session.commit()

            # Render PDF from the Typst file
            pdf_path = renderer.render_a_pdf_from_typst(typst_file_path)

            task.pdf_path = str(pdf_path)
            task.status = "resume_complete"
            task.logs.append(
                f"[{datetime.now()}] PDF rendered successfully at {pdf_path}"
            )

            session.add(task)
            session.commit()

            logger.info(f"--- Finished Resume Task: {task_id} ---")

        except Exception as e:
            logger.error(f"Resume Error: {e}", exc_info=True)
            task.status = "error"
            task.logs.append(f"[{datetime.now()}] Error: {str(e)}")
            session.add(task)
            session.commit()


# ============================================================================
# API Endpoints - User Profile Management
# ============================================================================
@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.post("/users", dependencies=[Depends(verify_api_key)], response_model=APIResponse)
def create_user(
    request: CreateUserRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Create a new user profile with personal information"""
    existing = session.exec(
        select(UserProfile).where(UserProfile.user_id == user_id)
    ).first()

    if existing:
        return APIResponse(
            success=False, message="User already exists", error="USER_EXISTS"
        )

    # Convert Pydantic models to dictionaries for storage
    if request.profile_data:
        profile_data_json = json.dumps(
            request.profile_data.model_dump(exclude_none=True)
        )
    else:
        # Default empty profile
        default_profile = {
            "education": [],
            "experience": [],
            "skills": {},
        }
        profile_data_json = json.dumps(default_profile)

    user = UserProfile(user_id=user_id, profile_data=profile_data_json)
    session.add(user)
    session.commit()
    session.refresh(user)

    # Create personal info with provided data
    social_networks_list = []
    if request.social_networks:
        social_networks_list = [sn.model_dump() for sn in request.social_networks]

    personal = PersonalInfo(
        user_id=user_id,
        name=request.name,
        email=request.email,
        phone=request.phone,
        country_code=request.country_code,
        location=request.location,
        social_networks=json.dumps(social_networks_list),
    )
    session.add(personal)
    session.commit()
    session.refresh(personal)

    return APIResponse(
        success=True,
        message="User profile created successfully",
        data={
            "user_id": user_id,
            "name": personal.name,
            "email": personal.email,
            "created_at": user.created_at.isoformat(),
        },
    )


@app.post(
    "/users/from-resume",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def create_user_from_resume(
    request: ResumeParseRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """
    Parse resume text using AI and automatically create user profile.

    This endpoint uses AI to extract:
    - Contact information (name, email, phone, location, social networks)
    - Education history
    - Work experience
    - Skills (categorized)

    The extracted data is then used to create a complete user profile.
    """
    try:
        # Check if user already exists
        existing_user = session.exec(
            select(UserProfile).where(UserProfile.user_id == user_id)
        ).first()

        if existing_user and not request.overwrite_existing:
            return APIResponse(
                success=False,
                message="User already exists. Set overwrite_existing=true to update.",
                error="USER_EXISTS",
            )

        # Parse resume with AI
        logger.info(f"Parsing resume for user: {user_id}")
        parser = create_resume_parser_agent()

        response = parser.run(
            f"Parse the following resume and extract all information:\n\n{request.resume_text}"
        )

        extracted: ExtractedProfile = response.content

        # Track tokens used
        tokens_used = 0
        if response.metrics:
            try:
                tokens_used = response.metrics.to_dict().get("total_tokens", 0)
            except:
                tokens_used = getattr(response.metrics, "total_tokens", 0)

        logger.info(f"Resume parsed successfully. Tokens used: {tokens_used}")

        # Convert extracted data to storage format
        profile_data = {
            "education": [
                edu.model_dump(exclude_none=True) for edu in extracted.education
            ],
            "experience": [
                exp.model_dump(exclude_none=True) for exp in extracted.experience
            ],
            "skills": extracted.skills.model_dump(exclude_none=True),
        }

        # Create or update user profile
        if existing_user:
            # Update existing
            existing_user.profile_data = json.dumps(profile_data)
            existing_user.updated_at = datetime.now()
            session.add(existing_user)
            logger.info(f"Updated existing user profile: {user_id}")
        else:
            # Create new
            user = UserProfile(user_id=user_id, profile_data=json.dumps(profile_data))
            session.add(user)
            logger.info(f"Created new user profile: {user_id}")

        session.commit()

        # Create or update personal info
        existing_personal = session.exec(
            select(PersonalInfo).where(PersonalInfo.user_id == user_id)
        ).first()

        contact = extracted.contact_info
        social_networks_json = json.dumps(contact.social_networks or [])

        if existing_personal:
            # Update existing
            existing_personal.name = contact.name
            existing_personal.email = contact.email
            existing_personal.phone = contact.phone
            existing_personal.country_code = contact.country_code
            existing_personal.location = contact.location
            existing_personal.social_networks = social_networks_json
            existing_personal.updated_at = datetime.now()
            session.add(existing_personal)
            logger.info(f"Updated personal info for: {user_id}")
        else:
            # Create new
            personal = PersonalInfo(
                user_id=user_id,
                name=contact.name,
                email=contact.email,
                phone=contact.phone,
                country_code=contact.country_code,
                location=contact.location,
                social_networks=social_networks_json,
            )
            session.add(personal)
            logger.info(f"Created personal info for: {user_id}")

        session.commit()

        return APIResponse(
            success=True,
            message="User profile created successfully from resume",
            data={
                "user_id": user_id,
                "name": contact.name,
                "email": contact.email,
                "education_count": len(extracted.education),
                "experience_count": len(extracted.experience),
                "tokens_used": tokens_used,
                "extracted_data": {
                    "contact_info": contact.model_dump(),
                    "education": [edu.model_dump() for edu in extracted.education],
                    "experience": [exp.model_dump() for exp in extracted.experience],
                    "skills": extracted.skills.model_dump(exclude_none=True),
                },
            },
        )

    except Exception as e:
        logger.error(f"Error parsing resume: {e}", exc_info=True)
        return APIResponse(
            success=False,
            message=f"Failed to parse resume: {str(e)}",
            error="PARSING_ERROR",
        )


@app.get(
    "/users/me", dependencies=[Depends(verify_api_key)], response_model=APIResponse
)
def get_user_profile(
    user_id: str = Depends(get_user_id), session: Session = Depends(get_session)
):
    """Get complete user profile (personal info + work history)"""
    user, personal = get_or_create_user_profile(session, user_id)

    return APIResponse(
        success=True,
        message="User profile retrieved successfully",
        data={
            "user_id": user.user_id,
            "profile_data": json.loads(user.profile_data),
            "personal_info": {
                "name": personal.name,
                "email": personal.email,
                "phone": personal.phone,
                "country_code": personal.country_code,
                "location": personal.location,
                "social_networks": json.loads(personal.social_networks),
            },
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat(),
        },
    )


@app.put(
    "/users/me/contact",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def update_personal_info(
    data: PersonalInfoUpdate,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Update personal contact information"""
    _, personal = get_or_create_user_profile(session, user_id)

    personal.name = data.name
    personal.email = data.email
    personal.phone = data.phone
    personal.country_code = data.country_code
    personal.location = data.location

    if data.social_networks:
        social_networks_list = [sn.model_dump() for sn in data.social_networks]
        personal.social_networks = json.dumps(social_networks_list)

    personal.updated_at = datetime.now()
    session.add(personal)
    session.commit()

    return APIResponse(
        success=True,
        message="Contact information updated successfully",
        data={"user_id": user_id},
    )


@app.put(
    "/users/me/profile",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def update_profile_data(
    data: ProfileDataUpdate,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Update user's work history, education, skills"""
    user, _ = get_or_create_user_profile(session, user_id)

    current_data = json.loads(user.profile_data)

    if data.education is not None:
        current_data["education"] = [
            edu.model_dump(exclude_none=True) for edu in data.education
        ]
    if data.experience is not None:
        current_data["experience"] = [
            exp.model_dump(exclude_none=True) for exp in data.experience
        ]
    if data.skills is not None:
        current_data["skills"] = data.skills.model_dump(exclude_none=True)

    user.profile_data = json.dumps(current_data)
    user.updated_at = datetime.now()

    session.add(user)
    session.commit()

    return APIResponse(
        success=True,
        message="Profile data updated successfully",
        data={"user_id": user_id},
    )


# ============================================================================
# API Endpoints - Job Management
# ============================================================================
@app.get("/jobs", dependencies=[Depends(verify_api_key)], response_model=APIResponse)
def list_jobs(
    user_id: str = Depends(get_user_id), session: Session = Depends(get_session)
):
    """Get all job descriptions submitted by user"""
    jobs = session.exec(
        select(JobDescription)
        .where(JobDescription.user_id == user_id)
        .order_by(JobDescription.created_at.desc())
    ).all()

    return APIResponse(
        success=True,
        message=f"Retrieved {len(jobs)} jobs",
        data={
            "user_id": user_id,
            "count": len(jobs),
            "jobs": [
                {
                    "id": j.id,
                    "title": j.title,
                    "company": j.company,
                    "created_at": j.created_at.isoformat(),
                }
                for j in jobs
            ],
        },
    )


@app.post("/jobs", dependencies=[Depends(verify_api_key)], response_model=APIResponse)
def create_job_with_analysis(
    request: JobRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Create job and run analysis only (manual resume generation)"""
    task_id = str(uuid.uuid4())
    content_hash = hashlib.sha256(
        request.description.strip().encode("utf-8")
    ).hexdigest()

    task = TaskLog(
        task_id=task_id,
        user_id=user_id,
        status="pending",
        logs=[f"[{datetime.now()}] Task created"],
        ai_model=request.ai_model,
    )

    existing_jd = session.exec(
        select(JobDescription)
        .where(JobDescription.content_hash == content_hash)
        .where(JobDescription.user_id == user_id)
    ).first()

    if existing_jd:
        task.job_id = existing_jd.id
        existing_analysis = session.exec(
            select(JobAnalysis)
            .where(JobAnalysis.job_id == existing_jd.id)
            .where(JobAnalysis.user_id == user_id)
        ).first()

        if existing_analysis:
            task.status = "analysis_complete"
            task.logs.append(f"[{datetime.now()}] Used cached analysis")
            session.add(task)
            session.commit()
            return APIResponse(
                success=True,
                message="Analysis loaded from cache",
                data={
                    "task_id": task_id,
                    "status": "analysis_complete",
                },
            )

        session.add(task)
        session.commit()
        process_analysis_task(
            task_id,
            user_id,
            request.title,
            request.company,
            request.description,
            existing_jd.id,
            auto_resume=False,
            ai_model=request.ai_model,
        )
        return APIResponse(
            success=True,
            message="JD found, starting analysis",
            data={
                "task_id": task_id,
                "status": "analysis_pending",
            },
        )

    new_jd = JobDescription(
        user_id=user_id,
        title=request.title,
        company=request.company,
        description=request.description,
        content_hash=content_hash,
    )
    session.add(new_jd)
    session.commit()
    session.refresh(new_jd)

    task.job_id = new_jd.id
    session.add(task)
    session.commit()

    process_analysis_task(
        task_id,
        user_id,
        request.title,
        request.company,
        request.description,
        new_jd.id,
        auto_resume=False,
        ai_model=request.ai_model,
    )

    return APIResponse(
        success=True,
        message="Job submitted, analysis started",
        data={
            "task_id": task_id,
            "status": "analysis_pending",
        },
    )


@app.post(
    "/jobs/complete", dependencies=[Depends(verify_api_key)], response_model=APIResponse
)
def create_job_with_resume(
    request: JobRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Create job and automatically run analysis + resume generation"""
    task_id = str(uuid.uuid4())
    content_hash = hashlib.sha256(
        request.description.strip().encode("utf-8")
    ).hexdigest()

    task = TaskLog(
        task_id=task_id,
        user_id=user_id,
        status="pending",
        logs=[f"[{datetime.now()}] Task created (One-Shot)"],
        ai_model=request.ai_model,
    )

    existing_jd = session.exec(
        select(JobDescription)
        .where(JobDescription.content_hash == content_hash)
        .where(JobDescription.user_id == user_id)
    ).first()

    if existing_jd:
        task.job_id = existing_jd.id
        existing_analysis = session.exec(
            select(JobAnalysis)
            .where(JobAnalysis.job_id == existing_jd.id)
            .where(JobAnalysis.user_id == user_id)
        ).first()

        if existing_analysis:
            task.status = "analysis_complete"
            task.logs.append(f"[{datetime.now()}] Cache hit. Skipping analysis.")
            session.add(task)
            session.commit()

            # Start resume generation with optional configs
            process_resume_task(
                task_id,
                design_override=request.design_config,
                locale_override=request.locale_config,
                rendercv_settings_override=request.rendercv_settings,
                ai_model_override=request.ai_model,
            )

            return APIResponse(
                success=True,
                message="Analysis cached. Resume generation started.",
                data={
                    "task_id": task_id,
                    "status": "resume_pending",
                },
            )

        session.add(task)
        session.commit()
        process_analysis_task(
            task_id,
            user_id,
            request.title,
            request.company,
            request.description,
            existing_jd.id,
            auto_resume=True,
            ai_model=request.ai_model,
        )
        return APIResponse(
            success=True,
            message="JD found. Starting Analysis -> Resume.",
            data={
                "task_id": task_id,
                "status": "analysis_pending",
            },
        )

    new_jd = JobDescription(
        user_id=user_id,
        title=request.title,
        company=request.company,
        description=request.description,
        content_hash=content_hash,
    )
    session.add(new_jd)
    session.commit()
    session.refresh(new_jd)

    task.job_id = new_jd.id
    session.add(task)
    session.commit()

    process_analysis_task(
        task_id,
        user_id,
        request.title,
        request.company,
        request.description,
        new_jd.id,
        auto_resume=True,
        ai_model=request.ai_model,
    )

    return APIResponse(
        success=True,
        message="Job submitted. Starting Analysis -> Resume.",
        data={
            "task_id": task_id,
            "status": "analysis_pending",
        },
    )


# ============================================================================
# API Endpoints - Task Management
# ============================================================================
@app.get("/tasks", dependencies=[Depends(verify_api_key)], response_model=APIResponse)
def list_tasks(
    user_id: str = Depends(get_user_id), session: Session = Depends(get_session)
):
    """Get all tasks for a specific user"""
    tasks = session.exec(
        select(TaskLog)
        .where(TaskLog.user_id == user_id)
        .order_by(TaskLog.created_at.desc())
    ).all()

    return APIResponse(
        success=True,
        message=f"Retrieved {len(tasks)} tasks",
        data={
            "user_id": user_id,
            "count": len(tasks),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "status": t.status,
                    "total_tokens": t.total_tokens,
                    "ai_model": t.ai_model,
                    "created_at": t.created_at.isoformat(),
                    "pdf_url": f"/tasks/{t.task_id}/download" if t.pdf_path else None,
                }
                for t in tasks
            ],
        },
    )


@app.get(
    "/tasks/{task_id}",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def get_task(
    task_id: str,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Get status and details of a specific task"""
    task = session.get(TaskLog, task_id)
    if not task or task.user_id != user_id:
        return APIResponse(
            success=False, message="Task not found", error="TASK_NOT_FOUND"
        )

    pdf_url = None
    if task.status == "resume_complete" and task.pdf_path:
        pdf_url = f"/tasks/{task_id}/download"

    return APIResponse(
        success=True,
        message="Task retrieved successfully",
        data={
            "task_id": task.task_id,
            "status": task.status,
            "logs": task.logs,
            "total_tokens": task.total_tokens,
            "pdf_url": pdf_url,
            "output_folder": task.output_folder,
            "ai_model": task.ai_model,
        },
    )


@app.get(
    "/tasks/{task_id}/analysis",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def get_task_analysis(
    task_id: str,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Get job analysis for a specific task"""
    task = session.get(TaskLog, task_id)
    if not task or task.user_id != user_id:
        return APIResponse(
            success=False, message="Task not found", error="TASK_NOT_FOUND"
        )

    analysis = session.exec(
        select(JobAnalysis)
        .where(JobAnalysis.job_id == task.job_id)
        .where(JobAnalysis.user_id == user_id)
        .order_by(JobAnalysis.id.desc())
    ).first()

    if not analysis:
        return APIResponse(
            success=False, message="Analysis not found", error="ANALYSIS_NOT_FOUND"
        )

    return APIResponse(
        success=True,
        message="Analysis retrieved successfully",
        data={
            "task_id": task_id,
            "job_id": task.job_id,
            "analysis_text": analysis.analysis_text,
            "ai_model": analysis.ai_model,
        },
    )


@app.get(
    "/tasks/{task_id}/resume",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def get_task_resume(
    task_id: str,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Get the AI-generated resume content (JSON)"""
    resume = session.exec(
        select(ResumeContent)
        .where(ResumeContent.task_id == task_id)
        .where(ResumeContent.user_id == user_id)
    ).first()

    if not resume:
        return APIResponse(
            success=False, message="Resume content not found", error="RESUME_NOT_FOUND"
        )

    return APIResponse(
        success=True,
        message="Resume content retrieved successfully",
        data={
            "task_id": resume.task_id,
            "job_id": resume.job_id,
            "content": json.loads(resume.content),
            "design_config": json.loads(resume.design_config)
            if resume.design_config
            else None,
            "locale_config": json.loads(resume.locale_config)
            if resume.locale_config
            else None,
            "rendercv_settings": json.loads(resume.rendercv_settings)
            if resume.rendercv_settings
            else None,
            "ai_model": resume.ai_model,
            "improvement_remarks": resume.improvement_remarks,
            "created_at": resume.created_at.isoformat(),
        },
    )


@app.post(
    "/tasks/{task_id}/resume",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def create_task_resume(
    task_id: str,
    request: Optional[RegenerateResumeRequest] = None,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Manually trigger resume generation after analysis (with optional config overrides)"""
    task = session.get(TaskLog, task_id)
    if not task or task.user_id != user_id:
        return APIResponse(
            success=False, message="Task not found", error="TASK_NOT_FOUND"
        )

    if "analysis_complete" not in task.status and "resume" not in task.status:
        return APIResponse(
            success=False,
            message="Analysis not yet complete",
            error="ANALYSIS_INCOMPLETE",
        )

    task.status = "resume_pending"
    session.add(task)
    session.commit()

    # Extract optional overrides from request
    design_override = request.design_config if request else None
    locale_override = request.locale_config if request else None
    settings_override = request.rendercv_settings if request else None
    ai_model_override = request.ai_model if request else None
    improvement_remarks = request.improvement_remarks if request else None

    process_resume_task(
        task_id,
        design_override=design_override,
        locale_override=locale_override,
        rendercv_settings_override=settings_override,
        ai_model_override=ai_model_override,
        improvement_remarks=improvement_remarks,
    )

    return APIResponse(
        success=True,
        message="Resume generation started",
        data={
            "task_id": task_id,
            "status": "resume_pending",
        },
    )


@app.post(
    "/tasks/{task_id}/regenerate",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def regenerate_task_resume(
    task_id: str,
    request: RegenerateResumeRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """
    Regenerate resume from existing task with new configurations and/or improvement remarks.
    Creates a new task with the same job but different design/AI settings.
    """
    original_task = session.get(TaskLog, task_id)
    if not original_task or original_task.user_id != user_id:
        return APIResponse(
            success=False, message="Original task not found", error="TASK_NOT_FOUND"
        )

    if not original_task.job_id:
        return APIResponse(
            success=False,
            message="Original task has no associated job",
            error="NO_JOB_FOUND",
        )

    # Create new task
    new_task_id = str(uuid.uuid4())
    new_task = TaskLog(
        task_id=new_task_id,
        user_id=user_id,
        job_id=original_task.job_id,
        status="resume_pending",
        logs=[f"[{datetime.now()}] Task created (Regeneration from {task_id})"],
        ai_model=request.ai_model or original_task.ai_model,
    )
    session.add(new_task)
    session.commit()

    # Start resume generation with new configs and improvement remarks
    process_resume_task(
        new_task_id,
        design_override=request.design_config,
        locale_override=request.locale_config,
        rendercv_settings_override=request.rendercv_settings,
        ai_model_override=request.ai_model,
        improvement_remarks=request.improvement_remarks,
    )

    return APIResponse(
        success=True,
        message="Resume regeneration started with new configuration",
        data={
            "task_id": new_task_id,
            "original_task_id": task_id,
            "status": "resume_pending",
            "has_improvement_remarks": bool(request.improvement_remarks),
        },
    )


@app.get("/tasks/{task_id}/download")
def download_task_pdf(
    task_id: str, x_api_key: str = Header(...), x_user_id: str = Header(...)
):
    """Download PDF file for a specific task"""
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with Session(engine) as session:
        task = session.get(TaskLog, task_id)
        if not task or task.user_id != x_user_id:
            raise HTTPException(status_code=404, detail="Task not found")

        if not task.pdf_path:
            raise HTTPException(status_code=404, detail="PDF not yet generated")

        file_path = Path(task.pdf_path)

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=file_path, filename=file_path.name, media_type="application/pdf"
        )


# ============================================================================
# API Endpoints - Statistics
# ============================================================================
@app.get(
    "/users/me/stats",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def get_user_stats(
    user_id: str = Depends(get_user_id), session: Session = Depends(get_session)
):
    """Get user-specific statistics"""
    tasks = session.exec(
        select(TaskLog)
        .where(TaskLog.user_id == user_id)
        .order_by(TaskLog.created_at.desc())
        .limit(10)
    ).all()

    all_time_tokens = sum(
        [
            t.total_tokens
            for t in session.exec(
                select(TaskLog).where(TaskLog.user_id == user_id)
            ).all()
        ]
    )

    recent_list = [
        {
            "id": t.task_id,
            "status": t.status,
            "tokens": t.total_tokens,
            "ai_model": t.ai_model,
            "time": t.created_at.isoformat(),
        }
        for t in tasks
    ]

    return APIResponse(
        success=True,
        message="Statistics retrieved successfully",
        data={
            "total_tasks": len(recent_list),
            "total_tokens_used": all_time_tokens,
            "recent_tasks": recent_list,
        },
    )
