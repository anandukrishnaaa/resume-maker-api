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
        extra = "allow"


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
    start_date: str
    end_date: str
    highlights: List[str] = Field(default_factory=list)
    location: Optional[str] = None


class ExperienceEntry(BaseModel):
    """Experience entry matching RenderCV format"""

    company: str
    position: str
    start_date: str
    end_date: str
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
    """The main sections of the resume generated by AI (Job-tailored)"""

    summary: List[str] = Field(
        description="A brief professional summary as a list of 2-4 strings"
    )
    education: List[EducationEntry]
    experience: List[ExperienceEntry]
    projects: Optional[List[ProjectEntry]] = None
    skills: List[SkillEntry]


class GeneralResumeSections(BaseModel):
    """Resume sections for general-purpose resume (not job-specific)"""

    summary: List[str] = Field(
        description="A brief professional summary as a list of 2-4 strings"
    )
    education: List[EducationEntry]
    experience: List[ExperienceEntry]
    projects: Optional[List[ProjectEntry]] = None
    skills: List[SkillEntry]


class ProfileAnalysis(BaseModel):
    """Analysis and feedback about user's raw profile"""

    overall_score: int = Field(
        description="Overall profile strength score out of 100", ge=0, le=100
    )
    strengths: List[str] = Field(description="List of 3-5 key strengths in the profile")
    areas_for_improvement: List[str] = Field(
        description="List of 3-5 specific areas that need improvement"
    )
    actionable_suggestions: List[str] = Field(
        description="List of 5-7 concrete, actionable steps to improve the profile"
    )
    missing_elements: List[str] = Field(
        description="Important elements missing from the profile (e.g., certifications, projects, metrics)"
    )
    keyword_optimization: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Suggested keywords to add, organized by category",
    )
    summary_feedback: str = Field(
        description="2-3 paragraph summary with personalized advice"
    )


class ResumeAnalysis(BaseModel):
    """Analysis and feedback about a generated resume"""

    overall_score: int = Field(
        description="Overall resume quality score out of 100", ge=0, le=100
    )
    ats_compatibility_score: int = Field(
        description="ATS (Applicant Tracking System) compatibility score (0-100)",
        ge=0,
        le=100,
    )
    strengths: List[str] = Field(description="List of 3-5 key strengths in the resume")
    areas_for_improvement: List[str] = Field(
        description="List of 3-5 specific areas that need improvement"
    )
    actionable_suggestions: List[str] = Field(
        description="List of 5-7 concrete, actionable steps to improve the resume"
    )
    formatting_issues: List[str] = Field(
        default_factory=list, description="List of formatting or structural issues"
    )
    keyword_optimization: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Suggested keywords to add, organized by category",
    )
    job_alignment_score: Optional[int] = Field(
        default=None,
        description="How well the resume aligns with target job (0-100, only for job-tailored resumes)",
        ge=0,
        le=100,
    )
    summary_feedback: str = Field(
        description="2-3 paragraph summary with personalized advice"
    )


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
    profile_data: str
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
    social_networks: str
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
    job_id: Optional[int] = SQLField(default=None, foreign_key="jobdescription.id")
    content: str
    design_config: Optional[str] = None
    locale_config: Optional[str] = None
    rendercv_settings: Optional[str] = None
    ai_model: Optional[str] = None
    improvement_remarks: Optional[str] = None
    created_at: datetime = SQLField(default_factory=datetime.now)


class ProfileAnalysisRecord(SQLModel, table=True):
    """Stores profile analysis results"""

    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: str = SQLField(foreign_key="userprofile.user_id", index=True)
    analysis_data: str
    ai_model: Optional[str] = None
    created_at: datetime = SQLField(default_factory=datetime.now)


class ResumeAnalysisRecord(SQLModel, table=True):
    """Stores resume analysis results"""

    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: str = SQLField(foreign_key="userprofile.user_id", index=True)
    task_id: str = SQLField(foreign_key="tasklog.task_id")
    analysis_data: str
    ai_model: Optional[str] = None
    created_at: datetime = SQLField(default_factory=datetime.now)


class TaskLog(SQLModel, table=True):
    """Tracks the lifecycle of a request via UUID"""

    task_id: str = SQLField(primary_key=True, index=True)
    user_id: str = SQLField(foreign_key="userprofile.user_id", index=True)
    job_id: Optional[int] = SQLField(default=None, foreign_key="jobdescription.id")
    task_type: str = SQLField(default="job_tailored")
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
    profile_data: Optional[ProfileDataInput] = Field(default=None)


class ResumeParseRequest(BaseModel):
    """Request to parse resume text"""

    resume_text: str = Field(
        ...,
        example="""John Doe
john.doe@example.com | +91-9876543210 | Mumbai, India | linkedin.com/in/johndoe

EXPERIENCE

Senior Software Engineer | TechCorp Inc. | Mumbai, India | June 2022 - Present
• Led development of microservices architecture serving 1M+ users
• Reduced API response time by 40% through optimization

EDUCATION

Bachelor of Technology in Computer Science | IIT Bombay | 2016 - 2020
• GPA: 3.8/4.0

SKILLS

Languages: Python, JavaScript, Go
Frameworks: FastAPI, Django, React
""",
        description="Complete resume text to parse",
    )
    overwrite_existing: bool = Field(
        default=False, description="If true, overwrites existing profile"
    )


class GeneralResumeRequest(BaseModel):
    """Request to generate a general-purpose resume"""

    enhancement_remarks: Optional[str] = Field(
        default=None,
        example="Make the resume more professional and emphasize leadership skills",
        description="Optional instructions to enhance the resume",
    )
    ai_model: Optional[str] = Field(
        default=None, example="google/gemini-2.5-flash-lite"
    )
    design_config: Optional[Dict[str, Any]] = Field(
        default=None, example={"theme": "moderncv"}
    )
    locale_config: Optional[Dict[str, Any]] = None
    rendercv_settings: Optional[Dict[str, Any]] = None


class JobResumeRequest(BaseModel):
    """Request to generate job-tailored resume"""

    title: str = Field(..., example="Senior Backend Engineer")
    company: str = Field(..., example="BigTech Corp")
    description: str = Field(
        ...,
        example="""We are looking for a Senior Backend Engineer.

Requirements:
- 3+ years Python experience
- FastAPI or Django
- Microservices architecture
- AWS or GCP

Responsibilities:
- Design scalable backend services
- Mentor junior developers""",
    )
    ai_model: Optional[str] = Field(default=None)
    design_config: Optional[Dict[str, Any]] = Field(default=None)
    locale_config: Optional[Dict[str, Any]] = Field(default=None)
    rendercv_settings: Optional[Dict[str, Any]] = Field(default=None)


class RegenerateResumeRequest(BaseModel):
    """Request to regenerate resume with new configs"""

    ai_model: Optional[str] = Field(default=None)
    design_config: Optional[Dict[str, Any]] = Field(default=None)
    locale_config: Optional[Dict[str, Any]] = None
    rendercv_settings: Optional[Dict[str, Any]] = None
    improvement_remarks: Optional[str] = Field(
        default=None,
        example="Make the summary more concise and add quantifiable metrics",
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


class AnalyzeProfileRequest(BaseModel):
    """Request to analyze user profile"""

    ai_model: Optional[str] = Field(
        default=None, example="google/gemini-2.5-flash-lite"
    )


class AnalyzeResumeRequest(BaseModel):
    """Request to analyze generated resume"""

    ai_model: Optional[str] = Field(
        default=None, example="google/gemini-2.5-flash-lite"
    )


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


def create_job_tailored_resume_generator_agent(model_id: Optional[str] = None) -> Agent:
    db = SqliteDb(db_file="tmp/agents.db", session_table="resume_generator_sessions")
    return Agent(
        name="Job-Tailored Resume Generator",
        model=get_openrouter_model(model_id),
        instructions=[
            "You are an expert resume writer specializing in tailoring resumes for specific job openings.",
            "You will be given a Job Description, an AI Analysis of that job, and a User Profile.",
            "Your task is to select and adapt the user's experience to fit the job requirements.",
            "Do NOT invent false information. Use the user's actual profile but rephrase highlights to match keywords.",
            "For dates, use YYYY-MM format or 'present'.",
            "",
            "IMPORTANT FORMATTING RULES:",
            "- For skills: label is the category, details is a comma-separated string",
            "- For projects: Only include if the user has relevant projects",
            "- For experience highlights: Use action verbs and quantify achievements",
            "- Keep summary to 2-4 concise bullet points that align with job requirements",
        ],
        output_schema=ResumeSections,
        db=db,
        markdown=False,
    )


def create_general_resume_generator_agent(model_id: Optional[str] = None) -> Agent:
    """Agent for generating general-purpose resumes"""
    db = SqliteDb(db_file="tmp/agents.db", session_table="general_resume_sessions")
    return Agent(
        name="General Resume Generator",
        model=get_openrouter_model(model_id),
        instructions=[
            "You are an expert resume writer.",
            "Create a polished, professional, general-purpose resume from the user's profile.",
            "Do NOT invent false information. Use only the user's actual profile data.",
            "For dates, use YYYY-MM format or 'present'.",
            "",
            "FORMATTING RULES:",
            "- For skills: label is the category, details is a comma-separated string",
            "- For projects: Only include if the user has relevant projects",
            "- For experience highlights: Use strong action verbs and quantify achievements",
            "- Keep summary to 2-4 concise bullet points",
        ],
        output_schema=GeneralResumeSections,
        db=db,
        markdown=False,
    )


def create_profile_analyzer_agent(model_id: Optional[str] = None) -> Agent:
    """Agent for analyzing user profiles"""
    db = SqliteDb(db_file="tmp/agents.db", session_table="profile_analyzer_sessions")
    return Agent(
        name="Profile Analyzer",
        model=get_openrouter_model(model_id),
        instructions=[
            "You are an expert career advisor analyzing a user's raw profile.",
            "Provide comprehensive feedback on their work history, skills, and experience.",
            "",
            "CRITICAL: You MUST include ALL required fields in your response.",
            "",
            "SCORING (0-100):",
            "- 90-100: Exceptional, ready for top-tier opportunities",
            "- 75-89: Strong, minor improvements needed",
            "- 60-74: Good foundation, several improvements recommended",
            "- 40-59: Needs significant improvement",
            "- 0-39: Major gaps",
            "",
            "STRENGTHS (3-5 items):",
            "- Highlight genuine strong points",
            "- Be specific (e.g., 'Strong quantifiable metrics')",
            "",
            "AREAS FOR IMPROVEMENT (3-5 items):",
            "- Be constructive and specific",
            "- Focus on high-impact improvements",
            "",
            "ACTIONABLE SUGGESTIONS (5-7 items):",
            "- Concrete, implementable steps",
            "- Prioritize by impact",
            "",
            "MISSING ELEMENTS:",
            "- Identify critical gaps (projects, certifications, metrics, etc.)",
            "",
            "KEYWORD OPTIMIZATION (REQUIRED):",
            "- MUST be a dictionary with string keys and list values",
            "- Example: {'technical_skills': ['Docker', 'AWS'], 'soft_skills': ['leadership']}",
            "- If no suggestions, return empty dict: {}",
            "- NEVER omit this field",
            "",
            "SUMMARY FEEDBACK (2-3 paragraphs):",
            "- Personalized, encouraging advice",
            "- End with motivation and next steps",
        ],
        output_schema=ProfileAnalysis,
        db=db,
        markdown=False,
    )


def create_resume_analyzer_agent(model_id: Optional[str] = None) -> Agent:
    """Agent for analyzing generated resumes"""
    db = SqliteDb(db_file="tmp/agents.db", session_table="resume_analyzer_sessions")
    return Agent(
        name="Resume Analyzer",
        model=get_openrouter_model(model_id),
        instructions=[
            "You are an expert resume consultant analyzing a generated resume.",
            "Evaluate structure, formatting, ATS compatibility, and effectiveness.",
            "",
            "CRITICAL: You MUST include ALL required fields in your response.",
            "",
            "OVERALL SCORE (0-100):",
            "- Overall resume quality",
            "",
            "ATS COMPATIBILITY SCORE (0-100):",
            "- Keyword density, format simplicity, section clarity",
            "- Consider: proper headers, no complex tables, keyword usage",
            "",
            "JOB ALIGNMENT SCORE (0-100, for job-tailored resumes only):",
            "- How well resume aligns with target job",
            "- Set to null for general resumes",
            "",
            "STRENGTHS (3-5 items):",
            "- What works well in this resume",
            "",
            "AREAS FOR IMPROVEMENT (3-5 items):",
            "- Specific weaknesses to address",
            "",
            "ACTIONABLE SUGGESTIONS (5-7 items):",
            "- Concrete steps to improve the resume",
            "",
            "FORMATTING ISSUES (list):",
            "- List structural or formatting problems",
            "- Examples: 'Summary too long', 'Missing dates'",
            "- If none, return empty list []",
            "",
            "KEYWORD OPTIMIZATION (REQUIRED):",
            "- MUST be a dictionary with string keys and list values",
            "- Example: {'technical_skills': ['Docker', 'AWS'], 'soft_skills': ['leadership']}",
            "- If no suggestions, return empty dict: {}",
            "- NEVER omit this field",
            "",
            "SUMMARY FEEDBACK (2-3 paragraphs):",
            "- Personalized advice",
            "- Clear next steps",
        ],
        output_schema=ResumeAnalysis,
        db=db,
        markdown=False,
    )


def create_resume_parser_agent(model_id: Optional[str] = None) -> Agent:
    """Agent that parses resume text into structured data"""
    db = SqliteDb(db_file="tmp/agents.db", session_table="resume_parser_sessions")
    return Agent(
        name="Resume Parser",
        model=get_openrouter_model(model_id),
        instructions=[
            "You are an expert resume parser.",
            "Extract structured information from resume text.",
            "",
            "FORMATTING RULES:",
            "1. Dates: Use YYYY-MM format. If only year, use YYYY-01 for start, YYYY-12 for end",
            "2. Currently employed: use 'present' as end_date",
            "3. Phone: digits only, no special characters",
            "4. Country code: Extract if present, otherwise +1",
            "5. Location: Include city, state/province, country",
            "6. Social networks: Extract LinkedIn, GitHub, etc.",
            "7. Skills: Categorize into languages, frameworks, tools, databases, soft_skills, certifications",
            "8. Be thorough - extract ALL information",
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
    ai_model: Optional[str] = None,
):
    """Background task: Runs AI analysis and generates job-tailored resume"""
    logger.info(f"--- Starting Job Analysis: {task_id} ---")

    with Session(engine) as session:
        task = session.get(TaskLog, task_id)
        if not task:
            return

        try:
            task.status = "analysis_in_progress"
            task.ai_model = ai_model
            task.logs.append(f"[{datetime.now()}] Analysis started")
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

            process_job_tailored_resume_task(task_id)

        except Exception as e:
            logger.error(f"Analysis Error: {e}", exc_info=True)
            task.status = "error"
            task.logs.append(f"[{datetime.now()}] Error: {str(e)}")
            session.add(task)
            session.commit()


@huey.task()
def process_job_tailored_resume_task(
    task_id: str,
    design_override: Optional[Dict] = None,
    locale_override: Optional[Dict] = None,
    rendercv_settings_override: Optional[Dict] = None,
    ai_model_override: Optional[str] = None,
    improvement_remarks: Optional[str] = None,
):
    """Background task: Generates job-tailored resume"""
    logger.info(f"--- Starting Job-Tailored Resume: {task_id} ---")

    with Session(engine) as session:
        task = session.get(TaskLog, task_id)
        if not task or not task.job_id:
            return

        try:
            task.status = "resume_in_progress"
            task.logs.append(f"[{datetime.now()}] Resume generation started")

            if ai_model_override:
                task.ai_model = ai_model_override

            session.add(task)
            session.commit()

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

            existing_resume = session.exec(
                select(ResumeContent)
                .where(ResumeContent.job_id == task.job_id)
                .where(ResumeContent.user_id == task.user_id)
                .order_by(ResumeContent.created_at.desc())
            ).first()

            sections_dict = None
            tokens_used = 0

            if improvement_remarks or not existing_resume:
                task.logs.append(
                    f"[{datetime.now()}] Calling AI to generate/improve content"
                )

                generator = create_job_tailored_resume_generator_agent(
                    ai_model_override or task.ai_model
                )
                prompt = f"Generate Resume Content for:\nJOB: {jd.title}\nANALYSIS: {analysis.analysis_text}\nUSER: {user.profile_data}"

                if improvement_remarks and existing_resume:
                    prompt += f"\n\nPREVIOUS:\n{existing_resume.content}\n\nIMPROVEMENTS:\n{improvement_remarks}"

                response = generator.run(prompt)
                resume_sections_obj = response.content

                if response.metrics:
                    try:
                        tokens_used = response.metrics.to_dict().get("total_tokens", 0)
                    except:
                        tokens_used = getattr(response.metrics, "total_tokens", 0)

                task.total_tokens += int(tokens_used)
                task.logs.append(
                    f"[{datetime.now()}] Content generated. Tokens: {tokens_used}"
                )
                sections_dict = resume_sections_obj.model_dump(exclude_none=True)
            else:
                task.logs.append(
                    f"[{datetime.now()}] Reusing existing content (design-only change)"
                )
                sections_dict = json.loads(existing_resume.content)

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

            _render_pdf(
                task,
                sections_dict,
                personal,
                design_override,
                locale_override,
                rendercv_settings_override,
                session,
            )

        except Exception as e:
            logger.error(f"Resume Error: {e}", exc_info=True)
            task.status = "error"
            task.logs.append(f"[{datetime.now()}] Error: {str(e)}")
            session.add(task)
            session.commit()


@huey.task()
def process_general_resume_task(
    task_id: str,
    enhancement_remarks: Optional[str] = None,
    design_override: Optional[Dict] = None,
    locale_override: Optional[Dict] = None,
    rendercv_settings_override: Optional[Dict] = None,
    ai_model: Optional[str] = None,
):
    """Background task: Generates general-purpose resume"""
    logger.info(f"--- Starting General Resume: {task_id} ---")

    with Session(engine) as session:
        task = session.get(TaskLog, task_id)
        if not task:
            return

        try:
            task.status = "resume_in_progress"
            task.ai_model = ai_model
            task.logs.append(f"[{datetime.now()}] General resume generation started")
            session.add(task)
            session.commit()

            user, personal = get_or_create_user_profile(session, task.user_id)

            generator = create_general_resume_generator_agent(ai_model)
            prompt = f"Create a professional general-purpose resume:\n\nPROFILE:\n{user.profile_data}"

            if enhancement_remarks:
                prompt += f"\n\nENHANCEMENTS:\n{enhancement_remarks}"

            response = generator.run(prompt)
            resume_sections_obj: GeneralResumeSections = response.content

            tokens_used = 0
            if response.metrics:
                try:
                    tokens_used = response.metrics.to_dict().get("total_tokens", 0)
                except:
                    tokens_used = getattr(response.metrics, "total_tokens", 0)

            task.total_tokens += int(tokens_used)
            task.logs.append(
                f"[{datetime.now()}] Content generated. Tokens: {tokens_used}"
            )

            sections_dict = resume_sections_obj.model_dump(exclude_none=True)

            resume_content = ResumeContent(
                user_id=task.user_id,
                task_id=task_id,
                job_id=None,
                content=json.dumps(sections_dict),
                design_config=json.dumps(design_override) if design_override else None,
                locale_config=json.dumps(locale_override) if locale_override else None,
                rendercv_settings=json.dumps(rendercv_settings_override)
                if rendercv_settings_override
                else None,
                ai_model=ai_model,
                improvement_remarks=enhancement_remarks,
            )
            session.add(resume_content)
            session.commit()

            _render_pdf(
                task,
                sections_dict,
                personal,
                design_override,
                locale_override,
                rendercv_settings_override,
                session,
            )

        except Exception as e:
            logger.error(f"General Resume Error: {e}", exc_info=True)
            task.status = "error"
            task.logs.append(f"[{datetime.now()}] Error: {str(e)}")
            session.add(task)
            session.commit()


def _render_pdf(
    task,
    sections_dict,
    personal,
    design_override,
    locale_override,
    rendercv_settings_override,
    session,
):
    """Helper function to render PDF from resume sections"""
    if "projects" in sections_dict and not sections_dict["projects"]:
        del sections_dict["projects"]

    if "skills" in sections_dict:
        skills_formatted = []
        for skill in sections_dict["skills"]:
            if "label" in skill and "details" in skill:
                skills_formatted.append(skill)
        sections_dict["skills"] = skills_formatted

    full_phone = personal.country_code + personal.phone.replace("-", "").replace(
        " ", ""
    )

    full_cv_dict = {
        "cv": {
            "name": personal.name,
            "location": personal.location,
            "email": personal.email,
            "phone": full_phone,
            "sections": sections_dict,
        }
    }

    if design_override:
        full_cv_dict["design"] = design_override
    if locale_override:
        full_cv_dict["locale"] = locale_override
    if rendercv_settings_override:
        full_cv_dict["rendercv_settings"] = rendercv_settings_override

    social_networks = json.loads(personal.social_networks)
    if social_networks:
        full_cv_dict["cv"]["social_networks"] = social_networks

    output_dir = Path("output") / task.task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    task.output_folder = str(output_dir)

    cv_data_model = RenderCVDataModel(**full_cv_dict)

    task.logs.append(f"[{datetime.now()}] Generating Typst file...")
    session.add(task)
    session.commit()

    typst_file_path = renderer.create_a_typst_file_and_copy_theme_files(
        cv_data_model, output_dir
    )

    task.logs.append(f"[{datetime.now()}] Rendering PDF...")
    session.add(task)
    session.commit()

    pdf_path = renderer.render_a_pdf_from_typst(typst_file_path)

    task.pdf_path = str(pdf_path)
    task.status = "resume_complete"
    task.logs.append(f"[{datetime.now()}] PDF rendered at {pdf_path}")
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

    if request.profile_data:
        profile_data_json = json.dumps(
            request.profile_data.model_dump(exclude_none=True)
        )
    else:
        default_profile = {"education": [], "experience": [], "skills": {}}
        profile_data_json = json.dumps(default_profile)

    user = UserProfile(user_id=user_id, profile_data=profile_data_json)
    session.add(user)
    session.commit()
    session.refresh(user)

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
    """Parse resume text using AI and automatically create user profile"""
    try:
        existing_user = session.exec(
            select(UserProfile).where(UserProfile.user_id == user_id)
        ).first()

        if existing_user and not request.overwrite_existing:
            return APIResponse(
                success=False,
                message="User already exists. Set overwrite_existing=true to update.",
                error="USER_EXISTS",
            )

        logger.info(f"Parsing resume for user: {user_id}")
        parser = create_resume_parser_agent()

        response = parser.run(f"Parse the following resume:\n\n{request.resume_text}")

        extracted: ExtractedProfile = response.content

        tokens_used = 0
        if response.metrics:
            try:
                tokens_used = response.metrics.to_dict().get("total_tokens", 0)
            except:
                tokens_used = getattr(response.metrics, "total_tokens", 0)

        logger.info(f"Resume parsed. Tokens: {tokens_used}")

        profile_data = {
            "education": [
                edu.model_dump(exclude_none=True) for edu in extracted.education
            ],
            "experience": [
                exp.model_dump(exclude_none=True) for exp in extracted.experience
            ],
            "skills": extracted.skills.model_dump(exclude_none=True),
        }

        if existing_user:
            existing_user.profile_data = json.dumps(profile_data)
            existing_user.updated_at = datetime.now()
            session.add(existing_user)
        else:
            user = UserProfile(user_id=user_id, profile_data=json.dumps(profile_data))
            session.add(user)

        session.commit()

        existing_personal = session.exec(
            select(PersonalInfo).where(PersonalInfo.user_id == user_id)
        ).first()

        contact = extracted.contact_info
        social_networks_json = json.dumps(contact.social_networks or [])

        if existing_personal:
            existing_personal.name = contact.name
            existing_personal.email = contact.email
            existing_personal.phone = contact.phone
            existing_personal.country_code = contact.country_code
            existing_personal.location = contact.location
            existing_personal.social_networks = social_networks_json
            existing_personal.updated_at = datetime.now()
            session.add(existing_personal)
        else:
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
    """Get complete user profile"""
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
# API Endpoints - Profile Analysis
# ============================================================================
@app.post(
    "/profile/analyze",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def analyze_profile(
    request: AnalyzeProfileRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """
    Analyze user's raw profile and get actionable feedback.

    Returns comprehensive analysis including:
    - Overall profile strength score (0-100)
    - Key strengths
    - Areas for improvement
    - Actionable suggestions
    - Missing elements
    - Keyword optimization tips
    - Detailed summary feedback
    """
    try:
        user, personal = get_or_create_user_profile(session, user_id)

        logger.info(f"Analyzing profile for user: {user_id}")

        analyzer = create_profile_analyzer_agent(request.ai_model)

        profile_summary = f"""
USER PROFILE:
{user.profile_data}

PERSONAL INFO:
Name: {personal.name}
Email: {personal.email}
Location: {personal.location}
Social Networks: {personal.social_networks}
"""

        response = analyzer.run(
            f"Analyze this profile and provide comprehensive feedback:\n\n{profile_summary}"
        )

        # Check if response.content is a valid ProfileAnalysis object
        if not isinstance(response.content, ProfileAnalysis):
            logger.error(
                f"AI agent returned invalid response type: {type(response.content)}"
            )
            return APIResponse(
                success=False,
                message="Failed to analyze profile - AI response validation failed",
                error="AI_VALIDATION_ERROR",
            )

        analysis: ProfileAnalysis = response.content

        tokens_used = 0
        if response.metrics:
            try:
                tokens_used = response.metrics.to_dict().get("total_tokens", 0)
            except:
                tokens_used = getattr(response.metrics, "total_tokens", 0)

        logger.info(f"Profile analyzed. Tokens: {tokens_used}")

        analysis_record = ProfileAnalysisRecord(
            user_id=user_id,
            analysis_data=json.dumps(analysis.model_dump()),
            ai_model=request.ai_model,
        )
        session.add(analysis_record)
        session.commit()
        session.refresh(analysis_record)

        return APIResponse(
            success=True,
            message="Profile analysis complete",
            data={
                "analysis_id": analysis_record.id,
                "overall_score": analysis.overall_score,
                "strengths": analysis.strengths,
                "areas_for_improvement": analysis.areas_for_improvement,
                "actionable_suggestions": analysis.actionable_suggestions,
                "missing_elements": analysis.missing_elements,
                "keyword_optimization": analysis.keyword_optimization,
                "summary_feedback": analysis.summary_feedback,
                "tokens_used": tokens_used,
                "ai_model": request.ai_model or "default",
                "analyzed_at": analysis_record.created_at.isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Error analyzing profile: {e}", exc_info=True)
        return APIResponse(
            success=False,
            message=f"Failed to analyze profile: {str(e)}",
            error="ANALYSIS_ERROR",
        )


@app.get(
    "/profile/analyze/history",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def get_profile_analysis_history(
    user_id: str = Depends(get_user_id), session: Session = Depends(get_session)
):
    """Get history of profile analyses for the user"""
    analyses = session.exec(
        select(ProfileAnalysisRecord)
        .where(ProfileAnalysisRecord.user_id == user_id)
        .order_by(ProfileAnalysisRecord.created_at.desc())
    ).all()

    return APIResponse(
        success=True,
        message=f"Retrieved {len(analyses)} profile analyses",
        data={
            "count": len(analyses),
            "analyses": [
                {
                    "id": a.id,
                    "overall_score": json.loads(a.analysis_data).get("overall_score"),
                    "ai_model": a.ai_model,
                    "analyzed_at": a.created_at.isoformat(),
                }
                for a in analyses
            ],
        },
    )


@app.get(
    "/profile/analyze/{analysis_id}",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def get_profile_analysis_details(
    analysis_id: int,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Get detailed results of a specific profile analysis"""
    analysis_record = session.exec(
        select(ProfileAnalysisRecord)
        .where(ProfileAnalysisRecord.id == analysis_id)
        .where(ProfileAnalysisRecord.user_id == user_id)
    ).first()

    if not analysis_record:
        return APIResponse(
            success=False, message="Analysis not found", error="ANALYSIS_NOT_FOUND"
        )

    analysis_data = json.loads(analysis_record.analysis_data)

    return APIResponse(
        success=True,
        message="Profile analysis retrieved successfully",
        data={
            "analysis_id": analysis_record.id,
            **analysis_data,
            "ai_model": analysis_record.ai_model,
            "analyzed_at": analysis_record.created_at.isoformat(),
        },
    )


# ============================================================================
# API Endpoints - Resume Generation
# ============================================================================
@app.post(
    "/resumes/generate",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def generate_general_resume(
    request: GeneralResumeRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """
    Generate a general-purpose resume from user profile (no job description).

    Creates a polished, professional resume based on the user's profile.
    For feedback on your profile, use POST /profile/analyze instead.
    """
    task_id = str(uuid.uuid4())

    task = TaskLog(
        task_id=task_id,
        user_id=user_id,
        task_type="general",
        status="pending",
        logs=[f"[{datetime.now()}] General resume task created"],
        ai_model=request.ai_model,
    )
    session.add(task)
    session.commit()

    process_general_resume_task(
        task_id,
        enhancement_remarks=request.enhancement_remarks,
        design_override=request.design_config,
        locale_override=request.locale_config,
        rendercv_settings_override=request.rendercv_settings,
        ai_model=request.ai_model,
    )

    return APIResponse(
        success=True,
        message="General resume generation started",
        data={"task_id": task_id, "status": "resume_pending", "type": "general"},
    )


@app.post(
    "/resumes/job-tailored",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def generate_job_tailored_resume(
    request: JobResumeRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Generate a job-tailored resume optimized for a specific job description"""
    task_id = str(uuid.uuid4())
    content_hash = hashlib.sha256(
        request.description.strip().encode("utf-8")
    ).hexdigest()

    task = TaskLog(
        task_id=task_id,
        user_id=user_id,
        task_type="job_tailored",
        status="pending",
        logs=[f"[{datetime.now()}] Job-tailored resume task created"],
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
            task.logs.append(f"[{datetime.now()}] Using cached analysis")
            session.add(task)
            session.commit()

            process_job_tailored_resume_task(
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
                    "type": "job_tailored",
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
            ai_model=request.ai_model,
        )
        return APIResponse(
            success=True,
            message="JD found. Starting analysis and resume generation.",
            data={
                "task_id": task_id,
                "status": "analysis_pending",
                "type": "job_tailored",
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
        ai_model=request.ai_model,
    )

    return APIResponse(
        success=True,
        message="Job submitted. Starting analysis and resume generation.",
        data={"task_id": task_id, "status": "analysis_pending", "type": "job_tailored"},
    )


@app.post(
    "/resumes/{task_id}/analyze",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def analyze_resume(
    task_id: str,
    request: AnalyzeResumeRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """
    Analyze a generated resume and get actionable feedback.

    Returns comprehensive analysis including:
    - Overall resume quality score (0-100)
    - ATS compatibility score (0-100)
    - Job alignment score (for job-tailored resumes)
    - Strengths
    - Areas for improvement
    - Actionable suggestions
    - Formatting issues
    - Keyword optimization tips
    - Detailed summary feedback
    """
    try:
        # Verify task belongs to user
        task = session.get(TaskLog, task_id)
        if not task or task.user_id != user_id:
            return APIResponse(
                success=False,
                message="Task not found or does not belong to user",
                error="TASK_NOT_FOUND",
            )

        # Get resume content
        resume = session.exec(
            select(ResumeContent)
            .where(ResumeContent.task_id == task_id)
            .where(ResumeContent.user_id == user_id)
        ).first()

        if not resume:
            return APIResponse(
                success=False,
                message="Resume content not found for this task",
                error="RESUME_NOT_FOUND",
            )

        user, personal = get_or_create_user_profile(session, user_id)

        logger.info(f"Analyzing resume for task: {task_id}")

        resume_content = json.loads(resume.content)

        # Get job info if it's a job-tailored resume
        job_context = ""
        if task.job_id:
            job = session.get(JobDescription, task.job_id)
            if job:
                job_context = f"\n\nTARGET JOB:\nTitle: {job.title}\nCompany: {job.company}\n\nDescription:\n{job.description}"

        content_to_analyze = f"""
RESUME TYPE: {task.task_type}
{job_context}

PERSONAL INFO:
Name: {personal.name}
Email: {personal.email}
Location: {personal.location}

RESUME CONTENT:
{json.dumps(resume_content, indent=2)}
"""

        # Run AI analysis
        analyzer = create_resume_analyzer_agent(request.ai_model)

        response = analyzer.run(
            f"Analyze this generated resume:\n\n{content_to_analyze}"
        )

        # Check if response.content is a valid ResumeAnalysis object
        if not isinstance(response.content, ResumeAnalysis):
            logger.error(
                f"AI agent returned invalid response type: {type(response.content)}"
            )
            return APIResponse(
                success=False,
                message="Failed to analyze resume - AI response validation failed",
                error="AI_VALIDATION_ERROR",
            )

        analysis: ResumeAnalysis = response.content

        # Track tokens
        tokens_used = 0
        if response.metrics:
            try:
                tokens_used = response.metrics.to_dict().get("total_tokens", 0)
            except:
                tokens_used = getattr(response.metrics, "total_tokens", 0)

        logger.info(f"Resume analyzed. Tokens: {tokens_used}")

        # Store analysis
        analysis_record = ResumeAnalysisRecord(
            user_id=user_id,
            task_id=task_id,
            analysis_data=json.dumps(analysis.model_dump()),
            ai_model=request.ai_model,
        )
        session.add(analysis_record)
        session.commit()
        session.refresh(analysis_record)

        return APIResponse(
            success=True,
            message="Resume analysis complete",
            data={
                "analysis_id": analysis_record.id,
                "task_id": task_id,
                "resume_type": task.task_type,
                "overall_score": analysis.overall_score,
                "ats_compatibility_score": analysis.ats_compatibility_score,
                "job_alignment_score": analysis.job_alignment_score,
                "strengths": analysis.strengths,
                "areas_for_improvement": analysis.areas_for_improvement,
                "actionable_suggestions": analysis.actionable_suggestions,
                "formatting_issues": analysis.formatting_issues,
                "keyword_optimization": analysis.keyword_optimization,
                "summary_feedback": analysis.summary_feedback,
                "tokens_used": tokens_used,
                "ai_model": request.ai_model or "default",
                "analyzed_at": analysis_record.created_at.isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Error analyzing resume: {e}", exc_info=True)
        return APIResponse(
            success=False,
            message=f"Failed to analyze resume: {str(e)}",
            error="ANALYSIS_ERROR",
        )


@app.get(
    "/resumes/analyze/history",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def get_resume_analysis_history(
    user_id: str = Depends(get_user_id), session: Session = Depends(get_session)
):
    """Get history of all resume analyses for the user"""
    analyses = session.exec(
        select(ResumeAnalysisRecord)
        .where(ResumeAnalysisRecord.user_id == user_id)
        .order_by(ResumeAnalysisRecord.created_at.desc())
    ).all()

    return APIResponse(
        success=True,
        message=f"Retrieved {len(analyses)} resume analyses",
        data={
            "count": len(analyses),
            "analyses": [
                {
                    "id": a.id,
                    "task_id": a.task_id,
                    "overall_score": json.loads(a.analysis_data).get("overall_score"),
                    "ats_compatibility_score": json.loads(a.analysis_data).get(
                        "ats_compatibility_score"
                    ),
                    "ai_model": a.ai_model,
                    "analyzed_at": a.created_at.isoformat(),
                }
                for a in analyses
            ],
        },
    )


@app.get(
    "/resumes/analyze/{analysis_id}",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def get_resume_analysis_details(
    analysis_id: int,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Get detailed results of a specific resume analysis"""
    analysis_record = session.exec(
        select(ResumeAnalysisRecord)
        .where(ResumeAnalysisRecord.id == analysis_id)
        .where(ResumeAnalysisRecord.user_id == user_id)
    ).first()

    if not analysis_record:
        return APIResponse(
            success=False, message="Analysis not found", error="ANALYSIS_NOT_FOUND"
        )

    analysis_data = json.loads(analysis_record.analysis_data)

    return APIResponse(
        success=True,
        message="Resume analysis retrieved successfully",
        data={
            "analysis_id": analysis_record.id,
            "task_id": analysis_record.task_id,
            **analysis_data,
            "ai_model": analysis_record.ai_model,
            "analyzed_at": analysis_record.created_at.isoformat(),
        },
    )


# ============================================================================
# API Endpoints - Job Management (for reference)
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
                    "type": t.task_type,
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
            "type": task.task_type,
            "status": task.status,
            "logs": task.logs,
            "total_tokens": task.total_tokens,
            "pdf_url": pdf_url,
            "output_folder": task.output_folder,
            "ai_model": task.ai_model,
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
    """Get the AI-generated resume content"""
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
    "/tasks/{task_id}/regenerate",
    dependencies=[Depends(verify_api_key)],
    response_model=APIResponse,
)
def regenerate_resume(
    task_id: str,
    request: RegenerateResumeRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    """Regenerate resume from existing task with new configurations"""
    original_task = session.get(TaskLog, task_id)
    if not original_task or original_task.user_id != user_id:
        return APIResponse(
            success=False, message="Original task not found", error="TASK_NOT_FOUND"
        )

    new_task_id = str(uuid.uuid4())
    new_task = TaskLog(
        task_id=new_task_id,
        user_id=user_id,
        job_id=original_task.job_id,
        task_type=original_task.task_type,
        status="resume_pending",
        logs=[f"[{datetime.now()}] Regeneration from {task_id}"],
        ai_model=request.ai_model or original_task.ai_model,
    )
    session.add(new_task)
    session.commit()

    if original_task.task_type == "general":
        process_general_resume_task(
            new_task_id,
            enhancement_remarks=request.improvement_remarks,
            design_override=request.design_config,
            locale_override=request.locale_config,
            rendercv_settings_override=request.rendercv_settings,
            ai_model=request.ai_model,
        )
    else:
        process_job_tailored_resume_task(
            new_task_id,
            design_override=request.design_config,
            locale_override=request.locale_config,
            rendercv_settings_override=request.rendercv_settings,
            ai_model_override=request.ai_model,
            improvement_remarks=request.improvement_remarks,
        )

    return APIResponse(
        success=True,
        message="Resume regeneration started",
        data={
            "task_id": new_task_id,
            "original_task_id": task_id,
            "type": original_task.task_type,
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
            "type": t.task_type,
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
