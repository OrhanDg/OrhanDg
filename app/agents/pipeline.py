"""LangGraph pipeline orchestrating the CV tailoring and cover letter generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, TypedDict

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, START, StateGraph


@dataclass
class CandidateProfile:
    full_name: str
    contact: List[str]
    summary: str
    key_skills: List[str]
    achievements: List[str]
    experience: List[str]
    education: List[str]
    certifications: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)


@dataclass
class JobInsights:
    title: str
    company: str
    summary: str
    must_have: List[str]
    nice_to_have: List[str]
    keywords: List[str]


@dataclass
class TailoredResume:
    headline: str
    summary: str
    skills: List[str]
    experience: List[str]
    education: List[str]
    certifications: List[str]
    projects: List[str]


class TailoringState(TypedDict, total=False):
    resume_text: str
    job_description: str
    profile: CandidateProfile
    job: JobInsights
    tailored_resume: TailoredResume
    cover_letter: str


PROFILE_PROMPT = PromptTemplate.from_template(
    """You are an expert talent intelligence analyst.
Extract the candidate information from the resume text.
Return a JSON object with the fields and no additional commentary:
full_name, contact (list), summary, key_skills (list of up to 12 items), achievements (list),
experience (list of quantified bullet points), education (list), certifications (list), projects (list).

Resume:
{resume_text}
"""
)

JOB_PROMPT = PromptTemplate.from_template(
    """You are assisting with a job application.
Read the job description and produce a structured summary in JSON with the keys below. Respond with JSON only:
- title: Job title.
- company: Company name if present otherwise "".
- summary: A concise 3 sentence synopsis.
- must_have: list of core responsibilities or required skills.
- nice_to_have: list of secondary or preferred qualifications.
- keywords: list of keywords for ATS optimisation.

Job description:
{job_description}
"""
)

RESUME_PROMPT = PromptTemplate.from_template(
    """You are an ATS compliance specialist.
Leverage the candidate profile and job insights to tailor the resume.
Output JSON only with the keys: headline, summary, skills (list), experience (list of bullet strings),
education (list), certifications (list), projects (list).
Focus on quantifiable achievements, embed must-have keywords, and keep the tone professional.

Candidate profile:
{profile}

Job insights:
{job}
"""
)

COVER_LETTER_PROMPT = PromptTemplate.from_template(
    """You are a career coach drafting a bespoke cover letter.
Use the candidate profile, job insights, and tailored resume highlights.
Produce a polished cover letter in paragraphs.
Respond in plain text with multiple paragraphs separated by blank lines.
Address the hiring manager without a specific name, reference the company, and highlight
how the candidate meets the must-have skills.

Candidate profile:
{profile}

Job insights:
{job}

Tailored resume:
{tailored_resume}
"""
)


class TailoringPipeline:
    """Builds and executes the LangGraph pipeline for tailoring applications."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(TailoringState)

        workflow.add_node("profile", self._profile_node)
        workflow.add_node("job", self._job_node)
        workflow.add_node("tailor", self._tailor_node)
        workflow.add_node("cover", self._cover_letter_node)

        workflow.add_edge(START, "profile")
        workflow.add_edge("profile", "job")
        workflow.add_edge("job", "tailor")
        workflow.add_edge("tailor", "cover")
        workflow.add_edge("cover", END)

        return workflow.compile()

    def _profile_node(self, state: TailoringState) -> TailoringState:
        parser = JsonOutputParser()
        response = self.llm.invoke(
            PROFILE_PROMPT.format(resume_text=state["resume_text"])
        )
        payload = parser.parse(response.content)

        profile = CandidateProfile(
            full_name=payload.get("full_name", ""),
            contact=payload.get("contact", []) or [],
            summary=payload.get("summary", ""),
            key_skills=payload.get("key_skills", []) or [],
            achievements=payload.get("achievements", []) or [],
            experience=payload.get("experience", []) or [],
            education=payload.get("education", []) or [],
            certifications=payload.get("certifications", []) or [],
            projects=payload.get("projects", []) or [],
        )
        state["profile"] = profile
        return state

    def _job_node(self, state: TailoringState) -> TailoringState:
        parser = JsonOutputParser()
        response = self.llm.invoke(
            JOB_PROMPT.format(job_description=state["job_description"])
        )
        payload = parser.parse(response.content)
        job = JobInsights(
            title=payload.get("title", ""),
            company=payload.get("company", ""),
            summary=payload.get("summary", ""),
            must_have=payload.get("must_have", []) or [],
            nice_to_have=payload.get("nice_to_have", []) or [],
            keywords=payload.get("keywords", []) or [],
        )
        state["job"] = job
        return state

    def _tailor_node(self, state: TailoringState) -> TailoringState:
        parser = JsonOutputParser()
        response = self.llm.invoke(
            RESUME_PROMPT.format(
                profile=asdict(state["profile"]),
                job=asdict(state["job"]),
            )
        )
        payload = parser.parse(response.content)
        resume = TailoredResume(
            headline=payload.get("headline", ""),
            summary=payload.get("summary", ""),
            skills=payload.get("skills", []) or [],
            experience=payload.get("experience", []) or [],
            education=payload.get("education", []) or [],
            certifications=payload.get("certifications", []) or [],
            projects=payload.get("projects", []) or [],
        )
        state["tailored_resume"] = resume
        return state

    def _cover_letter_node(self, state: TailoringState) -> TailoringState:
        response = self.llm.invoke(
            COVER_LETTER_PROMPT.format(
                profile=asdict(state["profile"]),
                job=asdict(state["job"]),
                tailored_resume=asdict(state["tailored_resume"]),
            )
        )
        state["cover_letter"] = response.content.strip()
        return state

    def run(self, resume_text: str, job_description: str) -> TailoringState:
        initial_state: TailoringState = {
            "resume_text": resume_text,
            "job_description": job_description,
        }
        return self.graph.invoke(initial_state)


__all__ = [
    "CandidateProfile",
    "JobInsights",
    "TailoredResume",
    "TailoringPipeline",
]
