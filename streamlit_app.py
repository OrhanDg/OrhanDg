"""Streamlit app entry point for the agentic CV tailoring pipeline."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import List

import streamlit as st
from langchain_openai import ChatOpenAI

from app.agents.pipeline import TailoringPipeline
from app.utils.formatter import (
    ResumeSection,
    cover_letter_to_docx,
    plain_text_resume,
    resume_to_docx,
)
from app.utils.parser import extract_text_from_upload, normalise_job_description


st.set_page_config(page_title="Agentic CV Tailor", page_icon="🧠", layout="wide")

st.title("🧠 Agentic CV & Cover Letter Tailor")
st.caption("Upload your existing resume, paste a job description, and let the agentic pipeline craft ATS-friendly application materials.")


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Set it in your environment before running the app.")
        st.stop()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> TailoringPipeline:
    llm = get_llm()
    return TailoringPipeline(llm)


with st.sidebar:
    st.header("📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload a resume (PDF, DOCX, DOC, TXT)",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=False,
    )

job_description = st.text_area(
    "Job description",
    placeholder="Paste the job description here...",
    height=300,
)

if st.button("Generate tailored resume & cover letter", type="primary"):
    if not uploaded_file:
        st.warning("Please upload a resume before generating outputs.")
        st.stop()
    if not job_description.strip():
        st.warning("Please paste a job description before generating outputs.")
        st.stop()

    with st.spinner("Running agentic pipeline..."):
        resume_text = extract_text_from_upload(
            file_name=uploaded_file.name,
            file_bytes=uploaded_file.getvalue(),
            mime_type=uploaded_file.type,
        )
        job_text = normalise_job_description(job_description)

        pipeline = load_pipeline()
        result_state = pipeline.run(resume_text=resume_text, job_description=job_text)

    profile = result_state["profile"]
    job = result_state["job"]
    tailored_resume = result_state["tailored_resume"]
    cover_letter = result_state["cover_letter"]

    st.success("Application materials generated!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tailored Resume Preview")
        resume_sections: List[ResumeSection] = [
            ResumeSection("Professional Summary", [tailored_resume.summary]),
            ResumeSection("Core Skills", tailored_resume.skills),
            ResumeSection("Professional Experience", tailored_resume.experience),
            ResumeSection("Education", tailored_resume.education),
        ]
        if tailored_resume.certifications:
            resume_sections.append(ResumeSection("Certifications", tailored_resume.certifications))
        if tailored_resume.projects:
            resume_sections.append(ResumeSection("Projects", tailored_resume.projects))

        contact_block = profile.contact or ["Contact details not detected"]
        resume_text_preview = plain_text_resume(
            full_name=profile.full_name or "Candidate Name",
            contact_block=contact_block,
            sections=resume_sections,
        )
        st.text(resume_text_preview)

        resume_docx = resume_to_docx(
            full_name=profile.full_name or "Candidate Name",
            contact_block=contact_block,
            sections=resume_sections,
        )
        st.download_button(
            label="Download ATS Resume (DOCX)",
            data=resume_docx,
            file_name="tailored_resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    with col2:
        st.subheader("Cover Letter Preview")
        cover_letter_text = cover_letter.strip()
        st.text(cover_letter_text)

        cover_docx = cover_letter_to_docx(
            subject=f"Cover Letter - {job.title or 'Application'}",
            paragraphs=cover_letter_text.split("\n\n"),
            closing=profile.full_name or "Best regards",
        )
        st.download_button(
            label="Download Cover Letter (DOCX)",
            data=cover_docx,
            file_name="tailored_cover_letter.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    with st.expander("🔍 Pipeline Insights"):
        st.write("**Candidate Profile Extracted**")
        st.json(asdict(profile))
        st.write("**Job Understanding**")
        st.json(asdict(job))
        st.write("**Tailored Resume Structure**")
        st.json(asdict(tailored_resume))
        st.write("**Cover Letter**")
        st.write(cover_letter_text)
else:
    st.info("Upload a resume and paste a job description, then click the button to generate tailored materials.")
