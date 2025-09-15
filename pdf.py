import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM (text-only model)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ---- Functions ---- #
def extract_pdf_text(uploaded_file):
    """Extract text from all pages of a PDF."""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def get_gemini_response(prompt, resume_text, job_description):
    """Send resume + JD to Gemini model."""
    response = llm.invoke(
        [
            ("system", prompt),
            ("human", f"Job Description:\n{job_description}"),
            ("human", f"Resume:\n{resume_text}")
        ]
    )
    return response.content

# ---- Streamlit App ---- #
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Resume Analyzer")

# Job Description input
job_description = st.text_area("Paste Job Description:", key="jd")

# Resume upload
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
resume_text = ""
if uploaded_file is not None:
    st.success("PDF Uploaded Successfully")
    resume_text = extract_pdf_text(uploaded_file)

# Buttons
submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")

# Prompts
input_prompt1 = """
You are an experienced Technical HR Manager.
Review the provided resume against the job description.
Highlight the candidate's strengths and weaknesses.
"""

input_prompt3 = """
You are an ATS (Applicant Tracking System) scanner with deep knowledge of data science.
Evaluate the resume against the job description. 
Return:
1. Percentage Match
2. Missing Keywords
3. Final Thoughts
"""

# Actions
if submit1:
    if uploaded_file:
        response = get_gemini_response(input_prompt1, resume_text, job_description)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please upload a resume first!")

elif submit3:
    if uploaded_file:
        response = get_gemini_response(input_prompt3, resume_text, job_description)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please upload a resume first!")
