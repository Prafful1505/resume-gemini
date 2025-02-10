import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")

# Define the function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    doc = fitz.open(temp_file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    
    return text

# Define the prompt template
template = """
You are an HR assistant. Evaluate the resume based on the job description.
Job Description: {job_description}
Resume: {resume}

Provide a score from 1 to 10 and a short reason for your score.
"""

prompt = PromptTemplate(input_variables=["job_description", "resume"], template=template)

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose=True,
                             temperature=0.0,
                             google_api_key=api_key)  # Use the API key

# Create the chain using RunnablePassthrough and RunnableSequence
chain = (
    {"job_description": RunnablePassthrough(), "resume": RunnablePassthrough()}
    | prompt
    | llm
)

# Function to evaluate resume using the new chain
def evaluate_resume(job_description, resume):
    result = chain.invoke({"job_description": job_description, "resume": resume})
    return result

# Frontend
# Streamlit UI
st.title("Resume Evaluation System")

st.header("Input Job Description & Resume")

# Upload resume file
resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

# Job description input
job_description = st.text_area("Job Description", height=200)

if resume_file and job_description:
    # Extract resume text from PDF
    resume_text = extract_text_from_pdf(resume_file)
    
    st.write("Evaluating the resume against the job description...")

    # Evaluate resume
    evaluation = evaluate_resume(job_description, resume_text)

    # Display result
    st.subheader("Evaluation Result")
    st.write(evaluation.content)
else:
    st.write("Please upload a resume and provide a job description to get started.")
