import streamlit as st
from logic import evaluate_resume
import fitz  # PyMuPDF
import docx2txt
import tempfile

# --- File Readers ---
def getTextFromPDF(file):
    text = ""
    # Reset buffer pointer in case it was read before
    file.seek(0)
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def getTextFromDOCX(file):
    # Save the uploaded file to a temporary path for docx2txt
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    return docx2txt.process(tmp_path)

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Relevance Checker", layout="wide")

st.title("📄 Automated Resume Relevance Check System")
st.write("Upload a Job Description and a Resume to evaluate relevance.")

# Layout: Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    jd_input = st.text_area("Paste JD here", height=250)

with col2:
    st.subheader("Upload Resume")
    resume_file = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"])

# Submit Button
if st.button("Evaluate"):
    if jd_input and resume_file:
        # Extract resume text
        if resume_file.name.endswith(".pdf"):
            resume_text = getTextFromPDF(resume_file)
        else:
            resume_text = getTextFromDOCX(resume_file)

        # Call backend logic
        with st.spinner("Evaluating resume..."):
            result = evaluate_resume(jd_input, resume_text)

        # Display Results
        st.success("✅ Evaluation Complete!")
        st.metric("Final Relevance Score", f"{result['final_score']}%")
        st.write(f"**Verdict:** {result['verdict']}")

        st.subheader("📌 Skill Analysis")
        st.write(f"**Found Skills:** {', '.join(result['found_skills'])}")
        st.write(f"**Missing Skills:** {', '.join(result['missing_skills'])}")  

        st.subheader("💡 Feedback")
        st.info(result["feedback"])
    else:
        st.warning("⚠️ Please provide both Job Description and Resume file.")

