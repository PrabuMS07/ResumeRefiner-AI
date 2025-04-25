import streamlit as st
import pdfplumber
import spacy
import ollama
import mammoth  # Use mammoth for DOCX
from sentence_transformers import SentenceTransformer, util
import os
import time

# --- Configuration Constants ---
OLLAMA_MODEL = "mistral"  # Model to use with Ollama
NLP_MODEL = "en_core_web_sm"
SENTENCE_MODEL = "all-MiniLM-L6-v2"
# NOTE: This is a basic list. Consider expanding or using dynamic extraction.
SKILLS_LIST = [
    "Python", "Java", "C++", "JavaScript", "SQL", "NoSQL", "React", "Angular", "Vue",
    "Node.js", "Django", "Flask", "AWS", "Azure", "GCP", "Docker", "Kubernetes",
    "Machine Learning", "Deep Learning", "Data Science", "Data Analysis", "Pandas",
    "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "NLP", "Computer Vision",
    "Project Management", "Agile", "Scrum", "Communication", "Leadership", "Teamwork"
]

# --- Load Models ---
# Use caching to avoid reloading models on every interaction within a session
@st.cache_resource
def load_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(f"Spacy model '{model_name}' not found. Please download it: python -m spacy download {model_name}")
        return None

@st.cache_resource
def load_sentence_transformer(model_name):
    return SentenceTransformer(model_name)

nlp = load_spacy_model(NLP_MODEL)
st_model = load_sentence_transformer(SENTENCE_MODEL) # Renamed to avoid conflict

# --- Helper Functions ---

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        # pdfplumber works best with file paths or file-like objects
        with pdfplumber.open(uploaded_file) as pdf:
            full_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
            text = "\n".join(full_text)
        return text if text.strip() else "No extractable text found in the PDF."
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None # Return None on error

def extract_text_from_docx(uploaded_file):
    try:
        # Mammoth expects a file-like object, which uploaded_file is
        result = mammoth.extract_raw_text(uploaded_file)
        text = result.value.strip()
        return text if text else "No text found in the DOCX file."
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return None # Return None on error

def check_grammar(text):
    # NOTE: This is a very basic grammar check using SpaCy's POS tagging.
    # It's not comprehensive and might yield false positives/negatives.
    # Consider integrating a more robust tool like LanguageTool if needed.
    if not nlp:
        return ["SpaCy model not loaded. Grammar check skipped."]

    doc = nlp(text)
    mistakes = []
    # Example basic check (can be expanded)
    for token in doc:
        # Simple subject-verb agreement check (plural noun, singular verb)
        if token.dep_ == "nsubj" and token.tag_ in ["NNS", "NNPS"] and token.head.tag_ == "VBZ":
             mistakes.append(f"Potential Subject-Verb Agreement issue near '{token.head.text}' with subject '{token.text}'.")
        # Check for common typos or non-standard verbs (customize as needed)
        # if token.is_alpha and not token.is_stop and token.pos_ == "VERB" and not token.lemma_:
        #     mistakes.append(f"Possible non-standard verb or typo: '{token.text}'")

    return mistakes

def suggest_skills(resume_text, job_description):
    # NOTE: This relies on exact keyword matching from a predefined list.
    # It won't find variations or skills not in SKILLS_LIST.
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()
    missing_skills = [
        skill for skill in SKILLS_LIST
        if skill.lower() not in resume_lower and skill.lower() in jd_lower
    ]
    present_skills = [
        skill for skill in SKILLS_LIST
        if skill.lower() in resume_lower and skill.lower() in jd_lower
    ]
    return present_skills, missing_skills

def analyze_resume_with_ollama(resume_text, job_description):
    prompt = f"""
    You are an expert resume reviewer and career coach.
    Analyze the following resume in the context of the provided job description.
    Provide constructive, actionable feedback to help the candidate improve their resume for this specific job application.

    Focus on:
    1.  **Relevance & Keyword Alignment:** Does the resume effectively highlight experience and skills mentioned in the job description? Are relevant keywords missing?
    2.  **Impact & Accomplishments:** Are accomplishments quantified where possible? Are action verbs used effectively? Does it show impact rather than just listing duties?
    3.  **Clarity & Conciseness:** Is the resume easy to read and understand? Is there unnecessary jargon or overly complex language? Is it well-organized?
    4.  **Formatting & Presentation:** (Mention general issues if obvious, like inconsistent formatting, but focus less on minor visual details unless they hinder readability).
    5.  **Overall Impression:** Does the resume present the candidate as a strong fit for the role described in the job description?

    **Resume Text:**
    ---
    {resume_text}
    ---

    **Job Description:**
    ---
    {job_description}
    ---

    **Provide your analysis below in clear sections (e.g., Relevance, Impact, Clarity, Overall):**
    """

    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}. Is Ollama running and the model '{OLLAMA_MODEL}' downloaded?")
        return "Could not get analysis from Ollama."

def calculate_similarity(text1, text2):
    # NOTE: Calculates cosine similarity between the embeddings of the two texts.
    # Higher score indicates greater semantic similarity.
    if not st_model:
        return None
    try:
        embedding1 = st_model.encode(text1, convert_to_tensor=True)
        embedding2 = st_model.encode(text2, convert_to_tensor=True)
        similarity = util.cos_sim(embedding1, embedding2)
        return similarity.item() # Get the scalar value
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="AI Resume Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("📄 AI Resume Analyzer with Local LLM")
st.markdown("Upload your resume, paste the job description, and get AI-powered analysis and suggestions.")

# --- Sidebar ---
st.sidebar.header("📝 Inputs")
uploaded_file = st.sidebar.file_uploader("1. Upload Resume", type=["pdf", "docx"], help="Upload your resume in PDF or DOCX format.")
job_description = st.sidebar.text_area("2. Paste Job Description", height=200, placeholder="Paste the full job description here...", help="Provide the job description for comparative analysis.")

# --- Main Panel ---
resume_text = None # Initialize resume_text

if uploaded_file is not None:
    file_type = uploaded_file.type
    st.markdown("---")
    st.subheader("📑 Extracted Resume Text")

    with st.spinner(f"Extracting text from {uploaded_file.name}..."):
        if file_type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
             # Handle both .docx and potentially .doc if mammoth supports it well enough
            resume_text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload PDF or DOCX.")
            resume_text = None

    if resume_text:
        st.text_area("Resume Content (Editable)", resume_text, height=300, key="resume_content_area")
        # Allow editing extracted text
        resume_text = st.session_state.resume_content_area
    else:
        st.warning("Could not extract text from the uploaded file.")

# Only proceed to analysis if resume text is available and job description is provided
if resume_text and job_description and nlp and st_model: # Check if models loaded successfully
    st.markdown("---")
    st.subheader("📊 Resume Analysis Results")

    with st.spinner('Analyzing resume... This may take a moment.'):
        time.sleep(1) # Small delay for spinner visibility

        # --- Run Analysis Functions ---
        grammar_issues = check_grammar(resume_text)
        present_skills, missing_skills = suggest_skills(resume_text, job_description)
        similarity_score = calculate_similarity(resume_text, job_description)
        improvement_suggestions = analyze_resume_with_ollama(resume_text, job_description) # This might take longer

        # --- Display Results ---

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Semantic Similarity")
            if similarity_score is not None:
                st.metric(label="Resume vs. Job Description Similarity", value=f"{similarity_score:.2%}")
                st.progress(similarity_score)
                st.caption("Similarity score based on semantic meaning (higher is better alignment).")
            else:
                st.warning("Could not calculate similarity score.")

            st.markdown("#### 🔑 Skills Match")
            if present_skills:
                 st.success(f"**Skills Present:** {', '.join(present_skills)}")
            else:
                 st.info("No matching skills from the list found in both resume and job description.")

            if missing_skills:
                st.warning(f"**Potential Missing Skills:** {', '.join(missing_skills)}")
            else:
                st.success("No obvious missing skills (from the predefined list) detected for this job description. ✅")
            st.caption("Based on a predefined list and keyword matching.")


        with col2:
            st.markdown("#### ✓ Grammar & Basic Checks")
            if not grammar_issues:
                st.success("No major basic grammar issues detected. ✅")
            else:
                st.warning("Potential basic grammar issues found:")
                for issue in grammar_issues:
                    st.write(f"- {issue}")
            st.caption("Basic check, not comprehensive.")

        st.markdown("---")
        st.markdown("#### ✨ AI Improvement Suggestions (via Ollama)")
        with st.expander("Click to see detailed suggestions", expanded=True):
            st.markdown(improvement_suggestions) # Use markdown to render potential formatting from LLM

elif uploaded_file and not job_description:
    st.warning("Please paste the job description in the sidebar to enable analysis.")

# Add instructions if no file is uploaded
if not uploaded_file:
    st.info("Please upload your resume and paste a job description in the sidebar to begin analysis.")