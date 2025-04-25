
# 📄 Enhanced AI Resume Analyzer with Local LLM 🤖

This Streamlit application helps users analyze their resumes against job descriptions, providing **semantic similarity scores**, grammar checks, missing skill suggestions, and AI-powered improvement recommendations using local language models via Ollama.



## ✨ Features

*   **📄 Resume Upload:** Supports both PDF (`.pdf`) and DOCX (`.docx`) resume formats.
*   **📝 Text Extraction:** Automatically extracts text content using `pdfplumber` (PDF) and `python-mammoth` (DOCX).
*   **📋 Job Description Input:** Allows users to paste job descriptions for comparative analysis.
*   **🎯 Semantic Similarity:** Calculates and displays a cosine similarity score between the resume and job description using `Sentence Transformers`, indicating semantic alignment.
*   **🧐 Basic Grammar Check:** Identifies potential simple grammar issues using `spaCy` (Note: basic implementation).
*   **🔑 Skills Analysis:** Compares a predefined list of skills against the resume and job description, highlighting skills present in both and potentially missing skills relevant to the job.
*   **🤖 AI-Powered Feedback:** Leverages a locally running LLM (via Ollama, e.g., `mistral`) with a refined prompt to provide comprehensive suggestions on:
    *   Relevance and keyword alignment
    *   Impact and accomplishments
    *   Clarity, conciseness, and organization
    *   Overall fit for the role
*   **🔒 Local Processing:** Utilizes local models (`spaCy`, `Sentence Transformers`, `Ollama`) for enhanced privacy and offline capability (once models are downloaded).
*   **💻 Improved Web Interface:** Built with `Streamlit`, featuring spinners during processing and expandable sections for results.

## ⚙️ How It Works

1.  **Upload & Input:** User uploads a resume (PDF/DOCX) and pastes a job description.
2.  **Text Extraction:** Extracts plain text from the uploaded file. The extracted text is displayed and editable in the UI.
3.  **Analysis Pipeline:**
    *   **Similarity:** Calculates the semantic similarity score between the resume and job description using `Sentence Transformers`.
    *   **Grammar:** Performs basic grammar checks using `spaCy`.
    *   **Skills:** Compares resume/JD content against a predefined skills list.
    *   **LLM Suggestions:** Sends the resume, job description, and a structured prompt to a local Ollama LLM for detailed improvement feedback.
4.  **Display Results:** Presents the extracted text, similarity score, skills match, grammar notes, and AI suggestions in an organized Streamlit interface with progress indicators.

## 🚀 Technologies Used

*   **Frontend:** Streamlit
*   **Backend/Logic:** Python 3.x
*   **PDF Extraction:** pdfplumber
*   **DOCX Extraction:** python-mammoth
*   **NLP (Basic Grammar):** spaCy (`en_core_web_sm`)
*   **Embeddings/Similarity:** Sentence Transformers (`all-MiniLM-L6-v2`)
*   **Local LLM Interface:** Ollama Python Library (`ollama`)
*   **Local LLM Backend:** Ollama (requires separate installation and a running model like `mistral`)

## 🛠️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install Ollama:**
    *   Download and install Ollama from [ollama.ai](https://ollama.ai/).
    *   Ensure the Ollama application/service is running in the background.
3.  **Pull LLM Model:** Open your terminal and pull the model used in the script (default is `mistral`):
    ```bash
    ollama pull mistral
    ```
    *(Modify the `OLLAMA_MODEL` constant in the script if using a different model)*
4.  **Set up Python Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
5.  **Install Python Dependencies:** Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    pdfplumber
    spacy
    ollama
    sentence-transformers
    python-mammoth
    torch # Sentence Transformers often needs PyTorch
    torchvision # Might be needed depending on torch installation
    torchaudio # Might be needed depending on torch installation
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Depending on your system, installing PyTorch might have specific instructions. Refer to the [PyTorch website](https://pytorch.org/) if needed.)*
6.  **Download spaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```



## 💡 Future Improvements

*   **Advanced Grammar Checking:** Integrate more robust tools (e.g., LanguageTool-python).
*   **Dynamic Skill Extraction:** Use NLP/NER to identify skills directly from the text instead of a fixed list.
*   **Refined Similarity Metrics:** Explore more sophisticated methods or weight different sections for similarity calculation.
*   **Interactive Feedback:** Allow users to click on suggestions to see related parts of the resume/JD.
*   **User Configuration:** Allow selection of Ollama models or customization of the skills list.
*   **Improved UI/UX:** Enhance the display of results, potentially highlighting keywords or sections.
*   **Error Handling:** Add more granular error handling for model loading and API calls.

