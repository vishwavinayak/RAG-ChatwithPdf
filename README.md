# RAG - Chat with PDF

A comprehensive PDF processing and question-answering system using RAG (Retrieval-Augmented Generation) with Groq LLM.

## Features

- **PDF Text Extraction**: Extract text from PDF documents
- **Vector Search**: Create embeddings and perform semantic search
- **AI-Powered Summarization**: Generate summaries using Groq LLM
- **Chat Interface**: Ask questions about PDF content
- **Streamlit Web Interface**: User-friendly web application

## Setup

### 1. Prerequisites

- Python 3.8+
- Groq API key (get one from [Groq Console](https://console.groq.com/))

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RAG-ChatwithPdf

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Command Line Interface

Run the main PDF summarization script:

```bash
python app.py
```

This will:
1. Load the sample PDF
2. Create vector embeddings
3. Generate a comprehensive summary

### Streamlit Web Interface

#### Advanced PDF Assistant
```bash
streamlit run streamlit.py
```

Features:
- Upload any PDF
- Chat with the document
- Get quick summaries
- Interactive interface

#### Simple PDF Summarizer
```bash
streamlit run pdf_summarizer_streamlit.py
```

Features:
- Basic PDF text extraction
- Simple summarization
- File statistics

## Project Structure

```
RAG/
├── app.py                      # Main CLI application
├── streamlit.py               # Advanced Streamlit interface
├── pdf_summarizer_streamlit.py # Simple Streamlit interface
├── llm_config.py              # LLM and embedding configuration
├── rag_pipeline.py            # RAG pipeline implementation
├── prompt.py                  # Prompt templates
├── requirements.txt           # Python dependencies
├── sample.pdf                 # Sample PDF for testing
├── context.txt               # Generated context (auto-created)
└── README.md                 # This file
```

## Dependencies

- `langchain`: Core RAG framework
- `langchain-groq`: Groq LLM integration
- `langchain-community`: Community integrations
- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Text embeddings
- `streamlit`: Web interface
- `PyPDF2`: PDF processing
- `python-dotenv`: Environment variable management
