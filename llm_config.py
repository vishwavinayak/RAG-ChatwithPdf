import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings

# Load environment variables
load_dotenv()

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_groq_llm():
    """Initiaze and returns the groq llm"""
    return ChatGroq(
        temperature=0.1,
        model = "llama-3.3-70b-versatile",
        #model="groq-llama3-70b-chat",
        api_key=os.getenv("GROQ_API_KEY")
    )


def get_embedding_model():
    """Initialize and return embedding model"""
    try:
        # Use a lightweight embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        print(f"Warning: Could not load embedding model: {e}")
        return None