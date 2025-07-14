import streamlit as st
import PyPDF2
import io
from dotenv import load_dotenv
import os
from llm_config import get_groq_llm, get_embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# Simple vector store without FAISS
class SimpleVectorStore:
    def __init__(self, documents, embeddings_model=None):
        self.documents = documents
        self.chunks = [doc.page_content for doc in documents]
    
    def similarity_search(self, query, k=3):
        # Improved keyword matching with better scoring
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(chunk_words))
            
            # Calculate word frequency score
            frequency_score = sum(chunk_lower.count(word) for word in query_words)
            
            # Combined score
            score = overlap * 10 + frequency_score
            
            if score > 0:
                results.append((score, i, chunk))
        
        # Sort by relevance and return top k
        results.sort(reverse=True)
        return [Document(page_content=chunk) for _, _, chunk in results[:k]]

@st.cache_resource
def load_llm():
    """Load the Groq LLM"""
    try:
        return get_groq_llm()
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Debug: Print text length and preview
        print(f"Extracted text length: {len(text)} characters")
        print(f"Text preview: {text[:200]}...")
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def create_vector_store(text, embeddings=None):
    """Create vector store from text for fast retrieval"""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Debug: Print chunking info
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"Chunk {i+1} length: {len(chunk)} chars")
            print(f"Chunk {i+1} preview: {chunk[:100]}...")
        
        # Create documents
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Use simple vector store
        vector_store = SimpleVectorStore(documents)
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_relevant_chunks(query, vector_store, k=3):
    """Get relevant chunks for the query with debugging"""
    try:
        docs = vector_store.similarity_search(query, k=k)
        chunks = [doc.page_content for doc in docs]
        
        # Debug: Print what chunks we're getting
        print(f"Query: {query}")
        print(f"Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk[:100]}...")
        
        return chunks
    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {e}")
        return []

def quick_summarize(text, llm, vector_store=None):
    """Improved summarization using RAG-style context retrieval and a detailed prompt."""
    if not text or not llm:
        return "Unable to generate summary."
    try:
        # If a vector store is available, use it to get relevant chunks
        if vector_store is not None:
            # Use multiple queries to get diverse chunks
            queries = [
                "summary overview introduction",
                "main topics concepts",
                "key points conclusions"
            ]
            all_chunks = []
            for query in queries:
                chunks = get_relevant_chunks(query, vector_store, k=2)
                all_chunks.extend(chunks)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_chunks = []
            for chunk in all_chunks:
                if chunk not in seen:
                    seen.add(chunk)
                    unique_chunks.append(chunk)
            
            context = "\n\n".join(unique_chunks[:8])  # Limit to 8 chunks
        else:
            # Fallback: use the first 4000 characters
            context = text[:4000]
        
        # Debug: Show what context we're using
        print(f"Context length: {len(context)} characters")
        print(f"Context preview: {context[:200]}...")
        
        prompt = (
            "You are an expert summarizer. Create a comprehensive summary of the provided text.\n\n"
            "Guidelines:\n"
            "- Identify the main topics and key concepts\n"
            "- Include important definitions and explanations\n"
            "- Structure the summary with clear sections\n"
            "- Use bullet points for key points\n"
            "- Make it informative and educational\n\n"
            f"Text to summarize:\n{context}\n\n"
            "Summary:"
        )
        
        response = llm.invoke(prompt)
        
        # Debug: Print response details
        print(f"Response type: {type(response)}")
        print(f"Response has content attr: {hasattr(response, 'content')}")
        if hasattr(response, 'content'):
            print(f"Response content length: {len(response.content)}")
            print(f"Response content preview: {response.content[:200]}...")
        
        # Properly extract content from the response
        if hasattr(response, 'content'):
            return response.content.strip()
        elif isinstance(response, dict) and 'text' in response:
            return response['text'].strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            # Try to convert to string as fallback
            return str(response).strip()
    except Exception as e:
        return f"Error generating summary: {e}"

def chat_with_pdf(query, vector_store, llm, chat_history):
    """Chat with PDF using RAG"""
    try:
        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(query, vector_store)
        
        # Build context from relevant chunks
        context = "\n\n".join(relevant_chunks)
        
        prompt = f"""Based on the following document content, answer the user's question accurately and concisely.

Document Context:
{context}

Current Question: {query}

Answer:"""
        
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    st.title("📄 PDF AI Assistant")
    st.markdown("Upload a PDF and either get a quick summary or chat with it!")
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        st.stop()
    
    # Initialize LLM
    if st.session_state.llm is None:
        with st.spinner("Loading AI models..."):
            st.session_state.llm = load_llm()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    # Extract text
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    
                    if extracted_text and len(extracted_text.strip()) > 0:
                        st.session_state.pdf_text = extracted_text
                        
                        # Create vector store
                        vector_store = create_vector_store(extracted_text)
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.pdf_processed = True
                            st.success("🎉 PDF processed successfully!")
                        else:
                            st.error("❌ Failed to create vector store")
                    else:
                        st.error("❌ No text extracted from PDF")
    
    # Main content
    if st.session_state.pdf_processed:
        tab1, tab2 = st.tabs(["💬 Chat with PDF", "📝 Quick Summary"])
        
        with tab1:
            st.subheader("Chat with your PDF")
            
            # Display chat history
            for entry in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(entry['query'])
                with st.chat_message("assistant"):
                    st.write(entry['response'])
            
            # Chat input with unique key
            if query := st.chat_input("Ask a question about your PDF...", key="pdf_chat_input"):
                with st.chat_message("user"):
                    st.write(query)
                
                with st.chat_message("assistant"):
                    if st.session_state.llm and st.session_state.vector_store:
                        with st.spinner("Thinking..."):
                            response = chat_with_pdf(
                                query, 
                                st.session_state.vector_store, 
                                st.session_state.llm, 
                                st.session_state.chat_history
                            )
                            st.write(response)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'query': query,
                                'response': response
                            })
                    else:
                        st.error("LLM or vector store not available")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with tab2:
            st.subheader("Quick Summary")
            
            if st.button("⚡ Generate Quick Summary", type="primary"):
                if st.session_state.llm:
                    with st.spinner("Generating summary..."):
                        summary = quick_summarize(
                            st.session_state.pdf_text, 
                            st.session_state.llm, 
                            st.session_state.vector_store
                        )
                        st.write(summary)
                        
                        # Download summary
                        st.download_button(
                            label="💾 Download Summary",
                            data=summary,
                            file_name="quick_summary.txt",
                            mime="text/plain"
                        )
                else:
                    st.error("LLM not loaded")
    
    else:
        st.info("👆 Please upload and process a PDF file to get started!")

if __name__ == "__main__":
    main()
