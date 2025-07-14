from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def create_vector_store(pdf_path,embeddings):
    """
    Loads a PDF, splits it into chunks, and creates a FAISS vector store.
    
    Args:
        pdf_path (str): The path to the PDF file.
        embeddings: The embedding model to use.
        
    Returns:
        FAISS: The created vector store.
    """
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} text chunks.")

    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store created successfully.")

    return vectorstore
