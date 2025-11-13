# Medical Document Processing Helper Functions
# This module contains utilities for loading, processing, and embedding medical PDF documents
# Used to prepare medical knowledge base for the chatbot's vector search capabilities

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


def load_pdf_file(data):
    """
    Load and extract text content from all PDF files in a directory.
    
    Args:
        data (str): Path to directory containing PDF files
        
    Returns:
        List[Document]: List of Document objects with extracted text and metadata
        
    This function is essential for building the medical knowledge base by:
    - Scanning the specified directory for PDF files
    - Using PyPDFLoader to extract text from each PDF
    - Preserving document metadata (filename, page numbers, etc.)
    """
    # Create directory loader that finds all PDF files
    loader = DirectoryLoader(
        data,
        glob="*.pdf",          # Only load PDF files
        loader_cls=PyPDFLoader  # Use PDF-specific loader
    )
    
    # Load all documents and extract their text content
    documents = loader.load()
    
    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Clean document metadata to reduce storage overhead and improve processing efficiency.
    
    Args:
        docs (List[Document]): Original documents with full metadata
        
    Returns:
        List[Document]: Documents with minimal metadata (only source filename)
        
    This optimization:
    - Reduces vector database storage requirements
    - Keeps essential source tracking for citation purposes
    - Removes unnecessary metadata that could interfere with embeddings
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        # Extract only the source filename for reference
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,  # Keep full text content
                metadata={"source": src}         # Keep only source reference
            )
        )
    return minimal_docs


def text_split(extracted_data):
    """
    Split large documents into smaller, manageable chunks for vector embedding.
    
    Args:
        extracted_data (List[Document]): Documents to be split
        
    Returns:
        List[Document]: Smaller text chunks suitable for embedding
        
    Chunk parameters optimized for medical content:
    - chunk_size=500: Large enough to contain complete medical concepts
    - chunk_overlap=20: Ensures continuity across chunk boundaries
    - RecursiveCharacterTextSplitter: Respects paragraph and sentence boundaries
    """
    # Configure text splitter with medical-optimized parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # 500 characters per chunk (good for medical concepts)
        chunk_overlap=20   # 20 character overlap to maintain context
    )
    
    # Split documents while preserving structure
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks



def download_hugging_face_embeddings():
    """
    Initialize and return HuggingFace embeddings model for converting text to vectors.
    
    Returns:
        HuggingFaceEmbeddings: Configured embeddings model
        
    Model Selection Rationale:
    - 'sentence-transformers/all-MiniLM-L6-v2' chosen for:
      * High performance on semantic similarity tasks
      * 384-dimensional output (good balance of quality vs. speed)
      * Optimized for general domain text (works well with medical content)
      * Relatively small size (80MB) for faster loading
      * Strong performance on sentence-level semantic understanding
      
    This model converts text chunks into numerical vectors that capture
    semantic meaning, enabling similarity search in the Pinecone vector database.
    """
    # Initialize HuggingFace embeddings with optimized model
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'  # 384-dimensional embeddings
    )
    return embeddings