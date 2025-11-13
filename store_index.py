# Medical Knowledge Base Vector Store Creation Script
# 
# This script processes medical PDF documents and creates a searchable vector database
# using Pinecone for the medical chatbot's retrieval-augmented generation (RAG) system.
#
# Workflow:
# 1. Load medical PDFs from data/ directory
# 2. Extract and clean text content
# 3. Split into optimal chunks for embedding
# 4. Generate vector embeddings using HuggingFace model
# 5. Store embeddings in Pinecone vector database
# 6. Enable semantic search for chatbot queries

from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

# Load environment variables for API access
load_dotenv()

# Retrieve API credentials from environment
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# Set environment variables for LangChain integration
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# STEP 1: Document Processing Pipeline
# Load all medical PDFs from the data directory
extracted_data = load_pdf_file(data='data/')
print(f"Loaded {len(extracted_data)} documents from PDF files")

# Clean metadata to optimize storage and processing
filter_data = filter_to_minimal_docs(extracted_data)
print(f"Filtered document metadata for {len(filter_data)} documents")

# Split documents into chunks optimized for embeddings (500 chars each)
text_chunks = text_split(filter_data)
print(f"Created {len(text_chunks)} text chunks for embedding")

# STEP 2: Initialize Embeddings Model
# Load HuggingFace model for converting text to 384-dimensional vectors
embeddings = download_hugging_face_embeddings()
print("Initialized HuggingFace embeddings model")

# STEP 3: Pinecone Vector Database Setup
# Initialize Pinecone client with API key
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

# Define index configuration
index_name = "medical-chatbot"  # Vector database name for medical knowledge

# Create Pinecone index if it doesn't exist
if not pc.has_index(index_name):
    print(f"Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,                    # Match HuggingFace model output
        metric="cosine",                 # Cosine similarity for semantic search
        spec=ServerlessSpec(
            cloud="aws",                 # Use AWS infrastructure
            region="us-east-1"           # East coast region for low latency
        ),
    )
else:
    print(f"Using existing Pinecone index: {index_name}")

# Get reference to the index
index = pc.Index(index_name)

# STEP 4: Generate Embeddings and Store in Vector Database
# Process all text chunks, generate embeddings, and store in Pinecone
print(f"Processing {len(text_chunks)} chunks and storing in vector database...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,    # Medical text chunks to embed
    index_name=index_name,    # Target Pinecone index
    embedding=embeddings,     # HuggingFace embeddings model
)
print("Successfully created medical knowledge vector database!")
print(f"Vector store ready for similarity search with {len(text_chunks)} embedded documents")