# Medical Chatbot Flask Application
# This application creates a web-based medical chatbot using RAG (Retrieval-Augmented Generation)
# combining LangChain, OpenAI GPT-4, Pinecone vector database, and HuggingFace embeddings

from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Initialize Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# Set environment variables for LangChain components
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize HuggingFace embeddings model for converting text to vectors
# Uses 'sentence-transformers/all-MiniLM-L6-v2' which produces 384-dimensional embeddings
embeddings = download_hugging_face_embeddings()

# Name of the Pinecone index containing pre-processed medical documents
index_name = "medical-chatbot" 

# Connect to existing Pinecone vector store containing embedded medical document chunks
# This vector store was created by store_index.py and contains searchable medical knowledge
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Configure retriever to find the 3 most similar document chunks for each query
# Uses cosine similarity to match user questions with relevant medical content
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize OpenAI GPT-4o model for generating responses
chatModel = ChatOpenAI(model="gpt-4o")

# Create chat prompt template with system instructions and user input placeholder
# system_prompt contains medical-specific instructions imported from src.prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # Medical chatbot instructions and guidelines
        ("human", "{input}"),        # User's medical question
    ]
)

# Create document processing chain that combines retrieved docs with the LLM
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

# Create complete RAG chain: retrieve relevant docs → generate informed response
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# Route for the main chat interface
@app.route("/")
def index():
    """Serve the main chat webpage with HTML interface"""
    return render_template('chat.html')

# API endpoint for processing chat messages
@app.route("/get", methods=["GET", "POST"])
def chat():
    """
    Handle user messages and return AI-generated medical responses
    
    Process flow:
    1. Extract user message from form data
    2. Use RAG chain to retrieve relevant medical documents
    3. Generate contextual response using GPT-4o
    4. Return the answer to the frontend
    """
    msg = request.form["msg"]  # Get user's medical question
    input = msg
    print(f"User Question: {input}")  # Log the question for debugging
    
    # Invoke RAG chain: retrieve docs + generate response
    response = rag_chain.invoke({"input": msg})
    
    print(f"AI Response: {response['answer']}")  # Log the response
    return str(response["answer"])  # Return response to frontend

# Start the Flask development server
if __name__ == '__main__':
    # Run on all interfaces (0.0.0.0) on port 8080 with debug mode enabled
    app.run(host="0.0.0.0", port= 8080, debug= True)
