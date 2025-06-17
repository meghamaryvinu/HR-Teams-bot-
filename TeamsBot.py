import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from contextlib import asynccontextmanager

# Langchain imports
from langchain_openai import AzureChatOpenAI # Changed from langchain_groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load Environment Variables from .env file
load_dotenv()

# --- Azure OpenAI Environment Variables ---
# Ensure these are set in your .env file or as system environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01") # Default or specify your version
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") # Name of your deployed LLM model (e.g., "gpt-4-turbo")

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
    raise ValueError(
        "Azure OpenAI environment variables not set. Please set AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME in your .env file or environment."
    )

# Global Variables for LLM and Vector Store
llm = None
vectors = None

# Configuration Paths for Vector Store and Documents
FAISS_INDEX_PATH = "faiss_index_hr_chatbot"
HR_DOCS_FOLDER = 'policy_docs'
# --- Function to Get the HuggingFace Embeddings Model ---
def get_huggingface_embeddings_model_api():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_kwargs = {'device': 'cpu'} # Use 'cuda' if you have an NVIDIA GPU
    encode_kwargs = {'normalize_embeddings': True}

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"Embeddings model '{model_name}' loaded successfully for API.")
        return embeddings
    except Exception as e:
        print(f"Error loading HuggingFace Embeddings model for API: {e}")
        raise

# --- Function to Create or Load the Vector Store ---
def create_or_load_vector_store_api(index_path: str, docs_folder: str):
    global vectors

    embeddings = get_huggingface_embeddings_model_api()

    # Attempt to load existing vector store
    if os.path.exists(index_path):
        print(f"Attempting to load existing vector store from '{index_path}'...")
        try:
            # allow_dangerous_deserialization=True is needed for loading FAISS indexes
            vectors = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("Vector Store DB loaded successfully for API!")
            return vectors
        except Exception as e:
            print(f"Failed to load vector store from {index_path} for API: {e}")
            print("Attempting to recreate the vector store as loading failed or index is corrupt...")

    # If loading failed or index doesn't exist, create a new one
    print("Vector store not found or failed to load. Initializing and processing HR documents...")
    try:
        if not os.path.isdir(docs_folder):
            print(f"Error: Document folder '{docs_folder}' not found. Please ensure PDFs are deployed.")
            raise FileNotFoundError(f"HR Documents folder '{docs_folder}' not found.")

        loader = PyPDFDirectoryLoader(docs_folder)
        docs = loader.load()

        if not docs:
            print(f"No PDF documents found in the specified directory: {docs_folder}")
            raise ValueError("No PDF documents found to create vector store.")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        if not final_documents:
            print("No text chunks were generated from the documents after splitting.")
            raise ValueError("No text chunks generated from documents.")

        # Create new FAISS vector store and save it
        vectors = FAISS.from_documents(final_documents, embeddings)
        vectors.save_local(index_path)
        print("Documents processed and new Vector Store DB created and saved for API!")
        return vectors

    except Exception as e:
        print(f"An error occurred during API vector embedding setup: {e}")
        raise

# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes LLM and vector store before the app starts serving requests.
    This also handles any cleanup logic when the application is shutting down.
    """
    global llm, vectors
    print("FastAPI app starting up. Initializing resources...")
    try:
        # Initialize AzureChatOpenAI
        llm = AzureChatOpenAI(
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0.0, # Set to 0.0 for more deterministic answers in RAG
            max_retries=3 # Add retries for robustness
        )
        print("AzureChatOpenAI LLM initialized.")
    except Exception as e:
        print(f"Error initializing AzureChatOpenAI LLM: {e}")

    try:
        vectors = create_or_load_vector_store_api(FAISS_INDEX_PATH, HR_DOCS_FOLDER)
        print("Vector store initialized/loaded.")
    except Exception as e:
        print(f"Error initializing vector store: {e}")

    if llm and vectors:
        print("All critical resources initialized successfully.")
    else:
        # If either fails, the application might not function correctly
        print("WARNING: Some critical resources failed to initialize. Chatbot may not function correctly if dependencies are missing.")

    yield # The application will start serving requests now.

    # Code after 'yield' runs on shutdown (e.g., closing connections)
    print("FastAPI app shutting down. Performing cleanup (if any)...")


# --- Initialize FastAPI App ---
app = FastAPI(lifespan=lifespan, title="HR Chatbot with Azure OpenAI and Langchain")

# --- Request Body Model for Chat Endpoint ---
class ChatRequest(BaseModel):
    query: str

# --- API Endpoint for Chatbot Query ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a user query via a POST request and returns a chatbot response
    based on HR documents, leveraging Azure OpenAI and MultiQueryRetriever.
    """
    # Check if LLM and vector store are ready
    if not llm or not vectors:
        raise HTTPException(status_code=503, detail="Chatbot not ready. Resources are still loading or failed to initialize. Please try again in a moment.")

    # Define Main Prompt Template for RAG
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the given context.
        If the answer is not available in the provided context, politely state that you don't have enough information and cannot answer the question.
        Do not make up information.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )

    # Chain to combine documents with the prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # --- MultiQueryRetriever setup ---
    # This generates multiple queries from a single user input to improve retrieval
    query_gen_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. Provide these alternative questions separated by newlines, ensuring they are distinct and explore different facets of the original question."),
            ("human", "{question}"),
        ]
    )

    base_retriever = vectors.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 similar documents

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=query_gen_prompt,
    )

    # Combine retriever and document chain into a single retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    try:
        start_time = time.process_time()
        # Invoke the chain with the user's query
        response = retrieval_chain.invoke({'input': request.query})
        end_time = time.process_time()
        print(f"Response time: {end_time - start_time:.2f} seconds")

        return {
            "answer": response.get('answer', "I apologize, but I could not find a relevant answer in the provided HR documents."),
            "context": [doc.page_content for doc in response.get("context", [])],
            "metadata": [doc.metadata for doc in response.get("context", []) if doc.metadata],
            # MultiQueryRetriever sometimes returns generated_queries, sometimes not, depending on Langchain version and exact flow
            "generated_queries": response.get("generated_queries", "N/A")
        }
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while processing your query. Please try again. Error: {e}")

# --- For local development/testing ---
# This block ensures that Uvicorn runs the app directly when you execute this file.
@app.get("/")
async def root():
    return {"message": "✅ HR Chatbot is running. Use the /chat endpoint to ask questions."}

'''if __name__ == "__main__":
    import uvicorn
    # The --reload flag should ideally be used from the command line for better control.
    # When running programmatically, it can lead to issues with multiprocessing.
    # For simple local dev, it's fine, but for production, you'd run 'uvicorn PranabBot:app'
    uvicorn.run(app, host="0.0.0.0", port=8000)'''