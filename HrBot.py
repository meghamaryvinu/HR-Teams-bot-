import os
import tempfile
import shutil
import logging
import asyncio
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever

from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity
from starlette.responses import Response
from urllib.parse import urlparse
from starlette.middleware.cors import CORSMiddleware
import aiohttp  

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Config vars
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
MICROSOFT_APP_ID = os.getenv("MICROSOFT_APP_ID", "")
MICROSOFT_APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")

# Utility

def validate_env_vars():
    required_vars = {
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
        "AZURE_STORAGE_CONNECTION_STRING": AZURE_STORAGE_CONNECTION_STRING,
        "AZURE_STORAGE_CONTAINER_NAME": AZURE_STORAGE_CONTAINER_NAME,
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing env vars: {', '.join(missing_vars)}")

# Globals
llm = None
vectors = None
temp_docs_directory = None

# Embedder

def get_huggingface_embeddings_model():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    embeddings.embed_query("test")
    return embeddings

# Download PDFs

def download_pdfs_from_blob(connection_string, container_name, download_dir):
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    os.makedirs(download_dir, exist_ok=True)
    for blob in container_client.list_blobs():
        if blob.name.lower().endswith(".pdf"):
            blob_client = container_client.get_blob_client(blob.name)
            with open(os.path.join(download_dir, os.path.basename(blob.name)), "wb") as f:
                f.write(blob_client.download_blob().readall())

# FAISS operations

def upload_faiss_index_to_blob(local_dir, container_client):
    for filename in ["index.faiss", "index.pkl"]:
        path = os.path.join(local_dir, filename)
        if os.path.exists(path):
            container_client.get_blob_client(filename).upload_blob(open(path, "rb"), overwrite=True)

def download_faiss_index_from_blob(local_dir, container_client):
    found = True
    for filename in ["index.faiss", "index.pkl"]:
        try:
            blob = container_client.get_blob_client(filename)
            with open(os.path.join(local_dir, filename), "wb") as f:
                f.write(blob.download_blob().readall())
        except ResourceNotFoundError:
            found = False
            break
    return found

def create_vector_store_from_docs(doc_dir):
    all_docs = []
    for fname in os.listdir(doc_dir):
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(doc_dir, fname))
            all_docs.extend(loader.load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    embeddings = get_huggingface_embeddings_model()
    return FAISS.from_documents(chunks, embeddings)

# FastAPI lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, vectors, temp_docs_directory
    logger.info("🚀 Starting app lifespan setup")
    validate_env_vars()
    temp_docs_directory = tempfile.mkdtemp()
    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(AZURE_STORAGE_CONTAINER_NAME)

    if await asyncio.to_thread(download_faiss_index_from_blob, temp_docs_directory, container_client):
        embeddings_model = get_huggingface_embeddings_model()
        vectors = await asyncio.to_thread(FAISS.load_local, temp_docs_directory, embeddings_model, allow_dangerous_deserialization=True)
        logger.info("✅ Loaded FAISS index from blob")
    else:
        await asyncio.to_thread(download_pdfs_from_blob, AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME, temp_docs_directory)
        vectors = await asyncio.to_thread(create_vector_store_from_docs, temp_docs_directory)
        await asyncio.to_thread(vectors.save_local, temp_docs_directory)
        await asyncio.to_thread(upload_faiss_index_to_blob, temp_docs_directory, container_client)

    llm = AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=0.0,
        max_retries=3
    )
    logger.info("✅ LLM initialized")
    yield
    shutil.rmtree(temp_docs_directory)

# FastAPI app
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not llm or not vectors:
        raise HTTPException(status_code=503, detail="Resources not ready")

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context.
    If the answer is not in the context, say you don't have enough information.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    doc_chain = create_stuff_documents_chain(llm, prompt)

    query_prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate 5 diverse rephrasings of the user's query."),
        ("human", "{question}")
    ])

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectors.as_retriever(search_kwargs={"k": 5}),
        llm=llm,
        prompt=query_prompt
    )

    chain = create_retrieval_chain(retriever, doc_chain)
    response = chain.invoke({"input": request.query, "question": request.query})

    return {
        "answer": response.get("answer", "Sorry, no answer found."),
        "context": [doc.page_content for doc in response.get("context", [])],
        "metadata": [doc.metadata for doc in response.get("context", []) if doc.metadata]
    }

# Bot Framework adapter
settings = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

class HRBot:
    async def on_turn(self, turn_context: TurnContext):
        logger.info(f"🔁 Received activity of type: {turn_context.activity.type}")

        if turn_context.activity.type == "message":
            user_query = turn_context.activity.text
            logger.info(f"💬 User message: {user_query}")
            try:
                response = await chat_endpoint(ChatRequest(query=user_query))
                logger.info(f"🤖 Bot response: {response['answer']}")
                await turn_context.send_activity(response["answer"])
            except Exception as e:
                logger.exception("❌ Error while processing message")
                await turn_context.send_activity("Sorry, something went wrong.")

        elif turn_context.activity.type == "conversationUpdate":
            for member in turn_context.activity.members_added:
                if member.id != turn_context.activity.recipient.id:
                    await turn_context.send_activity(
                        "👋 Hi there! I'm **Dexter**, your AI-powered HR assistant. Ask me anything about HR policies, leave, or onboarding."
                    )

bot = HRBot()

'''@app.get("/test-credentials")
async def test_credentials():
    token_url = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": MICROSOFT_APP_ID,
        "client_secret": MICROSOFT_APP_PASSWORD,
        "scope": "https://api.botframework.com/.default"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(token_url, data=data) as resp:
            result = await resp.json()
            if resp.status == 200:
                return {
                    "status": "✅ Valid credentials",
                    "access_token_start": result['access_token'][:20] + "..."
                }
            else:
                return {
                    "status": "❌ Invalid credentials",
                    "error": result
                }
            '''
@app.get("/api/messages")
async def messages_test():
    logger.info("✅ GET request to /api/messages - Bot framework check")
    return {"message": "GET request successful"}


@app.post("/api/messages")
async def messages(request: Request):
    try:
        logger.info("🔔 /api/messages endpoint triggered")
        body = await request.json()
        logger.info(f"📦 Incoming request body: {body}")

        activity = Activity().deserialize(body)
        auth_header = request.headers.get("Authorization", "")

        async def call_bot(turn_context):
            await bot.on_turn(turn_context)

        await adapter.process_activity(activity, auth_header, call_bot)
        logger.info("✅ Activity processed successfully")
        return Response(status_code=200)

    except Exception as e:
        logger.exception("❌ Exception in /api/messages")
        raise HTTPException(status_code=500, detail=f"Bot error: {e}")

@app.get("/")
async def root():
    return {"message": "✅ HR Chatbot is running."}