import os

from dotenv import load_dotenv

# Load variables from a .env file (if present) into the environment
load_dotenv()


def require(name):
    """Return a required setting, or raise a clear error if it isn't set."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}. See .env.example.")
    return value


PINECONE_API_KEY = require("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "codebase-rag")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")  # Only used when ingest.py creates the index
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")  # Optional: repo selected by default in the app
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Only needed by the chatbot, not by ingest.py
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
