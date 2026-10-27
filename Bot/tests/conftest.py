import os
import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOT_DIR))

# Dummy settings so config.py imports without real credentials
# (set, not setdefault, so a developer's real environment never leaks into tests)
os.environ["PINECONE_API_KEY"] = "test-pinecone-key"
os.environ["PINECONE_INDEX_NAME"] = "codebase-rag"
os.environ["PINECONE_NAMESPACE"] = "https://github.com/example/repo"
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["GROQ_MODEL"] = "test-model"
