import time

import requests
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, GROQ_MODEL, require
from embeddings import get_huggingface_embeddings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CANDIDATE_K = 20  # Chunks fetched from Pinecone before filtering
TOP_K = 8  # Chunks actually sent to the LLM
MAX_CHUNKS_PER_FILE = 2  # Stops one large file from crowding out the rest
REQUEST_TIMEOUT = 60  # Seconds to wait for the Groq API
MAX_TOKENS = 4096  # Upper limit on the length of each answer
MAX_RETRIES = 3  # Retries when Groq's rate limit is hit
MAX_RETRY_WAIT = 30  # Longest wait (seconds) before a single retry
TRUNCATION_NOTE = "\n\n*(Answer cut off at the length limit. Try asking a narrower question.)*"

SYSTEM_PROMPT = (
    "You are a senior software engineer answering questions about a codebase. "
    "The CONTEXT contains excerpts of the codebase retrieved for this question; "
    "each excerpt starts with its file path.\n\n"
    "Rules:\n"
    "- Base your answer only on the CONTEXT. Do not invent code, files or behavior that it does not show.\n"
    "- Cite the file path for each claim, and quote short code snippets when they help.\n"
    "- If the CONTEXT does not contain enough to answer fully, say exactly what is missing "
    "instead of guessing. A partial answer that admits its gaps is better than a confident wrong one.\n"
    "- Start with a direct answer, then give the supporting details. Be concise."
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)


def list_namespaces():
    """Return the names of all codebases indexed in Pinecone (one namespace per codebase)."""
    return sorted(pinecone_index.describe_index_stats().namespaces)


def select_contexts(matches, top_k=TOP_K, max_per_file=MAX_CHUNKS_PER_FILE):
    """
    Pick the chunk texts to send to the LLM from Pinecone matches (best first).

    Skips chunks without text and exact duplicates, and takes at most max_per_file
    chunks from any one file so the context covers more of the codebase.
    """
    contexts, seen_texts, per_file = [], set(), {}
    for match in matches:
        metadata = match.get("metadata") or {}
        text = metadata.get("text")
        if not text or text in seen_texts:
            continue
        file_path = metadata.get("file_path") or metadata.get("source")
        if file_path is not None:
            if per_file.get(file_path, 0) >= max_per_file:
                continue
            per_file[file_path] = per_file.get(file_path, 0) + 1
        seen_texts.add(text)
        contexts.append(text)
        if len(contexts) == top_k:
            break
    return contexts


def perform_rag(query, namespace):
    """
    Perform Retrieval-Augmented Generation (RAG) to answer a query.

    Args:
        query (str): User query.
        namespace (str): Pinecone namespace of the codebase to search.

    Returns:
        str: LLM-generated response.
    """
    # Step 1: Generate embeddings for the query
    query_embedding = get_huggingface_embeddings(query).tolist()

    # Step 2: Query Pinecone for relevant contexts
    top_matches = pinecone_index.query(
        vector=query_embedding,
        top_k=CANDIDATE_K,
        include_metadata=True,
        namespace=namespace
    )
    contexts = select_contexts(top_matches["matches"])

    # Step 3: Construct an augmented query
    augmented_query = (
        "<CONTEXT>\n"
        + "\n\n-------\n\n".join(contexts)
        + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    )

    # Step 4: Query the Groq chat completions API
    headers = {
        "Authorization": f"Bearer {require('GROQ_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_query}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7
    }

    response = _post_with_retry(headers, payload)

    if response.status_code == 200:
        response_json = response.json()
        try:
            choice = response_json["choices"][0]
            answer = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                answer += TRUNCATION_NOTE
            return answer
        except (KeyError, IndexError):
            return "No response text found in the Groq API response."
    elif response.status_code == 429:
        raise ValueError("Groq's rate limit was reached. Please wait a minute and try again.")
    else:
        raise ValueError(f"Error from Groq API: {response.status_code} - {response.text}")


def _post_with_retry(headers, payload):
    """POST to Groq, waiting and retrying when the rate limit is hit."""
    for attempt in range(MAX_RETRIES + 1):
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code != 429 or attempt == MAX_RETRIES:
            return response
        # Groq says how many seconds to wait in the retry-after header
        try:
            wait = float(response.headers.get("retry-after", 2 ** attempt))
        except (TypeError, ValueError):
            wait = 2 ** attempt
        time.sleep(min(wait, MAX_RETRY_WAIT))
