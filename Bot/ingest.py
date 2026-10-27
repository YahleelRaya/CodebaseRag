"""
Index a codebase into Pinecone so the chatbot can answer questions about it.

Usage:
    python ingest.py https://github.com/owner/repo
    python ingest.py path/to/local/repo --namespace my-project
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_REGION
from embeddings import get_huggingface_embeddings

EMBEDDING_DIMENSION = 768  # Output size of all-mpnet-base-v2
CHUNK_SIZE = 1500  # Max characters per chunk (~384 tokens, the embedding model's input limit)
CHUNK_OVERLAP_LINES = 3  # Lines repeated between chunks so code isn't cut off without context
UPSERT_BATCH_SIZE = 100

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".rs",
    ".rb", ".php", ".swift", ".kt", ".scala", ".vue", ".svelte", ".html", ".css", ".scss",
    ".sql", ".sh", ".md", ".yaml", ".yml", ".toml", ".ipynb",
}
IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__", "dist", "build",
    ".next", "target", "vendor", ".idea", ".vscode",
}
MAX_FILE_BYTES = 500_000  # Skip very large (usually generated or minified) files


def clone_repository(repo_url, destination):
    """Shallow-clone a git repository into destination."""
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(destination)], check=True)
    return destination


def iter_source_files(repo_dir):
    """Yield (relative_path, content) for every supported, readable source file."""
    for path in sorted(repo_dir.rglob("*")):
        relative = path.relative_to(repo_dir)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if content.strip():
            yield relative.as_posix(), content


def chunk_text(content, chunk_size=CHUNK_SIZE, overlap_lines=CHUNK_OVERLAP_LINES):
    """
    Split text into chunks of whole lines, each at most about chunk_size characters.

    Returns:
        list[tuple[int, str]]: (1-based start line, chunk text) pairs.
    """
    lines = content.splitlines()
    chunks = []
    start = 0
    while start < len(lines):
        end, size = start, 0
        # Always take at least one line so a single long line can't stall the loop
        while end < len(lines) and (end == start or size + len(lines[end]) + 1 <= chunk_size):
            size += len(lines[end]) + 1
            end += 1
        chunk = "\n".join(lines[start:end])
        if chunk.strip():
            chunks.append((start + 1, chunk))
        if end >= len(lines):
            break
        start = max(end - overlap_lines, start + 1)
    return chunks


def build_records(repo_dir):
    """Turn every source file in repo_dir into Pinecone-ready records (without vectors)."""
    records = []
    for file_path, content in iter_source_files(repo_dir):
        for start_line, chunk in chunk_text(content):
            records.append({
                "id": f"{file_path}:{start_line}",
                # The file path is part of the text so the LLM knows where each chunk comes from
                "text": f"File: {file_path} (line {start_line})\n\n{chunk}",
                "file_path": file_path,
                "start_line": start_line,
            })
    return records


def get_or_create_index(pc):
    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def ingest(source, namespace=None):
    """Index a GitHub URL or local directory into the given (or derived) namespace."""
    # ignore_cleanup_errors: git marks some .git files read-only, which blocks deletion on Windows
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        if Path(source).is_dir():
            repo_dir = Path(source).resolve()
            namespace = namespace or repo_dir.name
        else:
            repo_dir = clone_repository(source, Path(tmp) / "repo")
            namespace = namespace or source

        records = build_records(repo_dir)
        if not records:
            raise SystemExit("No supported source files found.")
        print(f"Embedding {len(records)} chunks...")
        vectors = get_huggingface_embeddings([r["text"] for r in records])

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)

    # Clear the namespace first so re-running never leaves duplicates or stale chunks
    if namespace in index.describe_index_stats().namespaces:
        index.delete(delete_all=True, namespace=namespace)

    for i in range(0, len(records), UPSERT_BATCH_SIZE):
        batch = [
            {
                "id": r["id"],
                "values": v.tolist(),
                "metadata": {"text": r["text"], "file_path": r["file_path"], "start_line": r["start_line"]},
            }
            for r, v in zip(records[i:i + UPSERT_BATCH_SIZE], vectors[i:i + UPSERT_BATCH_SIZE])
        ]
        index.upsert(vectors=batch, namespace=namespace)

    print(f"Indexed {len(records)} chunks into namespace '{namespace}'.")
    return namespace, len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a codebase into Pinecone.")
    parser.add_argument("source", help="GitHub URL or path to a local repository")
    parser.add_argument("--namespace", help="Pinecone namespace (defaults to the URL or folder name)")
    args = parser.parse_args()
    ingest(args.source, args.namespace)
