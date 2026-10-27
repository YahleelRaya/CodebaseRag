# Codebase RAG

Chat with any GitHub repository in plain English. Point the tool at a repo and it indexes the code. Then ask questions like *"How is a pull request review triggered?"* or *"Where is the private key validated?"* and get answers that cite the files they come from.

It uses **Retrieval-Augmented Generation (RAG)**. Source code is split into chunks, embedded with a Sentence-Transformers model, and stored in a **Pinecone** vector index. For each question, the most relevant chunks are retrieved and passed to an LLM served by **Groq**. The chat UI is built with **Streamlit**.

```
                     INDEXING (ingest.py)
 GitHub repo ──clone──▶ filter files ──▶ line-based chunks ──▶ embed (768-d) ──▶ Pinecone
                                          + file path & line                    namespace per repo

                     ASKING (app.py → rag.py)
 question ──▶ embed ──▶ Pinecone: top 20 ──▶ pick best 8 (≤2 per file, no duplicates)
                                                        │
 answer with file citations ◀── Groq LLM ◀── grounded system prompt + <CONTEXT> + question
```

## Features

- **One-command indexing.** `python ingest.py https://github.com/owner/repo` clones a repo, chunks it and uploads it. Re-running it replaces that repo's old chunks, so you never get duplicates or stale code.
- **Several repos in one index.** Each repo gets its own Pinecone namespace, and the sidebar lets you switch between them.
- **Varied context.** Retrieval takes at most 2 chunks per file, so one large file can't crowd out the rest of the codebase.
- **Grounded answers.** The model must cite file paths and say what's missing rather than invent code.
- **Built-in error handling.** Waits and retries automatically when the Groq rate limit is hit, shows a note when an answer is cut off, and reports missing configuration clearly.
- **Tested offline.** 26 `pytest` tests with Pinecone, Groq and the embedding model mocked, so they need no API keys or network.

---

## Table of contents

- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Limitations and future work](#limitations-and-future-work)
- [Troubleshooting](#troubleshooting)

---

## Project structure

```
Codebase-Rag/
├── Bot/
│   ├── app.py               # Streamlit chat UI with a repo picker in the sidebar
│   ├── rag.py               # Question answering: embed → retrieve → select → prompt → Groq
│   ├── ingest.py            # CLI that indexes a GitHub repo or local folder into Pinecone
│   ├── embeddings.py        # Sentence-Transformers embeddings (model loaded once and cached)
│   ├── config.py            # Reads settings from environment variables / .env
│   ├── .env.example         # Template for your .env file
│   ├── requirements.txt     # Runtime dependencies
│   ├── requirements-dev.txt # Runtime + test dependencies
│   └── tests/
│       ├── conftest.py      # Dummy settings so tests never need real keys
│       ├── test_rag.py      # Retrieval, prompting, Groq errors/retries, config, Streamlit UI
│       └── test_ingest.py   # File filtering, chunking, indexing
├── .gitignore
└── README.md
```

---

## Quick start

### Prerequisites

- **Python 3.10+**
- **git** on your `PATH` (used to clone the repos you index)
- A free **Pinecone** API key: <https://app.pinecone.io>
- A free **Groq** API key: <https://console.groq.com/keys>
- About 2 GB of disk space for PyTorch and the embedding model (downloaded on first use)

### 1. Install

```bash
git clone https://github.com/YahleelRaya/CodebaseRag.git
cd CodebaseRag/Bot

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# Optional: CPU-only PyTorch is a much smaller download if you have no GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and set `PINECONE_API_KEY` and `GROQ_API_KEY`. Nothing else is required. `.env` is gitignored, so your keys never get committed.

### 3. Index a repository

```bash
python ingest.py https://github.com/owner/repo
```

The Pinecone index (`codebase-rag` by default) is created automatically the first time you run this.

### 4. Chat

```bash
streamlit run app.py
```

Open <http://localhost:8501>, pick the repo in the sidebar, and ask a question.

---

## Usage

### Indexing

```bash
python ingest.py https://github.com/owner/repo           # clone and index a GitHub repo
python ingest.py path/to/local/project                    # index a local folder
python ingest.py https://github.com/owner/repo --namespace my-project
```

| Behavior | Details |
|----------|---------|
| Namespace | Defaults to the GitHub URL, or the folder name for local paths. |
| Files indexed | Common source and config types: `.py .js .jsx .ts .tsx .java .go .rs .c .cpp .cs .rb .php .swift .kt .html .css .sql .sh .md .yaml .toml` and more. |
| Files skipped | Folders like `.git`, `node_modules`, `venv`, `dist` and `build`; binary or non-UTF-8 files; empty files; and files over 500 KB (usually generated or minified). |
| Re-running | Clears the namespace first, then uploads, so the index always matches the current code. |

After indexing, click **Refresh list** in the app's sidebar if the new repo doesn't appear.

### Example questions

- "What does this project do, and how does a request flow through it?"
- "Which files handle authentication?"
- "How are errors from the external API handled?"
- "Are there any security concerns in how secrets are loaded?"

---

## How it works

### Indexing (`ingest.py`)

1. **Clone** the repo with `git clone --depth 1` into a temporary folder.
2. **Filter** files by extension, size and folder (see the table above).
3. **Chunk** each file into pieces of about 1,500 characters, split on line boundaries so a statement is never cut in half. Consecutive chunks share 3 lines of overlap. The size is chosen to fit within the embedding model's 384-token input limit.
4. **Label** each chunk with its location (`File: src/app.ts (line 34)`) so the LLM can cite it.
5. **Embed** all chunks in batches with `sentence-transformers/all-mpnet-base-v2` (768 dimensions).
6. **Upload** to Pinecone in batches of 100. Each vector ID has the form `path:start_line`, and the metadata holds `text`, `file_path` and `start_line`.

### Answering (`rag.py`)

1. **Embed** the question with the same model used for indexing.
2. **Retrieve** the 20 most similar chunks from the selected namespace (cosine similarity).
3. **Select** the 8 best chunks, skipping chunks with no text and exact duplicates, and taking **at most 2 chunks per file**. Without this cap, one file with lots of related vocabulary (for example a prompts file) filled every slot, and the answer missed the file that actually mattered.
4. **Prompt** the LLM with a system prompt that requires answers to come only from the retrieved code, cite file paths, and **say what's missing instead of guessing**. The chunks are wrapped in a `<CONTEXT>` block, followed by the question.
5. **Generate** with Groq's OpenAI-compatible chat API (`openai/gpt-oss-120b` by default, temperature 0.7, up to 4,096 tokens):
   - **429 rate limit:** waits for the time in Groq's `retry-after` header (at most 30 s) and retries up to 3 times.
   - **Answer cut off at the token limit:** adds a note asking for a narrower question.
   - **Other errors:** raises a clear error, which the UI shows in the chat.

### UI (`app.py`)

- **Sidebar:** lists every indexed repo (cached for 60 s, with a Refresh button). `PINECONE_NAMESPACE` sets the default.
- **History:** chat history is kept in Streamlit session state and resets when you switch repos.
- **Errors:** they appear in the chat, so the app never crashes on a failed question.
- **Empty index:** if nothing is indexed yet, the chat box is disabled and the ingest command is shown.

---

## Configuration

All settings come from environment variables, which are loaded from `Bot/.env` if that file exists.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `PINECONE_API_KEY` | ✅ | none | Pinecone access |
| `GROQ_API_KEY` | ✅ for chatting | none | Groq access (not needed by `ingest.py`) |
| `PINECONE_INDEX_NAME` | | `codebase-rag` | Index used for all repos |
| `PINECONE_REGION` | | `us-east-1` | AWS region, used only when `ingest.py` creates the index |
| `PINECONE_NAMESPACE` | | none | Repo selected by default in the sidebar |
| `GROQ_MODEL` | | `openai/gpt-oss-120b` | Any chat model listed at <https://console.groq.com/docs/models> |

Retrieval and generation settings are constants at the top of `rag.py` (`CANDIDATE_K`, `TOP_K`, `MAX_CHUNKS_PER_FILE`, `MAX_TOKENS`, `MAX_RETRIES`, and others). Chunking settings are at the top of `ingest.py`.

---

## Testing

```bash
cd Bot
pip install -r requirements-dev.txt
pytest
```

All 26 tests run offline. Pinecone, Groq and the embedding model are mocked, and `conftest.py` sets dummy keys so a real `.env` is never used.

| Area | What's covered |
|------|----------------|
| Retrieval | Query settings, namespace routing, the per-file chunk cap, duplicate removal, chunks with missing metadata |
| Prompting | Exact `<CONTEXT>` format, system prompt, model and request options |
| Groq API | HTTP errors, malformed responses, cut-off answers, rate-limit retries (waits as long as Groq asks, up to 30 s), giving up with a clear message |
| Config | A clear error when a required key is missing |
| Embeddings | Model loaded only once |
| Ingestion | File filtering, line-based chunking with overlap, very long lines, record format, namespace replacement, index creation |
| UI | `streamlit.testing` checks: chat round-trip, repo picker defaults, a new conversation when switching repos, empty index, error display |

---

## Evaluation

To measure answer quality, I indexed a TypeScript project ([SecureAgent](https://github.com/YahleelRaya/SecureAgent), an AI pull-request review bot: 14 files, 49 chunks, indexed in about 30 s). I asked it 7 questions and checked every answer against the source code by hand.

| Version | Correct | Partly correct | Wrong | Answers with made-up code |
|---------|:-------:|:--------------:|:-----:|:-----------------------:|
| Initial (top 5 chunks, generic prompt) | 3 | 3 | 1 | 2 |
| Current (≤2 chunks per file + grounded prompt) | 6 | 1 | 0 | 0 |

What changed:
- **Per-file cap.** A question about posting inline comments previously got three chunks of `prompts.ts` and never saw `reviews.ts`. It now gets `reviews.ts` and quotes the real API call.
- **Grounded prompt.** Asked whether Python files were supported, the first version confidently said *yes*. That was wrong, because the Python parser exists but is never registered. The current version traces the code to the lookup function and says it can't see the mapping, instead of guessing.

---

## Limitations and future work

- **Pure vector search misses exact identifiers.** Questions that depend on a specific line, such as a lookup table in a constants file, can miss it if the wording doesn't match. Possible fixes: hybrid search (BM25 keywords plus vectors), or a code-specific embedding model.
- **No conversation memory.** Each question is answered on its own, so follow-ups like "what about the second one?" have no context. Sending recent turns to the LLM would fix this.
- **Groq free-tier rate limits.** The free tier allows 8,000 tokens per minute. Several questions in quick succession cause 15–35 s waits; the app retries automatically.
- **Fixed-size chunking.** Chunks follow line boundaries, not code structure. Splitting on functions and classes (for example with tree-sitter) would give more meaningful chunks.
- **Public repos only.** `ingest.py` runs a plain `git clone`. For a private repo, clone it yourself and pass the local folder path.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Missing required environment variable: …` | Create `Bot/.env` from `.env.example` and fill in the key. |
| Sidebar says "No codebases indexed yet." | Run `python ingest.py <repo-url>`, then click **Refresh list**. |
| `Error from Groq API: 401` | `GROQ_API_KEY` is invalid or revoked. |
| `Error from Groq API: 404 … model … does not exist` | Groq retired the model. Set `GROQ_MODEL` to one from <https://console.groq.com/docs/models>. |
| `Groq's rate limit was reached` | The free tier is busy; wait a minute and try again. |
| Answers don't match the code | Re-run `ingest.py` so the index matches the latest code, and check the right repo is selected. |
| First question is slow | The embedding model (~420 MB) downloads on first use, then stays cached. |
| `pip` fails with `No such file or directory` deep in `site-packages` (Windows) | The path is too long. [Enable long paths](https://pip.pypa.io/warnings/enable-long-paths) or create the venv in a shorter folder. |
| `git` not found during ingestion | Install git, or clone the repo yourself and pass the local path. |
