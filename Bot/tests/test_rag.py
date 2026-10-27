"""Offline tests: Pinecone, Groq and the embedding model are mocked, so no credentials or network are needed."""
import importlib
import sys
from unittest import mock

import numpy as np
import pytest

from conftest import BOT_DIR

NAMESPACE = "https://github.com/example/repo"

fake_index = mock.MagicMock()
fake_pc = mock.MagicMock()
fake_pc.Index.return_value = fake_index


@pytest.fixture(scope="module")
def rag():
    # Patch the Pinecone client so importing rag.py doesn't touch the network
    with mock.patch("pinecone.Pinecone", return_value=fake_pc):
        sys.modules.pop("rag", None)
        import rag as module
        yield module


@pytest.fixture(autouse=True)
def fake_embed(rag):
    with mock.patch.object(rag, "get_huggingface_embeddings", return_value=np.zeros(768, dtype=np.float32)) as m:
        yield m


def groq_ok(text="answer"):
    resp = mock.MagicMock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    return resp


def matches(*texts):
    return {"matches": [{"metadata": {"text": t}} for t in texts]}


def test_connects_to_configured_index(rag):
    fake_pc.Index.assert_called_with("codebase-rag")


def test_list_namespaces_is_sorted(rag):
    fake_index.describe_index_stats.return_value.namespaces = {"b-repo": {}, "a-repo": {}}
    assert rag.list_namespaces() == ["a-repo", "b-repo"]


def test_happy_path_builds_prompt_and_returns_answer(rag, fake_embed):
    fake_index.query.return_value = matches("def login(): ...", "class User: ...")
    with mock.patch.object(rag.requests, "post", return_value=groq_ok("It uses sessions.")) as post:
        out = rag.perform_rag("How does login work?", NAMESPACE)

    assert out == "It uses sessions."
    fake_embed.assert_called_once_with("How does login work?")

    query = fake_index.query.call_args.kwargs
    assert query["top_k"] == rag.CANDIDATE_K
    assert query["include_metadata"] is True
    assert query["namespace"] == NAMESPACE
    assert isinstance(query["vector"], list) and len(query["vector"]) == 768

    assert post.call_args.args[0] == rag.GROQ_URL
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-groq-key"
    assert post.call_args.kwargs["timeout"] == rag.REQUEST_TIMEOUT
    body = post.call_args.kwargs["json"]
    assert body["model"] == "test-model"
    system_msg, user_msg = body["messages"]
    assert system_msg == {"role": "system", "content": rag.SYSTEM_PROMPT}
    assert user_msg["content"] == (
        "<CONTEXT>\ndef login(): ...\n\n-------\n\nclass User: ...\n-------\n</CONTEXT>"
        "\n\n\n\nMY QUESTION:\nHow does login work?"
    )


def test_no_matches_still_calls_llm_with_empty_context(rag):
    fake_index.query.return_value = matches()
    with mock.patch.object(rag.requests, "post", return_value=groq_ok()) as post:
        rag.perform_rag("q", NAMESPACE)
    assert post.call_args.kwargs["json"]["messages"][1]["content"].startswith("<CONTEXT>\n\n-------\n</CONTEXT>")


def test_matches_without_text_are_skipped(rag):
    fake_index.query.return_value = {"matches": [
        {"metadata": {"filepath": "a.py"}},
        {"id": "no-metadata"},
        {"metadata": {"text": "real chunk"}},
    ]}
    with mock.patch.object(rag.requests, "post", return_value=groq_ok()) as post:
        rag.perform_rag("q", NAMESPACE)
    assert post.call_args.kwargs["json"]["messages"][1]["content"].startswith("<CONTEXT>\nreal chunk\n-------")


def chunk(file_path, text):
    return {"metadata": {"file_path": file_path, "text": text}}


def test_select_contexts_limits_chunks_per_file(rag):
    found = [chunk("prompts.ts", "p1"), chunk("prompts.ts", "p2"), chunk("prompts.ts", "p3"),
             chunk("reviews.ts", "r1"), chunk("prompts.ts", "p4")]
    assert rag.select_contexts(found, top_k=8, max_per_file=2) == ["p1", "p2", "r1"]


def test_select_contexts_skips_duplicates_and_stops_at_top_k(rag):
    found = [chunk("a.py", "same"), {"metadata": {"source": "b.py", "text": "same"}},
             chunk("c.py", "c"), chunk("d.py", "d"), chunk("e.py", "e")]
    assert rag.select_contexts(found, top_k=3) == ["same", "c", "d"]


def test_select_contexts_keeps_chunks_without_file_path(rag):
    found = [{"metadata": {"text": "x"}}, {"metadata": {"text": "y"}}, {"metadata": {"text": "z"}}]
    assert rag.select_contexts(found, max_per_file=1) == ["x", "y", "z"]


def test_groq_http_error_raises_value_error(rag):
    fake_index.query.return_value = matches("x")
    bad = mock.MagicMock(status_code=401, text="invalid api key")
    with mock.patch.object(rag.requests, "post", return_value=bad):
        with pytest.raises(ValueError, match="Groq API: 401 - invalid api key"):
            rag.perform_rag("q", NAMESPACE)


def rate_limited(retry_after="1.5"):
    return mock.MagicMock(status_code=429, headers={"retry-after": retry_after}, text="rate limit")


def test_rate_limit_waits_and_retries(rag):
    fake_index.query.return_value = matches("x")
    responses = [rate_limited("1.5"), rate_limited("999"), groq_ok("finally")]
    with mock.patch.object(rag.requests, "post", side_effect=responses) as post, \
         mock.patch.object(rag.time, "sleep") as sleep:
        assert rag.perform_rag("q", NAMESPACE) == "finally"
    assert post.call_count == 3
    # Waits as long as Groq asks, but never longer than MAX_RETRY_WAIT
    assert [c.args[0] for c in sleep.call_args_list] == [1.5, rag.MAX_RETRY_WAIT]


def test_rate_limit_gives_up_with_friendly_error(rag):
    fake_index.query.return_value = matches("x")
    with mock.patch.object(rag.requests, "post", return_value=rate_limited()) as post, \
         mock.patch.object(rag.time, "sleep"):
        with pytest.raises(ValueError, match="rate limit was reached"):
            rag.perform_rag("q", NAMESPACE)
    assert post.call_count == rag.MAX_RETRIES + 1


def test_missing_groq_key_raises_clear_error(rag, monkeypatch):
    fake_index.query.return_value = matches("x")
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        rag.perform_rag("q", NAMESPACE)


def test_truncated_answer_gets_note(rag):
    fake_index.query.return_value = matches("x")
    resp = mock.MagicMock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    with mock.patch.object(rag.requests, "post", return_value=resp):
        assert rag.perform_rag("q", NAMESPACE) == "partial" + rag.TRUNCATION_NOTE


def test_malformed_groq_body_returns_fallback(rag):
    fake_index.query.return_value = matches("x")
    resp = mock.MagicMock(status_code=200)
    resp.json.return_value = {"choices": []}
    with mock.patch.object(rag.requests, "post", return_value=resp):
        assert rag.perform_rag("q", NAMESPACE) == "No response text found in the Groq API response."


def test_embedding_model_is_loaded_once():
    import embeddings
    embeddings._load_model.cache_clear()
    with mock.patch.object(embeddings, "SentenceTransformer") as st_cls:
        st_cls.return_value.encode.return_value = np.zeros(768)
        embeddings.get_huggingface_embeddings("a")
        embeddings.get_huggingface_embeddings("b")
    st_cls.assert_called_once_with("sentence-transformers/all-mpnet-base-v2")
    embeddings._load_model.cache_clear()


def test_config_reports_missing_pinecone_key(monkeypatch):
    import config
    monkeypatch.delenv("PINECONE_API_KEY")
    with mock.patch("dotenv.load_dotenv"):  # don't let a local .env refill it
        with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
            importlib.reload(config)
    monkeypatch.undo()
    importlib.reload(config)


def run_app(rag, namespaces, **perform_rag_kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    patches = [
        mock.patch.object(rag, "list_namespaces", return_value=namespaces),
        mock.patch.object(rag, "perform_rag", **perform_rag_kwargs),
    ]
    for p in patches:
        p.start()
    try:
        at = AppTest.from_file(str(BOT_DIR / "app.py"), default_timeout=60).run()
        return at, rag.perform_rag, patches
    except Exception:
        for p in patches:
            p.stop()
        raise


def test_streamlit_chat_round_trip(rag):
    at, perform, patches = run_app(rag, ["other-repo", NAMESPACE], return_value="**mock answer**")
    try:
        assert at.title[0].value == "Codebase Chatbot"
        assert at.sidebar.selectbox[0].value == NAMESPACE  # PINECONE_NAMESPACE is preselected
        assert len(at.chat_message) == 1  # greeting
        at.chat_input[0].set_value("What does app.py do?").run()
        assert [m.name for m in at.chat_message] == ["assistant", "user", "assistant"]
        assert at.session_state.messages[-1]["content"] == "**mock answer**"
        perform.assert_called_once_with("What does app.py do?", NAMESPACE)

        # Switching codebases starts a fresh conversation
        at.sidebar.selectbox[0].set_value("other-repo").run()
        assert len(at.chat_message) == 1
    finally:
        for p in patches:
            p.stop()


def test_streamlit_without_codebases_disables_chat(rag):
    at, _, patches = run_app(rag, [], return_value="unused")
    try:
        assert at.sidebar.info[0].value == "No codebases indexed yet."
        assert at.chat_input[0].disabled
    finally:
        for p in patches:
            p.stop()


def test_streamlit_shows_errors(rag):
    at, _, patches = run_app(rag, [NAMESPACE], side_effect=ValueError("boom"))
    try:
        at.chat_input[0].set_value("hi").run()
        assert at.error[0].value == "An error occurred: boom"
        assert at.session_state.messages[-1]["content"] == "An error occurred: boom"
    finally:
        for p in patches:
            p.stop()
