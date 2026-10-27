"""Offline tests for ingest.py: Pinecone and the embedding model are mocked."""
from unittest import mock

import numpy as np

import ingest


def test_chunk_text_keeps_short_file_whole():
    assert ingest.chunk_text("a = 1\nb = 2") == [(1, "a = 1\nb = 2")]


def test_chunk_text_splits_on_lines_with_overlap():
    content = "\n".join(f"line {i:02d}" for i in range(1, 21))  # 20 lines of 7 chars
    chunks = ingest.chunk_text(content, chunk_size=40, overlap_lines=1)

    assert all(len(text) <= 40 for _, text in chunks)
    assert chunks[0] == (1, "line 01\nline 02\nline 03\nline 04\nline 05")
    # The next chunk starts by repeating the last line of the previous one
    assert chunks[1][0] == 5 and chunks[1][1].startswith("line 05\n")
    assert chunks[-1][1].endswith("line 20")


def test_chunk_text_handles_line_longer_than_chunk_size():
    chunks = ingest.chunk_text("x" * 100 + "\nshort", chunk_size=10, overlap_lines=3)
    assert chunks == [(1, "x" * 100), (2, "short")]


def test_iter_source_files_filters(tmp_path):
    (tmp_path / "app.py").write_text("print('hi')")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.py").write_text("   ")
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "util.js").write_text("export {}")

    assert [path for path, _ in ingest.iter_source_files(tmp_path)] == ["app.py", "src/util.js"]


def test_build_records_includes_file_path_in_text(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    pass")

    [record] = ingest.build_records(tmp_path)
    assert record["id"] == "src/main.py:1"
    assert record["text"] == "File: src/main.py (line 1)\n\ndef main():\n    pass"
    assert record["file_path"] == "src/main.py" and record["start_line"] == 1


def test_ingest_local_repo_replaces_namespace(tmp_path):
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.py").write_text("y = 2")

    fake_index = mock.MagicMock()
    fake_index.describe_index_stats.return_value.namespaces = {"my-ns": {}}
    fake_pc = mock.MagicMock()
    fake_pc.has_index.return_value = True
    fake_pc.Index.return_value = fake_index

    with mock.patch.object(ingest, "Pinecone", return_value=fake_pc), \
         mock.patch.object(ingest, "get_huggingface_embeddings", return_value=np.ones((2, 768))) as embed:
        namespace, count = ingest.ingest(str(tmp_path), namespace="my-ns")

    assert (namespace, count) == ("my-ns", 2)
    embed.assert_called_once()
    fake_pc.create_index.assert_not_called()
    fake_index.delete.assert_called_once_with(delete_all=True, namespace="my-ns")

    upserted = fake_index.upsert.call_args.kwargs
    assert upserted["namespace"] == "my-ns"
    assert [v["id"] for v in upserted["vectors"]] == ["a.py:1", "b.py:1"]
    assert len(upserted["vectors"][0]["values"]) == 768
    assert upserted["vectors"][0]["metadata"]["text"].startswith("File: a.py")


def test_ingest_creates_missing_index(tmp_path):
    (tmp_path / "a.py").write_text("x = 1")
    fake_pc = mock.MagicMock()
    fake_pc.has_index.return_value = False
    fake_pc.Index.return_value.describe_index_stats.return_value.namespaces = {}

    with mock.patch.object(ingest, "Pinecone", return_value=fake_pc), \
         mock.patch.object(ingest, "get_huggingface_embeddings", return_value=np.ones((1, 768))):
        namespace, _ = ingest.ingest(str(tmp_path))

    assert namespace == tmp_path.name  # defaults to the folder name
    kwargs = fake_pc.create_index.call_args.kwargs
    assert kwargs["dimension"] == 768 and kwargs["metric"] == "cosine"
    fake_pc.Index.return_value.delete.assert_not_called()  # nothing to clear in a new namespace
