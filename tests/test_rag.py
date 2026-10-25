import pytest

from rag import build_prompt, chunk_text


def test_chunk_text_returns_chunks():
    text = "A" * 2000
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_chunk_overlap_preserves_content():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=2)

    assert chunks[0][-2:] == chunks[1][:2]


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)


def test_prompt_contains_question_and_context():
    contexts = [
        {
            "source": "example.txt",
            "text": "RAG combines retrieval with generation.",
        }
    ]

    prompt = build_prompt("What is RAG?", contexts)

    assert "What is RAG?" in prompt
    assert "RAG combines retrieval with generation." in prompt
    assert "example.txt" in prompt
