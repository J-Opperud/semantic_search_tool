from chunking import chunk_text


def test_basic_chunking():
    text = "A" * 1000

    chunks = chunk_text(
        text,
        chunk_size=300,
        overlap=50,
    )

    assert len(chunks) > 1
    assert len(chunks[0]) == 300


def test_empty_text():
    assert chunk_text("") == []


def test_invalid_chunk_size():
    try:
        chunk_text("hello", chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_invalid_overlap():
    try:
        chunk_text(
            "hello",
            chunk_size=100,
            overlap=100,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")