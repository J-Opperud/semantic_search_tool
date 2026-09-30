"""Utilities for splitting documents into overlapping text chunks."""


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into fixed-size overlapping chunks.

    Args:
        text: The document text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between adjacent chunks.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If chunk_size or overlap is invalid.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks