"""
Text chunking strategies for splitting documents into manageable pieces.
Each strategy demonstrates a different approach with trade-offs.
"""

from dataclasses import dataclass
import re


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# ═════════════════════════════════════════════════════════════════════════
#  Basic strategies
# ═════════════════════════════════════════════════════════════════════════

def chunk_by_characters(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """
    Fixed-size character chunking with overlap.
    Simplest approach — splits at exact character boundaries.
    """
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                metadata={"strategy": "character", "start": start, "end": min(end, len(text))},
            ))
            idx += 1
        start += chunk_size - overlap
    return chunks


def chunk_by_sentences(text: str, max_sentences: int = 5, overlap_sentences: int = 1) -> list[Chunk]:
    """
    Sentence-based chunking. Respects sentence boundaries.
    Better semantic coherence than character chunking.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    idx = 0
    start = 0
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk_text = " ".join(sentences[start:end]).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                metadata={"strategy": "sentence", "sentences": list(range(start, end))},
            ))
            idx += 1
        start += max_sentences - overlap_sentences
        if start >= len(sentences):
            break
    return chunks


def chunk_by_paragraphs(text: str, max_paragraphs: int = 2, overlap_paragraphs: int = 0) -> list[Chunk]:
    """
    Paragraph-based chunking. Uses double newlines as delimiters.
    Best for well-structured documents.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    idx = 0
    start = 0
    while start < len(paragraphs):
        end = min(start + max_paragraphs, len(paragraphs))
        chunk_text = "\n\n".join(paragraphs[start:end]).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                metadata={"strategy": "paragraph", "paragraphs": list(range(start, end))},
            ))
            idx += 1
        step = max_paragraphs - overlap_paragraphs
        start += max(step, 1)
    return chunks


def chunk_recursive(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """
    Recursive character text splitter (similar to LangChain's approach).
    Tries to split on natural boundaries, falling back to smaller separators.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        sep = seps[0] if seps else ""
        remaining_seps = seps[1:] if len(seps) > 1 else [""]

        if sep:
            parts = text.split(sep)
        else:
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        result = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > chunk_size:
                    result.extend(_split(part, remaining_seps))
                else:
                    current = part.strip()
        if current:
            result.append(current)
        return result

    pieces = _split(text, separators)
    return [
        Chunk(text=p, index=i, metadata={"strategy": "recursive", "chunk_size": chunk_size})
        for i, p in enumerate(pieces) if p.strip()
    ]


# ═════════════════════════════════════════════════════════════════════════
#  Advanced strategies
# ═════════════════════════════════════════════════════════════════════════

def chunk_by_tokens(text: str, max_tokens: int = 256, overlap_tokens: int = 32) -> list[Chunk]:
    """
    Token-based chunking using tiktoken (cl100k_base / GPT-4 tokenizer).
    Ensures each chunk fits within a model's token window.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        enc = None

    if enc is None:
        words = text.split()
        approx_token = lambda w: max(1, len(w) // 4)
        chunks, current_words, current_tok, idx = [], [], 0, 0
        for w in words:
            t = approx_token(w)
            if current_tok + t > max_tokens and current_words:
                chunks.append(Chunk(
                    text=" ".join(current_words), index=idx,
                    metadata={"strategy": "token", "approx_tokens": current_tok},
                ))
                idx += 1
                keep = max(0, len(current_words) - int(overlap_tokens * 0.75))
                current_words = current_words[keep:]
                current_tok = sum(approx_token(ww) for ww in current_words)
            current_words.append(w)
            current_tok += t
        if current_words:
            chunks.append(Chunk(
                text=" ".join(current_words), index=idx,
                metadata={"strategy": "token", "approx_tokens": current_tok},
            ))
        return chunks

    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = enc.decode(tokens[start:end]).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text, index=idx,
                metadata={"strategy": "token", "tokens": end - start},
            ))
            idx += 1
        start += max_tokens - overlap_tokens
    return chunks


def chunk_by_markdown(text: str, max_chunk_size: int = 1500) -> list[Chunk]:
    """
    Markdown / header-aware chunking.
    Splits on headings (#, ##, ###, etc.) and keeps the heading with its content.
    Falls back to paragraph splitting within large sections.
    """
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    splits = []
    last_end = 0
    for m in header_pattern.finditer(text):
        if m.start() > last_end:
            pre_text = text[last_end:m.start()].strip()
            if pre_text:
                splits.append(pre_text)
        last_end = m.start()
    if last_end < len(text):
        splits.append(text[last_end:].strip())

    if not splits:
        splits = [text]

    chunks = []
    idx = 0
    for section in splits:
        if len(section) <= max_chunk_size:
            if section.strip():
                chunks.append(Chunk(
                    text=section.strip(), index=idx,
                    metadata={"strategy": "markdown"},
                ))
                idx += 1
        else:
            paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 > max_chunk_size and current:
                    chunks.append(Chunk(text=current.strip(), index=idx, metadata={"strategy": "markdown"}))
                    idx += 1
                    current = para
                else:
                    current = (current + "\n\n" + para).strip() if current else para
            if current.strip():
                chunks.append(Chunk(text=current.strip(), index=idx, metadata={"strategy": "markdown"}))
                idx += 1
    return chunks


def chunk_sliding_window(text: str, window_size: int = 500, step_size: int = 250) -> list[Chunk]:
    """
    Sliding window chunking with configurable stride.
    Creates overlapping windows over the text — good for ensuring no context is lost at boundaries.
    """
    chunks = []
    idx = 0
    start = 0
    while start < len(text):
        end = start + window_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text, index=idx,
                metadata={"strategy": "sliding_window", "start": start, "end": min(end, len(text))},
            ))
            idx += 1
        start += step_size
        if start >= len(text):
            break
    return chunks


def chunk_semantic(text: str, max_sentences: int = 3, similarity_threshold: float = 0.5) -> list[Chunk]:
    """
    Semantic chunking — groups consecutive sentences that are topically similar.
    Uses sentence-level TF-IDF similarity to detect topic shifts.
    When similarity between adjacent groups drops below the threshold, a new chunk begins.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= max_sentences:
        return [Chunk(text=text.strip(), index=0, metadata={"strategy": "semantic"})]

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    try:
        tfidf = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(sentences)
    except ValueError:
        return [Chunk(text=text.strip(), index=0, metadata={"strategy": "semantic"})]

    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1))
    norms[norms == 0] = 1

    chunks = []
    current_group: list[str] = [sentences[0]]
    idx = 0

    for i in range(1, len(sentences)):
        if len(current_group) >= max_sentences:
            a_vec = matrix[i - len(current_group):i].mean(axis=0)
            b_vec = matrix[i]
            a_arr = np.asarray(a_vec).flatten()
            b_arr = np.asarray(b_vec.todense()).flatten()
            na = np.linalg.norm(a_arr)
            nb = np.linalg.norm(b_arr)
            sim = float(np.dot(a_arr, b_arr) / (na * nb + 1e-10)) if (na > 0 and nb > 0) else 0

            if sim < similarity_threshold:
                chunks.append(Chunk(
                    text=" ".join(current_group), index=idx,
                    metadata={"strategy": "semantic", "similarity_at_break": round(sim, 3)},
                ))
                idx += 1
                current_group = []
        current_group.append(sentences[i])

    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group), index=idx,
            metadata={"strategy": "semantic"},
        ))
    return chunks


# ═════════════════════════════════════════════════════════════════════════
#  Registry
# ═════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "character": chunk_by_characters,
    "sentence": chunk_by_sentences,
    "paragraph": chunk_by_paragraphs,
    "recursive": chunk_recursive,
    "token": chunk_by_tokens,
    "markdown": chunk_by_markdown,
    "sliding_window": chunk_sliding_window,
    "semantic": chunk_semantic,
}

STRATEGY_INFO = {
    "character":      "Fixed-size character splits with overlap",
    "sentence":       "Groups of N sentences with overlap",
    "paragraph":      "Split on double-newline paragraph boundaries",
    "recursive":      "Smart recursive split on natural boundaries",
    "token":          "Split by token count (GPT tokenizer)",
    "markdown":       "Header-aware split for Markdown / structured docs",
    "sliding_window": "Overlapping sliding windows with configurable stride",
    "semantic":       "Groups sentences by topical similarity (TF-IDF)",
}


def chunk_text(text: str, strategy: str = "recursive", **kwargs) -> list[Chunk]:
    """Chunk text using the specified strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy](text, **kwargs)
