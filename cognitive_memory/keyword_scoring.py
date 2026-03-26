"""Lightweight BM25 keyword scoring for cognitive memory.

Provides a term-frequency based scoring signal complementary to semantic
embedding similarity. Ported from Ori-Mnemos's BM25 signal.

Reference: Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond"
"""
import math
from collections import Counter
from typing import List

STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'with', 'and', 'or', 'for',
    'in', 'on', 'at', 'to', 'of', 'it', 'its', 'by', 'as', 'that', 'this',
    'from', 'has', 'have', 'be', 'been', 'what', 'how', 'where', 'when',
    'which', 'who', 'do', 'does', 'did', 'not', 'but', 'so', 'if', 'than',
    'then', 'about', 'up', 'out', 'would', 'could', 'should', 'can', 'will',
    'just', 'also', 'than', 'more', 'very', 'much', 'some', 'any', 'all',
    'into', 'over', 'after', 'before', 'between', 'under', 'through',
    'i', 'we', 'you', 'he', 'she', 'they', 'me', 'us', 'my', 'our', 'your',
})


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_doc_len: float,
    doc_count: int,
    doc_freq: dict,  # term -> number of docs containing term
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a single document against a query."""
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in tf:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
        term_freq = tf[term]
        numerator = term_freq * (k1 + 1)
        denominator = term_freq + k1 * (1 - b + b * dl / max(avg_doc_len, 1))
        score += idf * numerator / denominator
    return score


def compute_bm25_scores(
    query: str,
    documents: List[str],
) -> List[float]:
    """Compute BM25 scores for a list of documents against a query.

    Returns a list of scores, one per document.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return [0.0] * len(documents)

    # Tokenize all documents
    doc_token_lists = [tokenize(doc) for doc in documents]
    doc_count = len(documents)
    avg_doc_len = sum(len(dt) for dt in doc_token_lists) / max(doc_count, 1)

    # Compute document frequency for each term
    doc_freq: Counter = Counter()
    for dt in doc_token_lists:
        for term in set(dt):
            doc_freq[term] += 1

    return [
        bm25_score(query_tokens, dt, avg_doc_len, doc_count, doc_freq)
        for dt in doc_token_lists
    ]
