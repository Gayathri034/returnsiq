"""Retrieval over policy chunks: sparse (TF-IDF), dense (embeddings), or hybrid."""
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .ingest import Chunk

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def stem(word: str) -> str:
    """Light suffix stemmer so 'reporting'/'report' and 'damaged'/'damage' match."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9%]+", text.lower()) if t not in ENGLISH_STOP_WORDS]


def dense_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _minmax(x: np.ndarray) -> np.ndarray:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


class Retriever:
    def __init__(self, chunks: list[Chunk], method: str = "tfidf"):
        if method not in {"tfidf", "dense", "hybrid"}:
            raise ValueError("method must be tfidf, dense, or hybrid")
        if method != "tfidf" and not dense_available():
            raise RuntimeError("Dense retrieval needs: pip install sentence-transformers")
        self.chunks, self.method = chunks, method
        texts = [c.full_text for c in chunks]

        # Sparse index (always built: also used for sentence scoring, coverage, and groundedness).
        # Settings chosen by ablation in eval: stemming on, unigrams, no heading boost.
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                          sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.vocab = set(self.vectorizer.vocabulary_)

        self.encoder = None
        if method in {"dense", "hybrid"}:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(DENSE_MODEL)
            self.dense_matrix = self.encoder.encode(texts, normalize_embeddings=True)

    def _sparse_scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return (self.tfidf_matrix @ q.T).toarray().ravel()

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        return self.dense_matrix @ q

    def scores(self, query: str) -> np.ndarray:
        if self.method == "tfidf":
            return self._sparse_scores(query)
        if self.method == "dense":
            return self._dense_scores(query)
        # Hybrid: average of min-max normalized scores, scaled by the raw
        # dense maximum so an off-topic query still scores low overall.
        sparse, dense = self._sparse_scores(query), self._dense_scores(query)
        return (0.5 * _minmax(sparse) + 0.5 * _minmax(dense)) * float(max(dense.max(), 0))

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        order = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in order]

    def coverage(self, query: str) -> float:
        """Share of the query's content words that appear anywhere in the policies.
        Low coverage means the question is about something the documents never discuss."""
        terms = tokenize(query)
        return sum(t in self.vocab for t in terms) / len(terms) if terms else 0.0

    def sentence_similarity(self, a: list[str], b: list[str]) -> np.ndarray:
        """TF-IDF cosine between two lists of sentences (rows a, columns b)."""
        va, vb = self.vectorizer.transform(a), self.vectorizer.transform(b)
        return (va @ vb.T).toarray()
