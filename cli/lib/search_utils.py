import json
import os
from typing import Any, TypedDict
import string

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SEARCH_LIMIT = 10
DEFAULT_CHUNK_SIZE = 200
DEFAULT_MAX_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1
SCORE_PRECISION = 3
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75

class Movie(TypedDict):
    id: int
    title: str
    description: str
    
class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]
    
def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]
    
def load_stopwords() -> set[str]:
    # Load stopwords from a file
    with open(STOPWORDS_PATH, "r") as f:
        stopwords = []
        stopwords_raw = f.read().splitlines()
        for stopword in stopwords_raw:
            stopword = stopword.strip()
            if stopword:
                stopword = stopword.lower()
                stopword = stopword.translate(str.maketrans("", "", string.punctuation))
                stopwords.append(stopword)
    return set(stopwords)

def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }