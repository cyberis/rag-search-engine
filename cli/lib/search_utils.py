import json
import os
from typing import Any, TypedDict
import string

DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

class Movie(TypedDict):
    id: int
    title: str
    description: str
    
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

