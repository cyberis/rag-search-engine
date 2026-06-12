from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    stopwords = load_stopwords()
    preprocessed_query = preprocess_text(query, stopwords)
    for movie in movies:
        prerocessed_title = preprocess_text(movie["title"], stopwords)
        if found(preprocessed_query, prerocessed_title):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def preprocess_text(text: str, stopwords: set[str]) -> set[str]:
    # Step one, make lowercase
    text = text.lower()
    # Step two, remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Step three, tokenize (split into words)
    tokens = set(text.split())
    # Step four, remove stopwords
    tokens = tokens - stopwords
    # Step five, apply stemming
    tokens = set(stemmer.stem(token) for token in tokens)
    return tokens

def found(query_tokens: set[str], title_tokens: set[str]) -> bool:
    # Check if a word in the query is found as a substring in any of the words in the title
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False