from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    preprocessed_query = preprocess_text(query)
    for movie in movies:
        if found(preprocessed_query, preprocess_text(movie["title"])):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def preprocess_text(text: str) -> set[str]:
    # Step one, make lowercase
    text = text.lower()
    # Step two, remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Step three, tokenize (split into words)
    tokens = set(text.split())
    return tokens

def found(query: set[str], title: set[str]) -> bool:
    # Check if a word in the query is found as a substring in any of the words in the title
    for q in query:
        for t in title:
            if q in t:
                return True
    return False