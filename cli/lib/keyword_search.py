from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    preprocessed_query = preprocess_text(query)
    for movie in movies:
        if preprocessed_query in preprocess_text(movie["title"]):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def preprocess_text(text: str) -> str:
    # Step one, make lowercase
    text = text.lower()
    # Step two, remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text