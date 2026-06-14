import os
import pickle
import string
import math
from typing import Set
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    CACHE_DIR,
    Movie, 
    load_movies, 
    load_stopwords,
)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.term_freqs: dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freqs_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.stopwords = load_stopwords()
        
    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text, self.stopwords)
        for token in tokens:
            self.index[token].add(doc_id)
            if doc_id not in self.term_freqs:
                self.term_freqs[doc_id] = Counter()
            self.term_freqs[doc_id][token] += 1
                
    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_freqs_path, "wb") as f:
            pickle.dump(self.term_freqs, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_freqs_path, "rb") as f:
            self.term_freqs = pickle.load(f)
            
    def get_documents(self, token: str) -> list[int]:
        return sorted(self.index.get(token, set()))
    
    def get_term_frequency(self, doc_id: int, token: str) -> int:
        return self.term_freqs.get(doc_id, Counter()).get(token, 0)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Inverted index not found. Please run 'build' command first.")
        return []
    seen, results = set(), []
    query_tokens = preprocess_text(query, idx.stopwords)
    for token in query_tokens:
        doc_ids = idx.get_documents(token)
        for doc_id in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(idx.docmap[doc_id])
            if len(results) >= limit:
                break
    return results

def tf_command(query:str, doc_id: int) -> int:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Inverted index not found. Please run 'build' command first.")
        return 0
    query_token = get_token(query, idx.stopwords)
    return idx.get_term_frequency(doc_id, query_token)

def idf_command(query: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Inverted index not found. Please run 'build' command first.")
        return 0.0
    query_token = get_token(query, idx.stopwords)
    num_docs_with_token = len(idx.get_documents(query_token))
    total_docs = len(idx.docmap)
    if num_docs_with_token == 0:
        return 0.0
    return math.log((total_docs + 1) / (num_docs_with_token + 1))

def tf_idf_command(query: str, doc_id: int) -> float:
    tf = tf_command(query, doc_id)
    idf = idf_command(query)
    return tf * idf

def bm25_idf_command(query: str) -> float:
    idx = InvertedIndex()
    try:        
        idx.load()
    except FileNotFoundError:
        print("Inverted index not found. Please run 'build' command first.")
        return 0.0
    query_token = get_token(query, idx.stopwords)
    num_docs_with_token = len(idx.get_documents(query_token))
    total_docs = len(idx.docmap)
    return math.log((total_docs - num_docs_with_token + 0.5) / (num_docs_with_token + 0.5) + 1)

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
    stemmer = PorterStemmer()
    tokens = set(stemmer.stem(token) for token in tokens)
    return tokens

def tokenize(text: str, stopwords: set[str]) -> list[str]:
    # Step one, make lowercase
    text = text.lower()
    # Step two, remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Step three, tokenize (split into words)
    tokens = text.split()
    # Step four, remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    # Step five, apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def get_token(token: str, stopwords: set[str]) -> str:
    tokens = preprocess_text(token, stopwords)
    if len(tokens) > 1:
        raise ValueError(f"Expected a single token, but got multiple: {tokens}")
    elif len(tokens) == 0:
        raise ValueError("Expected a single token, but got none")
    return tokens.pop()

# This function is no longer required now that I am using an inverted index, but I will keep it here for reference
def found(query_tokens: set[str], title_tokens: set[str]) -> bool:
    # Check if a word in the query is found as a substring in any of the words in the title
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False