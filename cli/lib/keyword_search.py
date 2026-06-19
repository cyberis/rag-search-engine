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
    BM25_K1,
    BM25_B,
)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.term_freqs: dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freqs_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.doc_lengths: dict[int, int] = {}
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
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_freqs[doc_id].update(tokens)  # This is a shortcut I discovered in the instructor solution.
        self.doc_lengths[doc_id] = len(tokens)
            
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
                
    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_freqs_path, "wb") as f:
            pickle.dump(self.term_freqs, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
    
    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_freqs_path, "rb") as f:
            self.term_freqs = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
            
    def get_documents(self, token: str) -> list[int]:
        return sorted(self.index.get(token, set()))
    
    def get_term_frequency(self, doc_id: int, token: str) -> int:
        return self.term_freqs.get(doc_id, Counter()).get(token, 0)
    
    def get_idf(self, token: str) -> float:
        num_docs_with_token = len(self.get_documents(token))
        total_docs = len(self.docmap)
        if num_docs_with_token == 0:
            return 0.0
        return math.log((total_docs + 1) / (num_docs_with_token + 1))
    
    def get_bm25_idf(self, token: str) -> float:
        num_docs_with_token = len(self.get_documents(token))
        total_docs = len(self.docmap)
        return math.log((total_docs - num_docs_with_token + 0.5) / (num_docs_with_token + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_term_frequency(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_term_frequency(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def bm25(self, doc_id: int, token: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        bm25_idf = self.get_bm25_idf(token)
        bm25_tf = self.get_bm25_tf(doc_id, token, k1, b)
        return bm25_idf * bm25_tf
    
    def bm25_search(self, query: str, k1: float = BM25_K1, b: float = BM25_B) -> list[tuple[int, float]]:
        query_tokens = tokenize(query, self.stopwords)
        scores: dict[int, float] = {}
        for doc_id in self.docmap.keys():
            scores[doc_id] = 0.0
        for query_token in query_tokens:
            for doc_id in self.docmap.keys():
                scores[doc_id] += self.bm25(doc_id, query_token, k1, b)
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        return list(scores.items())[0:DEFAULT_SEARCH_LIMIT - 1]

#### CLI Commands ####

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

def tf_command(doc_id: int, query: str) -> int:
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
    return idx.get_idf(query_token)

def tf_idf_command(query: str, doc_id: int) -> float:
    tf = tf_command(doc_id, query)
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
    return idx.get_bm25_idf(query_token)

def bm25_tf_command(query: str, doc_id: int, k1=BM25_K1, b=BM25_B) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Inverted index not found. Please run 'build' command first.")
        return 0.0
    query_token = get_token(query, idx.stopwords)
    return idx.get_bm25_tf(doc_id, query_token, k1, b)

def bm25_search_command(query: str, k1=BM25_K1, b=BM25_B) -> list[tuple[int, str, float]]:
    results = []
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Inverted index not found. Please run 'build' command first.")
        return results
    scores = idx.bm25_search(query, k1, b)
    for doc_id, score in scores:
        results.append((doc_id, idx.docmap[doc_id]['title'], score))
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