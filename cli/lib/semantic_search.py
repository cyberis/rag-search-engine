from sentence_transformers import SentenceTransformer
import numpy as np
from torch import Tensor
from typing import Any, TypedDict
import os
import json

DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings: Tensor = Tensor([])
        self.documents: list[Movie] = []
        self.document_map: dict[int, Movie] = {}
        
    def generate_embedding(self, text: str) -> Tensor:
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        return self.model.encode(text)
    
    def build_embeddings(self, documents: list[Movie]) -> Tensor:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        texts = [f"{doc['title']} {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[Movie]) -> Tensor:
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            self.documents = documents
            self.document_map = {doc["id"]: doc for doc in documents}
            return self.embeddings
        else:
            return self.build_embeddings(documents)

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            qe_array = query_embedding.detach().cpu().numpy()
            de_array = doc_embedding.detach().cpu().numpy()
            similarity = cosine_similarity(qe_array, de_array)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results



#### CLI Commands ####
def verify_model() -> None:
    try:
        search = SemanticSearch()
        print(f"Model loaded: {search.model}")
        print(f"Max sequence length: {search.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
        
def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding =  search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    search = SemanticSearch()
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    movies: list[Movie] = data["movies"]
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of documents: {len(search.documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    
def embed_query_text(text: str) -> None:
    search = SemanticSearch()
    embedding =  search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    
def semantic_search(query, limit=DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()
    
#### Utility Functions ####

def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]
    
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)