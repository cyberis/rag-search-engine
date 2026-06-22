import os
import re
import json
from typing import TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR, 
    DEFAULT_SEARCH_LIMIT, 
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
    load_movies,
    Movie
)

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        chunks = []
        metadata = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            if len(doc["description"]) == 0:
                continue
            doc_chunks = semantic_chunk_text(doc["description"], DEFAULT_MAX_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            chunks.extend(doc_chunks)
            for i in range(len(doc_chunks)):
                metadata.append(ChunkMetadata(movie_idx=doc["id"], chunk_idx=i, total_chunks=len(doc_chunks)))

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                metadata_json = json.load(f)
                chunk_metadata = metadata_json.get("chunks", [])
                total_chunks = metadata_json.get("total_chunks", 0)
                if len(chunk_embeddings) == total_chunks == len(chunk_metadata):
                    self.chunk_embeddings = chunk_embeddings
                    self.chunk_metadata = [ChunkMetadata(**md) for md in chunk_metadata]
                    return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


#### CLI Commands ###

def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


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


def chunk_text(text, chunk_size=DEFAULT_MAX_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def semantic_chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for i in range(0, len(sentences), chunk_size - chunk_overlap):
        chunk = " ".join(sentences[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(sentences):
            break
    return chunks


#### Utility Functions ####

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
