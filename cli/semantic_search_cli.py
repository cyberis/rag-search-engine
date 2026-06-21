#!/usr/bin/env python3

import argparse
from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
)
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
    chunk_text,
    semantic_chunk_text,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the semantic search model")
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    subparsers.add_parser("embed_text", help="Generate embedding for input text").add_argument("text", type=str, help="Text to embed")
    subparsers.add_parser("embed_query", help="Generate embedding for query text").add_argument("text", type=str, help="Query text to embed")
    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    chunk_parser = subparsers.add_parser("chunk", help="Chunk input text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Number of words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Number of words to overlap between chunks")   
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk input text into smaller pieces based on sentences")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Number of sentences to overlap between chunks")   
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_MAX_CHUNK_SIZE, help="Maximum number of sentences per chunk")
    args = parser.parse_args()
    
    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.text)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "semantic_chunk":
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()