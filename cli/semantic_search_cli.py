#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()