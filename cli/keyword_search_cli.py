#!/usr/bin/env python3

import argparse
from lib.search_utils import BM25_K1, BM25_B
from lib.keyword_search import (
    search_command, 
    build_command, 
    tf_command, 
    idf_command, 
    tf_idf_command,
    bm25_idf_command,
    bm25_tf_command,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency of a word in a document")
    tf_parser.add_argument("doc_id", type=int, help="The ID of the document to check")
    tf_parser.add_argument("query", type=str, help="The word to check term frequency for")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency of a word")
    idf_parser.add_argument("query", type=str, help="The word to check inverse document frequency for")
    tf_idf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF of a word in a document")
    tf_idf_parser.add_argument("doc_id", type=int, help="The ID of the document to check")
    tf_idf_parser.add_argument("query", type=str, help="The word to check TF-IDF for")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF of a word")
    bm25_idf_parser.add_argument("query", type=str, help="The word to check BM25 IDF for")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF of a word in a document")
    bm25_tf_parser.add_argument("doc_id", type=int, help="The ID of the document to check")
    bm25_tf_parser.add_argument("query", type=str, help="The word to check BM25 TF for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="BM25 k1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="BM25 b parameter")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "tf":
            doc_id = int(args.doc_id)
            query = args.query
            tf = tf_command(query, doc_id)
            print(f"Term frequency of '{query}' in document {doc_id}: {tf}")
        case "idf":
            idf = idf_command(args.query)
            print(f"Inverse document frequency of '{args.query}': {idf:.2f}")
        case "tfidf":
            doc_id = int(args.doc_id)
            query = args.query
            tf_idf = tf_idf_command(query, doc_id)
            print(f"TF-IDF of '{query}' in document {doc_id}: {tf_idf:.2f}")
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.query)
            print(f"BM25 IDF score of '{args.query}': {bm25_idf:.2f}")
        case "bm25tf":
            doc_id = int(args.doc_id)
            query = args.query
            k1 = args.k1
            b = args.b
            bm25_tf = bm25_tf_command(query, doc_id, k1=k1, b=b)
            print(f"BM25 TF score of '{query}' in document {doc_id}: {bm25_tf:.2f}")
        case _:
            parser.print_help()
            

if __name__ == "__main__":
    main()