#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command, build_command, tf_command, idf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency of a word in a document")
    tf_parser.add_argument("doc_id", type=int, help="The ID of the document to check")
    tf_parser.add_argument("query", type=str, help="The word to check term frequency for")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency of a word")
    idf_parser.add_argument("query", type=str, help="The word to check inverse document frequency for")
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
        case _:
            parser.print_help()
            

if __name__ == "__main__":
    main()