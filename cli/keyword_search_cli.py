#!/usr/bin/env python3

import argparse
import json

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
             search_movies(args.query)
        case _:
            parser.print_help()
            
def search_movies(query: str) -> None:
    # Placeholder for the actual search logic
    print(f"Searching for: {query}\n")
    with open("data/movies.json", "r") as f:    
        movies = json.load(f)
        results = [movie for movie in movies["movies"] if query.lower() in movie["title"].lower()]
        print(f"Found {len(results)} results:") # This is an optional line no in the lesson specifications
        for i, movie in enumerate(results, start=1):
            print(f"{i}. {movie['title']}")
            if i >= 5:  # Limit to top 5 results
                break

if __name__ == "__main__":
    main()