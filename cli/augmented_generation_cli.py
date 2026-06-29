import argparse

from lib.augmented_generation import rag_command, summarize_command, citations_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser = subparsers.add_parser(
        "summarize", help="Summarize a given text using RAG"
    )
    summary_parser.add_argument("query", type=str, help="Search query for summarization")
    summary_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )
    citations_parser = subparsers.add_parser(
        "citations", help="Generate citations for a given query using RAG"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(result["answer"])
        case "summarize":
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Summary:")
            print(result["summary"])
        case "citations":
            result = citations_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Answer:")
            print(result["citations"])
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()