import argparse

from lib.hybrid_search import normalize_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_command(args.scores)
            if len(normalized_scores) > 0:
                for score in normalized_scores:
                    print(f"{score:.4f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()