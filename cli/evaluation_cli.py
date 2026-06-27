import argparse

from lib.evaluation import evaluate_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",        
    )

    args = parser.parse_args()
    results = evaluate_command(args.limit)

    print(f"k={args.limit}\n")
    for query, result in results["results"].items():
        print(f"- Query: {query}")
        print(f"  - Precision@{args.limit}: {result['precision']:.4f}")
        print(f"  - Recall@{args.limit}: {result['recall']:.4f}")
        print(f"  - F1 Score: {result['f1_score']:.4f}")
        print(f"  - Retrieved: {', '.join(result['retrieved'])}")
        print(f"  - Relevant: {', '.join(result['relevant'])}\n")

if __name__ == "__main__":
    main()