import argparse

from lib.multimodal_search import (
    verify_image_embedding,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify that an image can be embedded"
    )
    verify_image_parser.add_argument(
        "image_path", type=str, help="Path to the image file to verify"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            try:
                verify_image_embedding(args.image_path)
                print(f"Image embedding verification successful for: {args.image_path}")
            except Exception as e:
                print(f"Image embedding verification failed for: {args.image_path}. Error: {e}")
        case _:
            parser.print_help()
            
if __name__ == "__main__":
    main()