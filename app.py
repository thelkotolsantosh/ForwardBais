import argparse
from pathlib import Path

from rag import (
    build_index,
    retrieve,
    answer_with_ollama,
    DEFAULT_INDEX_DIR,
)


def cmd_index(args):
    source = Path(args.source)
    count = build_index(source, Path(args.index_dir))
    print(f"\nIndexed {count} chunks from: {source}")
    print(f"Vector index saved to: {args.index_dir}")


def cmd_ask(args):
    results = retrieve(
        query=args.question,
        index_dir=Path(args.index_dir),
        top_k=args.top_k,
    )

    if not results:
        print("No relevant context was found.")
        return

    print("\n--- Retrieved Context ---")
    for i, item in enumerate(results, start=1):
        print(f"\n[{i}] Score: {item['score']:.4f}")
        print(f"Source: {item['source']}")
        print(item["text"][:700])

    if args.no_llm:
        return

    print("\n--- Answer ---")
    answer = answer_with_ollama(
        question=args.question,
        contexts=results,
        model=args.model,
    )
    print(answer)


def main():
    parser = argparse.ArgumentParser(
        description="Forward Bias - a simple local RAG project"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index",
        help="Read files, create chunks and build the vector index.",
    )
    index_parser.add_argument(
        "--source",
        default="data",
        help="File or folder containing .txt, .md or .pdf files.",
    )
    index_parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="Directory used to save the FAISS index and metadata.",
    )
    index_parser.set_defaults(func=cmd_index)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Retrieve relevant chunks and optionally answer with Ollama.",
    )
    ask_parser.add_argument("question", help="Question to ask your documents.")
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Ollama model name.",
    )
    ask_parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
    )
    ask_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only show retrieved chunks. Do not call Ollama.",
    )
    ask_parser.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
