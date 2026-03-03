"""CLI: Generate synthetic data."""

import argparse
from swaadstack.config import DataConfig
from swaadstack.data.pipeline import run_data_generation_pipeline


def main():
    parser = argparse.ArgumentParser(description="SwaadStack AI - Data Generator")
    parser.add_argument("--sessions", type=int, default=5000, help="Number of sessions")
    parser.add_argument("--users", type=int, default=500, help="Number of users")
    parser.add_argument("--no-bert", action="store_true", help="Skip Sentence-BERT, use random embeddings")
    args = parser.parse_args()

    config = DataConfig(num_sessions=args.sessions, num_users=args.users)
    run_data_generation_pipeline(config, use_sentence_bert=not args.no_bert)


if __name__ == "__main__":
    main()
