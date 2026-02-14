import argparse
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle rows in a CSV file.")
    parser.add_argument("input", help="Path to input CSV")
    parser.add_argument("output", help="Path to output CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    df.to_csv(args.output, index=False)

    print(f"Shuffled {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
