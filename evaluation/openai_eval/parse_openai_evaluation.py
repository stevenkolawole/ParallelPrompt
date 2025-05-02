import json
import argparse
import os


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Parse evaluation results and compute statistics."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the evaluation results JSON file.",
    )
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Input file '{args.input}' not found.")
        return

    # Initialize counters for the "serial" and "parallel" wins and ties
    stats = {
        "accuracy": {"serial": 0, "parallel": 0, "tie": 0},
        "grammar": {"serial": 0, "parallel": 0, "tie": 0},
        "detail": {"serial": 0, "parallel": 0, "tie": 0},
        "preference": {"serial": 0, "parallel": 0, "tie": 0},
    }

    # Read the evaluation results from the JSON file
    with open(args.input, "r") as infile:
        data = json.load(infile)

    # Analyze each prompt and update statistics
    for result in data:
        evaluation = result["evaluation"]

        # Update statistics for each category
        for category in ["accuracy", "grammar", "detail", "preference"]:
            score = evaluation.get(category)
            if score == 1:
                stats[category]["serial"] += 1
            elif score == 2:
                stats[category]["parallel"] += 1
            else:
                stats[category]["tie"] += 1

    # Print the statistics
    print("Evaluation Statistics:")
    for category, results in stats.items():
        print(f"\n{category.capitalize()} Results:")
        print(f"Serial wins: {results['serial']}")
        print(f"Parallel wins: {results['parallel']}")
        print(f"Ties: {results['tie']}")


if __name__ == "__main__":
    main()
