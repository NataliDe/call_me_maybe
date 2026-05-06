"""Input/output helpers for the function-calling generator."""


import argparse
import json
import os
from typing import Any


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Translate prompts into structured function calls."
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the JSON file with function definitions.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file with input prompts.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path where the output JSON file will be written.",
    )

    return parser.parse_args()


def load_json_file(path: str) -> Any:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def save_results(output_path: str, results: list[dict[str, Any]]) -> None:
    """Save generated results to a JSON file.

    Args:
        output_path: Path where the output file should be written.
        results: Function-call results to save.
    """
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
