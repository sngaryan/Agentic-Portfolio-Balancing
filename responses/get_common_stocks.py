import argparse
import json
import os
from typing import Set


def get_keys_from_json(filepath: str) -> Set[str]:
    """Extract all keys from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
        # If data is a dictionary, return its keys
        if isinstance(data, dict):
            return set(data.keys())
        # If data is a list of dictionaries, get unique keys from all dictionaries
        elif isinstance(data, list):
            all_keys = set()
            for item in data:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            return all_keys
    return set()


def main():
    parser = argparse.ArgumentParser(
        description="Verify or extract common stock tickers across model response JSON files."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help="Explicitly write the extracted list to common_stocks.json. Default is check-only.",
    )
    args = parser.parse_args()

    # Resolve directory relative to script location so it runs safely from any working directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Scan only intended model response JSON files; explicitly exclude output files
    json_files = sorted([
        f
        for f in os.listdir(base_dir)
        if f.endswith(".json")
        and f != "common_stocks.json"
        and f.startswith(("gemma_", "llama_", "qwen_"))
    ])

    if not json_files:
        raise FileNotFoundError(f"No valid model response JSON files found in {base_dir}")

    print(f"Scanning {len(json_files)} model response files in {base_dir}...")

    # Extract keys from all files
    all_keys = None
    for json_file in json_files:
        file_path = os.path.join(base_dir, json_file)
        file_keys = get_keys_from_json(file_path)
        if not file_keys:
            raise ValueError(f"Extracted 0 keys from {json_file}. Aborting.")
        if all_keys is None:
            all_keys = file_keys
        else:
            all_keys = all_keys.intersection(file_keys)

    # Strict validation check
    if not all_keys or len(all_keys) < 100:
        raise ValueError(
            f"Intersection produced suspiciously few keys ({len(all_keys) if all_keys else 0}). Aborting."
        )

    output_path = os.path.join(base_dir, "common_stocks.json")
    existing_keys = None
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_keys = set(existing_data)

    print(f"Computed common stock intersection count: {len(all_keys)}")
    if existing_keys is not None:
        matches = all_keys == existing_keys
        print(f"Existing common_stocks.json ticker count: {len(existing_keys)}")
        print(f"Does computed intersection match existing list set? {matches}")

    if args.write:
        sorted_keys = sorted(list(all_keys))
        with open(output_path, "w") as f:
            json.dump(sorted_keys, f)
        print(f"Wrote {len(sorted_keys)} common stocks to {output_path}")
    else:
        print("Check completed successfully. File not rewritten (use --write to overwrite).")


if __name__ == "__main__":
    main()
