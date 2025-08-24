#!/usr/bin/env python3
"""
Script to process CSV files from raw_data directory and save them as gzipped CSVs
in sae_probes/data directory with the same relative paths.

If a CSV contains a 'prompt' field, truncates entries to 100k characters.
"""

from pathlib import Path

import pandas as pd


def truncate_prompt_field(df: pd.DataFrame, max_length: int = 8192) -> pd.DataFrame:
    """
    Truncate the 'prompt' field to max_length characters if it exists.

    Args:
        df: DataFrame to process
        max_length: Maximum length for prompt field

    Returns:
        DataFrame with truncated prompt field if applicable
    """
    if "prompt" in df.columns:
        df = df.copy()
        df["prompt"] = df["prompt"].astype(str).str[:max_length]
        print(f"  Truncated 'prompt' field to {max_length} characters")
    return df


def process_csv_file(input_path: Path, output_path: Path) -> None:
    """
    Process a single CSV file: read, truncate prompts if needed, save as gzipped CSV.

    Args:
        input_path: Path to input CSV file
        output_path: Path where output .csv.zst file should be saved
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        print(f"Processing {input_path} -> {output_path}")
        print(f"  Shape: {df.shape}")

        # Truncate prompt field if present
        df = truncate_prompt_field(df)

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as gzipped CSV with proper quoting to handle embedded newlines
        df.to_csv(
            output_path,
            compression={"method": "zstd", "level": 17},
            index=False,
            quoting=1,  # csv.QUOTE_ALL - quote all fields to handle embedded newlines
            lineterminator="\n",  # Ensure consistent line endings
        )
        print(f"  Saved as gzipped CSV: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def find_csv_files(directory: Path) -> list[Path]:
    """
    Recursively find all CSV files in a directory.

    Args:
        directory: Directory to search

    Returns:
        List of paths to CSV files
    """
    csv_files = []
    for file_path in directory.rglob("*.csv"):
        csv_files.append(file_path)
    return sorted(csv_files)


def main() -> None:
    """Main function to process all CSV files from raw_data to sae_probes/data."""
    # Define base directories
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    raw_data_dir = project_root / "raw_data"
    output_base_dir = project_root / "sae_probes" / "data"

    print(f"Processing CSV files from: {raw_data_dir}")
    print(f"Output directory: {output_base_dir}")
    print()

    # Find all CSV files
    csv_files = find_csv_files(raw_data_dir)
    print(f"Found {len(csv_files)} CSV files to process")
    print()

    # Process each CSV file
    for csv_file in csv_files:
        # Calculate relative path from raw_data_dir
        relative_path = csv_file.relative_to(raw_data_dir)

        # Create output path with .csv.zst extension
        output_path = output_base_dir / relative_path.with_suffix(".csv.zst")

        # Process the file
        process_csv_file(csv_file, output_path)
        print()

    print(f"Completed processing {len(csv_files)} CSV files")


if __name__ == "__main__":
    main()
