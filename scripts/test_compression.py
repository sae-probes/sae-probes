#!/usr/bin/env python3
"""
CSV Compression Comparison Script

This script takes all .csv files in a directory and converts them to different
compressed formats (Parquet, Gzipped CSV, Feather, HDF5) to compare compression ratios.
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd


def get_file_size(filepath):
    """Get file size in bytes"""
    return os.path.getsize(filepath)


def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def convert_csv_files(input_dir, output_base_dir="compressed_outputs"):
    """Convert all CSV files to different formats"""

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return

    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # Create output directories
    formats = {
        "parquet": ".parquet",
        "gzipped_csv": ".csv.gz",
        "zipped_csv": ".csv.zip",
        "zstd_csv": ".csv.zst",
        # "xz_csv": ".csv.xz",
        "feather": ".feather",
        # "hdf5": ".h5",
    }

    output_dirs = {}
    for format_name in formats.keys():
        output_dirs[format_name] = Path(output_base_dir) / format_name
        output_dirs[format_name].mkdir(parents=True, exist_ok=True)

    # Track compression results
    results = []

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        try:
            # Read the CSV file
            start_time = time.time()
            df = pd.read_csv(csv_file)
            read_time = time.time() - start_time

            original_size = get_file_size(csv_file)
            print(f"  Original size: {format_size(original_size)}")
            print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")

            file_results = {
                "filename": csv_file.name,
                "original_size": original_size,
                "read_time": read_time,
            }

            # Convert to each format
            for format_name, extension in formats.items():
                output_file = output_dirs[format_name] / (csv_file.stem + extension)

                try:
                    start_time = time.time()

                    if format_name == "parquet":
                        df.to_parquet(output_file, index=False)
                    elif format_name == "gzipped_csv":
                        df.to_csv(output_file, compression="zstd", index=False)
                    elif format_name == "zipped_csv":
                        df.to_csv(output_file, compression="zip", index=False)
                    elif format_name == "zstd_csv":
                        df.to_csv(
                            output_file,
                            compression={"method": "zstd", "level": 15},
                            index=False,
                        )
                    elif format_name == "xz_csv":
                        df.to_csv(output_file, compression="xz", index=False)
                    elif format_name == "feather":
                        df.to_feather(output_file)

                    write_time = time.time() - start_time
                    compressed_size = get_file_size(output_file)
                    compression_ratio = original_size / compressed_size
                    space_saved = (1 - compressed_size / original_size) * 100

                    print(
                        f"  {format_name.upper()}: {format_size(compressed_size)} "
                        f"({compression_ratio:.2f}x smaller, {space_saved:.1f}% saved)"
                    )

                    file_results[f"{format_name}_size"] = compressed_size
                    file_results[f"{format_name}_time"] = write_time
                    file_results[f"{format_name}_ratio"] = compression_ratio
                    file_results[f"{format_name}_saved"] = space_saved

                except Exception as e:
                    print(f"  ERROR with {format_name}: {str(e)}")
                    file_results[f"{format_name}_size"] = None
                    file_results[f"{format_name}_time"] = None
                    file_results[f"{format_name}_ratio"] = None
                    file_results[f"{format_name}_saved"] = None

            results.append(file_results)

        except Exception as e:
            print(f"  ERROR reading {csv_file.name}: {str(e)}")
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("COMPRESSION SUMMARY")
    print("=" * 80)

    if results:
        # Calculate averages
        total_original = sum(r["original_size"] for r in results)

        for format_name in formats.keys():
            sizes = [
                r[f"{format_name}_size"]
                for r in results
                if r[f"{format_name}_size"] is not None
            ]
            if sizes:
                total_compressed = sum(sizes)
                avg_ratio = total_original / total_compressed
                avg_saved = (1 - total_compressed / total_original) * 100

                print(f"\n{format_name.upper()}:")
                print(f"  Total size: {format_size(total_compressed)}")
                print(f"  Overall compression: {avg_ratio:.2f}x smaller")
                print(f"  Space saved: {avg_saved:.1f}%")

        # Save detailed results to CSV
        results_df = pd.DataFrame(results)
        results_file = Path(output_base_dir) / "compression_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

    print(f"\nOutput directories created in: {Path(output_base_dir).absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Compare CSV compression formats")
    parser.add_argument("input_dir", help="Directory containing CSV files")
    parser.add_argument(
        "-o",
        "--output",
        default="compressed_outputs",
        help="Base output directory (default: compressed_outputs)",
    )

    args = parser.parse_args()
    convert_csv_files(args.input_dir, args.output)


if __name__ == "__main__":
    # If no command line args, use current directory
    import sys

    if len(sys.argv) == 1:
        convert_csv_files(".")
    else:
        main()
