"""
Convert LJSpeech-format dataset to ZipVoice TSV format.
Expects metadata.csv with id|phonemes columns and wav/ directory.

Usage:
    uv run scripts/ljspeech_to_tsv.py dataset-raw/ dataset/raw/all.tsv
"""

import argparse
import csv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Convert SASpeech-style metadata to ZipVoice TSV format."
    )
    parser.add_argument("input_dir", type=str, help="Input dataset directory (contains metadata.csv and wav/)")
    parser.add_argument("output_tsv", type=str, help="Output TSV file path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_tsv = Path(args.output_tsv).resolve()

    metadata_path = input_dir / "metadata.csv"
    wav_dir = input_dir / "wav"

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {metadata_path}")
    print(f"Writing: {output_tsv}")

    with metadata_path.open("r", encoding="utf-8") as fin, output_tsv.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="|")
        for row in reader:
            # expected: id, phonemes
            if len(row) < 2:
                continue
            uid, phonemes = row[0].strip(), row[1].strip()

            # handle ids that already include .wav extension
            fname = uid if uid.endswith(".wav") else f"{uid}.wav"
            wav_path = (wav_dir / fname).resolve()
            fout.write(f"{uid}\t{phonemes}\t{wav_path}\n")

    print(f"✅ Done! Wrote TSV with wav paths to: {output_tsv}")

if __name__ == "__main__":
    main()
