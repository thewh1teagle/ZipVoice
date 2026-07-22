#!/usr/bin/env python3
"""Convert the Hebrew IPA dataset (metadata.csv with `id|ipa_text` lines and a
wav/ directory) to ZipVoice finetune TSVs: `{uniq_id}\t{text}\t{wav_path}`.

The text is already espeak-style IPA with stress marks and punctuation, ready
for the raw_phoneme tokenizer.

Usage:
  python3 scripts/heb_to_tsv.py \
      --dataset-dir data/heb/heb-female-audio-ipa2 \
      --output-dir dataset/raw --prefix heb --dev-count 50
"""
import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="heb")
    parser.add_argument("--dev-count", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, only use the first N entries (for overfit tests)")
    args = parser.parse_args()

    entries = []
    with open(os.path.join(args.dataset_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            uid, text = line.rstrip("\n").split("|", 1)
            wav = os.path.abspath(os.path.join(args.dataset_dir, "wav", f"{uid}.wav"))
            if not os.path.isfile(wav):
                continue
            entries.append((f"{args.prefix}_{uid}", text.strip(), wav))

    if args.limit > 0:
        entries = entries[: args.limit]
        train, dev = entries, entries  # overfit: dev == train
    else:
        random.Random(42).shuffle(entries)
        dev, train = entries[: args.dev_count], entries[args.dev_count:]

    os.makedirs(args.output_dir, exist_ok=True)
    for name, subset in (("train", train), ("dev", dev)):
        path = os.path.join(args.output_dir, f"{name}.tsv")
        with open(path, "w", encoding="utf-8") as f:
            for uid, text, wav in subset:
                f.write(f"{uid}\t{text}\t{wav}\n")
        print(f"{path}: {len(subset)} entries")


if __name__ == "__main__":
    main()
