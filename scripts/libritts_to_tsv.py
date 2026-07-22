#!/usr/bin/env python3
"""Convert a LibriTTS-R subset to ZipVoice finetune TSVs with espeak IPA text.

Walks e.g. data/LibriTTS_R/train-clean-100 for *.wav with matching
*.normalized.txt transcripts, phonemizes with espeak-ng (en-us) keeping
punctuation and stress marks, and writes `{uniq_id}\t{ipa_text}\t{wav_path}`
lines compatible with the raw_phoneme tokenizer, so English and Hebrew IPA
share one token space.

Usage:
  python3 scripts/libritts_to_tsv.py \
      --subset-dir data/LibriTTS_R/train-clean-100 \
      --output-tsv dataset/raw/libritts_train.tsv --num-jobs 16
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import reduce

from piper_phonemize import phonemize_espeak


def phonemize_one(item):
    uid, text, wav = item
    try:
        tokens = phonemize_espeak(text, "en-us")
        ipa = "".join(reduce(lambda x, y: x + y, tokens))
        return (uid, ipa, wav)
    except Exception as e:
        print(f"skip {uid}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--num-jobs", type=int, default=16)
    args = parser.parse_args()

    items = []
    for root, _, files in os.walk(args.subset_dir):
        for fn in sorted(files):
            if not fn.endswith(".wav"):
                continue
            wav = os.path.abspath(os.path.join(root, fn))
            txt = wav[:-4] + ".normalized.txt"
            if not os.path.isfile(txt):
                continue
            with open(txt, encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                items.append((fn[:-4], text, wav))

    print(f"phonemizing {len(items)} utterances...")
    with ProcessPoolExecutor(max_workers=args.num_jobs) as ex:
        results = list(ex.map(phonemize_one, items, chunksize=64))
    results = [r for r in results if r is not None]

    os.makedirs(os.path.dirname(args.output_tsv), exist_ok=True)
    with open(args.output_tsv, "w", encoding="utf-8") as f:
        for uid, ipa, wav in results:
            f.write(f"{uid}\t{ipa}\t{wav}\n")
    print(f"{args.output_tsv}: {len(results)} entries")


if __name__ == "__main__":
    main()
