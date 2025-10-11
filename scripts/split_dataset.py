"""
Split the full dataset into train/dev/test.
For ZipVoice inference, the test set is written in 4-column format.

Usage:
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O prompt1.wav
    uv run scripts/split_dataset.py dataset/raw/all.tsv dataset/raw/
"""

import argparse
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_tsv", nargs="?", default="dataset/raw/all.tsv")
parser.add_argument("output_dir", nargs="?", default="dataset/raw/")
args = parser.parse_args()

input_file = Path(args.input_tsv)
output_dir = Path(args.output_dir)
train_out = output_dir / "train.tsv"
dev_out = output_dir / "dev.tsv"
test_out = output_dir / "test.tsv"

train_ratio, dev_ratio = 0.94, 0.05  # remaining 1% goes to test
random.seed(42)

lines = input_file.read_text(encoding="utf-8").strip().splitlines()
random.shuffle(lines)

n = len(lines)
train_n = int(n * train_ratio)
dev_n = int(n * dev_ratio)

train_lines = lines[:train_n]
dev_lines = lines[train_n:train_n + dev_n]
test_lines = lines[train_n + dev_n:]

# write train & dev normally (3 columns)
for path, subset in [(train_out, train_lines), (dev_out, dev_lines)]:
    path.write_text("\n".join(subset) + "\n", encoding="utf-8")
    print(f"✅ Wrote {len(subset)} lines to {path}")

# make test in ZipVoice format (4 columns)
prompt_wav = "prompt1.wav"
prompt_text = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."

converted = []
for line in test_lines[:10]:  # 10 samples is enough for a quick inference eval
    parts = line.split("\t")
    if len(parts) < 3:
        continue
    uid, text, wav_path = parts
    converted.append(f"{uid}\t{prompt_text}\t{prompt_wav}\t{text}")

test_out.write_text("\n".join(converted) + "\n", encoding="utf-8")
print(f"🎯 Wrote {len(converted)} 4-column test lines to {test_out}")
