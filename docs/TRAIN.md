# RecFM Training

This note documents the current ZipVoice RecFM training setup used in this
repo. For the model objective and architecture idea, see `docs/RFM.md`.

## Environment

Use `uv` from the project root.

On ARM CUDA, install the local k2 wheel before training:

```bash
uv pip install -r requirements-arm-cuda.txt
```

Verify k2 is active:

```bash
uv run python -c "import k2, torch; print(getattr(k2, '__dev_version__', 'k2-ok')); print(torch.__version__)"
```

k2 matters for training. Without it, ZipVoice falls back to local Swoosh
activation code that can produce finite forward losses but NaN gradients.

## Inputs

Current run uses:

- base checkpoint: `checkpoint-36600.pt`
- model config: `download/zipvoice/model.json`
- tokens: `download/zipvoice/tokens.txt`
- train manifest: `data/fbank/synth_recfm_cuts_train.jsonl.gz`
- dev manifest: `data/fbank/synth_recfm_cuts_dev.jsonl.gz`

The prepared dataset mix is about 85% Hebrew and 15% English from the same
speaker. Audio is prepared as Vocos fbank features for ZipVoice.

## Data Preparation

The source dataset is `synthetic-multilingual-speech/`. The RecFM run uses the
single-speaker filelists:

- Hebrew train: `synthetic-multilingual-speech/filelists/he_single_speaker.train.txt`
- Hebrew dev: `synthetic-multilingual-speech/filelists/he_single_speaker.val.txt`
- English train: `synthetic-multilingual-speech/filelists/en_single_speaker.train.txt`
- English dev: `synthetic-multilingual-speech/filelists/en_single_speaker.val.txt`

Those filelists are pipe-separated metadata:

```text
wav_path|text
```

The generated ZipVoice TSVs are tab-separated:

```text
utt_id<TAB>text<TAB>absolute_wav_path
```

For this run, English rows were lowercased to match the base ZipVoice token
file, then truncated so English is 15% of each split. The generated counts were:

- train: 10,743 Hebrew + 1,896 English = 12,639 rows
- dev: 500 Hebrew + 88 English = 588 rows

The resulting TSVs are:

- `data/raw/synth_recfm_train.tsv`
- `data/raw/synth_recfm_dev.tsv`

The exact TSV generation logic was:

```python
from pathlib import Path

root = Path("/home/yakov/Documents/audio/zipvoice")
src_root = root / "synthetic-multilingual-speech"
out_dir = root / "data" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)

splits = {
    "train": {
        "he": src_root / "filelists" / "he_single_speaker.train.txt",
        "en": src_root / "filelists" / "en_single_speaker.train.txt",
    },
    "dev": {
        "he": src_root / "filelists" / "he_single_speaker.val.txt",
        "en": src_root / "filelists" / "en_single_speaker.val.txt",
    },
}

def read_rows(path: Path, lang: str):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            wav, text = line.split("|", 1)
            if wav.startswith("data/synthetic-multilingual-speech/"):
                wav_path = src_root / Path(wav).relative_to(
                    "data/synthetic-multilingual-speech"
                )
            else:
                wav_path = root / wav
            text = text.replace("\t", " ").strip()
            if lang == "en":
                text = text.lower()
            rows.append((wav_path, text, lang))
    return rows

for subset, paths in splits.items():
    he_rows = read_rows(paths["he"], "he")
    en_rows = read_rows(paths["en"], "en")
    en_count = round(len(he_rows) * 0.15 / 0.85)
    rows = he_rows + en_rows[:en_count]
    with (out_dir / f"synth_recfm_{subset}.tsv").open("w", encoding="utf-8") as out:
        for idx, (wav_path, text, lang) in enumerate(rows, start=1):
            utt_id = f"{subset}_{idx:06d}_{lang}_{wav_path.stem}"
            out.write(f"{utt_id}\t{text}\t{wav_path}\n")
```

After generating TSVs, convert audio metadata to Lhotse manifests at 24 kHz:

```bash
for subset in train dev; do
  uv run python3 -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/synth_recfm_${subset}.tsv \
    --prefix synth_recfm \
    --subset ${subset} \
    --num-jobs 20 \
    --sampling-rate 24000 \
    --output-dir data/manifests
done
```

Tokenize with the same raw phoneme tokenizer and base ZipVoice token file:

```bash
for subset in train dev; do
  uv run python3 -m zipvoice.bin.prepare_tokens \
    --input-file data/manifests/synth_recfm_cuts_${subset}.jsonl.gz \
    --output-file data/manifests/synth_recfm_cuts_${subset}_tok.jsonl.gz \
    --num-jobs 20 \
    --tokenizer raw_phoneme \
    --lang default

  cp -f \
    data/manifests/synth_recfm_cuts_${subset}_tok.jsonl.gz \
    data/manifests/synth_recfm_cuts_${subset}.jsonl.gz
done
```

Finally compute Vocos fbanks:

```bash
for subset in train dev; do
  uv run python3 -m zipvoice.bin.compute_fbank \
    --source-dir data/manifests \
    --dest-dir data/fbank \
    --dataset synth_recfm \
    --subset ${subset} \
    --sampling-rate 24000 \
    --num-jobs 20
done
```

The final training manifests are:

- `data/fbank/synth_recfm_cuts_train.jsonl.gz`
- `data/fbank/synth_recfm_cuts_dev.jsonl.gz`

## Training Command

The active fp16 recipe is:

```bash
uv run python3 -m zipvoice.bin.train_zipvoice_recfm \
  --world-size 1 \
  --use-fp16 1 \
  --finetune 1 \
  --checkpoint checkpoint-36600.pt \
  --base-lr 1e-5 \
  --num-iters 10000 \
  --save-every-n 1000 \
  --keep-last-k 5 \
  --max-duration 200 \
  --max-len 20 \
  --num-buckets 10 \
  --model-config download/zipvoice/model.json \
  --start-epoch 1 \
  --tokenizer raw_phoneme \
  --lang default \
  --token-file download/zipvoice/tokens.txt \
  --dataset custom \
  --train-manifest data/fbank/synth_recfm_cuts_train.jsonl.gz \
  --dev-manifest data/fbank/synth_recfm_cuts_dev.jsonl.gz \
  --recfm-lambda 1.0 \
  --exp-dir exp/zipvoice_recfm_from_36600_k2_fp16
```

This trains only the flow decoder by default. The text encoder and embedding
stack stay frozen. Use `--train-full-model 1` only if decoder-only adaptation is
not enough.

## Monitoring

The log for the current tmux run is:

```bash
tail -f /tmp/zipvoice_recfm_train_k2_fp16.log
```

Healthy fp16 signs:

- `grad_scale` stays finite, usually `65536.0`
- ScaledAdam logs finite grad-norm quartiles
- no `bad-model-*.pt` files are written
- validation and train losses are finite

The train script now updates tqdm postfix for new runs with current iter, batch
size, losses, learning rate, and grad scale. An already-running process keeps
the older stale tqdm display until restarted.

## Checkpoints

There are two checkpoint paths:

- epoch checkpoints, for example `epoch-1.pt`
- global-batch checkpoints every `--save-every-n`, for example
  `checkpoint-1000.pt`

For this prepared dataset, one epoch is roughly a few hundred dynamic batches.
The `--save-every-n 1000` checkpoint comes later than the first epoch save.

## Inference Smoke Test

After a usable RecFM checkpoint exists, test 2-step inference with:

```bash
uv run python3 -m zipvoice.bin.infer_zipvoice \
  --model-name zipvoice_recfm \
  --model-dir exp/zipvoice_recfm_from_36600_k2_fp16 \
  --checkpoint-name epoch-1.pt \
  --tokenizer raw_phoneme \
  --lang default \
  --num-step 2 \
  --guidance-scale 1.0 \
  --prompt-wav <prompt.wav> \
  --prompt-text "<prompt text>" \
  --text "<target text>" \
  --res-wav-path results/recfm-2step.wav
```

One epoch is mainly a smoke test. Expect a more meaningful quality read after
several epochs or closer to the configured `10000` iterations.
