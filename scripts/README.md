# ZipVoice Hebrew Fine-tune

## Setup

```bash
git clone https://github.com/thewh1teagle/ZipVoice -b feat/hebrew-finetune
uv sync
```

## Prepare Dataset

Dataset is in LJSpeech format: `metadata.csv` with `id|phonemes` columns and `wav/0.wav`, `wav/1.wav`, etc.

```bash
uv run huggingface-cli download --repo-type dataset thewh1teagle/hebrew-tts-dataset heb-female-audio-ipa2-v2.7z --local-dir .
7z x heb-female-audio-ipa2-v2.7z
mv heb-female-audio-ipa2-v2 dataset-raw
uv run scripts/ljspeech_to_tsv.py dataset-raw/ dataset/raw/all.tsv
uv run scripts/split_dataset.py
```

### Mixing English (optional)

To mix in English data (recommended ~20-30% to prevent catastrophic forgetting of English phonemes while keeping Hebrew accent dominant), use the `ljspeech-enhanced` dataset (~10h, same LJSpeech format but with 4 columns: `id|phonemes|speaker_id|text`):

```bash
wget "https://huggingface.co/datasets/thewh1teagle/ljspeech-enhanced/resolve/main/ljspeech-enhanced-v1.7z" -O ljspeech-enhanced-v1.7z
7z x ljspeech-enhanced-v1.7z
mv ljspeech-enhanced-v1 dataset-raw-en

# Strip extra columns (id|phonemes|speaker_id|text -> id|phonemes) then convert
uv run python -c "
import csv, pathlib
with open('dataset-raw-en/metadata.csv') as fin, open('dataset-raw-en/metadata2.csv', 'w') as fout:
    for row in csv.reader(fin, delimiter='|'):
        if len(row) >= 2:
            fout.write(row[0] + '|' + row[1] + '\n')
"
mv dataset-raw-en/metadata2.csv dataset-raw-en/metadata.csv
uv run scripts/ljspeech_to_tsv.py dataset-raw-en/ dataset/raw/all_en.tsv

# Concatenate (adjust ratio by using head -N to limit English lines)
cat dataset/raw/all.tsv dataset/raw/all_en.tsv > dataset/raw/all_mixed.tsv
uv run scripts/split_dataset.py dataset/raw/all_mixed.tsv dataset/raw/
```

## Train

> **Important:** Training requires `k2` to be installed and importable.
> To find the right wheel for your environment:
> ```bash
> uv run scripts/find_k2.py
> ```
> Then install the printed wheel with `uv pip install <wheel>`.
> Verify it works with: `uv run python -c "import k2; print(k2.__dev_version__)"`

```bash
./egs/zipvoice/run_finetune.sh
```

Monitor training:
```bash
uv run tensorboard --logdir ./exp/zipvoice_finetune/
```

Monitor GPU:
```bash
uv pip install nvitop
uv run nvitop
```

## Infer (PyTorch)

```bash
wget https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/prompt_hebrew_male1.wav -O prompt.wav

uv run python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --model-dir exp/zipvoice_finetune/ \
    --checkpoint-name epoch-1.pt \
    --tokenizer raw_phoneme \
    --prompt-wav prompt.wav \
    --prompt-text "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan." \
    --text "simˈu lˈev noseʔˈim jekaʁˈim, haʁakˈevet tikanˈes letaχanˈat tˈel ʔavˈiv meʁkˈaz beʔˈod mispˈaʁ dakˈot, ʔˈana hitʁaχakˈu miktsˈe haʁatsˈif vehamtˈinu meʔaχoʁˈej hakˈav hatsahˈov, todˈa." \
    --res-wav-path result.wav \
    --num-step 16
```

To run on the test set:
```bash
uv run python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --model-dir exp/zipvoice_finetune/ \
    --checkpoint-name epoch-1.pt \
    --tokenizer raw_phoneme \
    --test-list dataset/raw/test.tsv \
    --res-dir results/test_finetune \
    --num-step 16
```

## Export ONNX & Upload

```bash
uv pip install onnx onnxruntime
uv run python3 -m zipvoice.bin.onnx_export \
    --model-name zipvoice \
    --model-dir exp/zipvoice_finetune \
    --checkpoint-name epoch-51.pt \
    --onnx-model-dir exp/zipvoice-onnx
cp exp/zipvoice_finetune/{model.json,tokens.txt} exp/zipvoice-onnx/

uv run hf upload --repo-type model thewh1teagle/zipvoice-heb ./exp/zipvoice_finetune/epoch-51.pt
uv run hf upload --repo-type model thewh1teagle/zipvoice-heb ./exp/zipvoice-onnx/ /zipvoice-onnx/
```

## Infer (ONNX)

```bash
uv run hf download --repo-type model thewh1teagle/zipvoice-heb --include 'zipvoice-onnx/' --local-dir .

uv run python3 -m zipvoice.bin.infer_zipvoice_onnx \
    --onnx-int8 True \
    --model-name zipvoice \
    --model-dir exp/zipvoice-onnx \
    --tokenizer raw_phoneme \
    --prompt-wav prompt1.wav \
    --prompt-text 'halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan.' \
    --text 'sˈimu lev nosʔˈim jekaʁˈim. haʁakˈevet letel ʔavˈiv meʁkˈaz tikanˈes leʁatsˈif mispˈaʁ ʃalˈoʃ beʔˈod mispˈaʁ dakˈot. ʔˈana hitʁaχakˈu miktsˈe haʁatsˈif vehamtˈinu meʔaχoʁˈej hakˈav hatsahˈov, todˈa!' \
    --res-wav-path result.wav
```

## Resume Training from Iteration Checkpoint

The `resume_checkpoint` function only looks for files named `epoch-{N}.pt`. To resume from an iteration checkpoint:

1. Copy the iteration checkpoint to epoch format:
   ```bash
   cp exp/zipvoice_finetune/checkpoint-8000.pt exp/zipvoice_finetune/epoch-1.pt
   ```
2. Set `--start-epoch 2` in the training script.

This restores the optimizer, scheduler, and `batch_idx_train` (8000) and continues from iteration 8000.

## Infer

```console
uv run python3 -m zipvoice.bin.infer_zipvoice \
      --model-name zipvoice \
      --model-dir exp/zipvoice_finetune/ \
      --checkpoint-name checkpoint-200.pt \
      --tokenizer raw_phoneme \
      --prompt-wav prompt.wav \
      --prompt-text "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan." \
      --text "simˈu lˈev noseʔˈim jekaʁˈim, haʁakˈevet tikanˈes letaχanˈat tˈel ʔavˈiv meʁkˈaz beʔˈod mispˈaʁ dakˈot, ʔˈana hitʁaχakˈu miktsˈe haʁatsˈif vehamtˈinu meʔaχoʁˈej hakˈav
  hatsahˈov, todˈa." \
      --res-wav-path result.wav \
      --num-step 16
```