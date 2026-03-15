#!/bin/bash

# This script is an example of fine-tuning ZipVoice on your custom datasets.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=6

# Number of jobs for data preparation
nj=20

# Whether the language of training data is one of Chinese and English
tokenizer=raw_phoneme # Espeak-ng compatible raw phonemes
# tokenizer=emilia # for Chinese
# tokenizer=espeak # for English

# Language identifier, used when language is not Chinese or English
# see https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
# Example of French: lang=fr
lang=default

if [ "$tokenizer" = "espeak" ]; then
      [ "$lang" = "default" ] && { echo "Error: lang is not set!" >&2; exit 1; }
fi

# You can set `max_len` according to statistics from the command 
# `lhotse cut describe dataset/fbank/custom-finetune_cuts_train.jsonl.gz`.
# Set `max_len` to 99% duration.

# Maximum length (seconds) of the training utterance, will filter out longer utterances
max_len=20

# Download directory for pre-trained models
download_dir=download/

# We suppose you have two TSV files: "dataset/raw/train.tsv" and
# "dataset/raw/dev.tsv".

# Each line of the TSV files should be in one of the following formats:
# (1) `{uniq_id}\t{text}\t{wav_path}` if the text corresponds to the full wav,
# (2) `{uniq_id}\t{text}\t{wav_path}\t{start_time}\t{end_time}` if text corresponds
#     to part of the wav. The start_time and end_time specify the start and end
#     times of the text within the wav, which should be in seconds.
# > Note: {uniq_id} must be unique for each line.
for subset in train dev;do
      file_path=dataset/raw/${subset}.tsv
      [ -f "$file_path" ] || { echo "Error: expect $file_path !" >&2; exit 1; }
done

### Prepare the training data (1 - 4)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for custom dataset from tsv files"

      for subset in train dev;do
            uv run python3 -m zipvoice.bin.prepare_dataset \
                  --tsv-path dataset/raw/${subset}.tsv \
                  --prefix custom-finetune \
                  --subset raw_${subset} \
                  --num-jobs ${nj} \
                  --output-dir dataset/manifests
      done
      # The output manifest files are "dataset/manifests/custom-finetune_cuts_raw_train.jsonl.gz".
      # and "dataset/manifests/custom-finetune_cuts_raw_dev.jsonl.gz".
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Add tokens to manifests"
      # For "emilia" and "espeak" tokenizers, it's better to prepare the tokens 
      # before training. Otherwise, the on-the-fly tokenization can significantly
      # slow down the training.
      for subset in train dev;do
            uv run python3 -m zipvoice.bin.prepare_tokens \
                  --input-file dataset/manifests/custom-finetune_cuts_raw_${subset}.jsonl.gz \
                  --output-file dataset/manifests/custom-finetune_cuts_${subset}.jsonl.gz \
                  --tokenizer ${tokenizer} \
                  --lang ${lang}
      done
      # The output manifest files are "dataset/manifests/custom-finetune_cuts_train.jsonl.gz".
      # and "dataset/manifests/custom-finetune_cuts_dev.jsonl.gz".
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
      echo "Stage 3: Compute Fbank for custom dataset"
      # You can skip this step and use `--on-the-fly-feats 1` in training stage
      for subset in train dev; do
            uv run python3 -m zipvoice.bin.compute_fbank \
                  --source-dir dataset/manifests \
                  --dest-dir dataset/fbank \
                  --dataset custom-finetune \
                  --subset ${subset} \
                  --num-jobs ${nj}
      done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Download pre-trained model, tokens file, and model config"
      # Uncomment this line to use HF mirror
      # export HF_ENDPOINT=https://hf-mirror.com
      hf_repo=k2-fsa/ZipVoice
      mkdir -p ${download_dir}
      for file in model.pt tokens.txt model.json; do
            uv run hf download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice/${file}
      done
fi

### Training ZipVoice (5 - 6)

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Fine-tune the ZipVoice model"

      [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

      # --world-size depends on the number of GPUs you have
      # Note: Reduced learning rate from 1e-4 to 1e-6 for fp16 stability (gradients exploding)
      uv run python3 -m zipvoice.bin.train_zipvoice \
            --world-size 1 \
            --use-fp16 1 \
            --finetune 1 \
            --checkpoint ${download_dir}/zipvoice/model.pt \
            --base-lr 1e-4 \
            --num-iters 9990000 \
            --save-every-n 200 \
            --keep-last-k 5 \
            --max-duration 200 \
            --max-len ${max_len} \
            --num-buckets 10 \
            --model-config ${download_dir}/zipvoice/model.json \
            --start-epoch 1 \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file ${download_dir}/zipvoice/tokens.txt \
            --dataset custom \
            --train-manifest dataset/fbank/custom-finetune_cuts_train.jsonl.gz \
            --dev-manifest dataset/fbank/custom-finetune_cuts_dev.jsonl.gz \
            --exp-dir exp/zipvoice_finetune

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Average the checkpoints for ZipVoice"
      uv run python3 -m zipvoice.bin.generate_averaged_model \
            --iter 10000 \
            --avg 2 \
            --model-name zipvoice \
            --exp-dir exp/zipvoice_finetune
      # The generated model is exp/zipvoice_finetune/iter-10000-avg-2.pt
fi

### Inference with PyTorch models (7)

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "Stage 7: Inference of the ZipVoice model"

      uv run python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --model-dir exp/zipvoice_finetune/ \
            --checkpoint-name iter-10000-avg-2.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --test-list test.tsv \
            --res-dir results/test_finetune\
            --num-step 16
fi
