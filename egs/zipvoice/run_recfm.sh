#!/bin/bash

# Fine-tune an existing ZipVoice checkpoint with a depth-2 RecFM loss.
# The resulting model is intended for 2-step inference.

export PYTHONPATH=../../:$PYTHONPATH

set -e
set -u
set -o pipefail

stage=1
stop_stage=4

# Training data. By default this reuses the custom fine-tuning manifests from
# run_finetune.sh after its data-preparation stages have completed.
dataset=custom
train_manifest=dataset/fbank/custom-finetune_cuts_train.jsonl.gz
dev_manifest=dataset/fbank/custom-finetune_cuts_dev.jsonl.gz
tokenizer=raw_phoneme
lang=default
max_len=20

# Existing checkpoint to adapt. Stage 1 downloads the released ZipVoice
# files into this directory; override these paths to use your own checkpoint.
download_dir=download
pretrained_dir=${download_dir}/zipvoice
checkpoint=${pretrained_dir}/model.pt
model_config=${pretrained_dir}/model.json
token_file=${pretrained_dir}/tokens.txt

# RecFM fine-tuning settings.
exp_dir=exp/zipvoice_recfm
world_size=1
num_iters=10000
save_every_n=1000
avg=2
base_lr=1e-5
max_duration=200
recfm_lambda=1.0

# Inference settings for the optional final stage.
test_list=test.tsv
res_dir=results/test_recfm
infer_checkpoint=iter-${num_iters}-avg-${avg}.pt
num_step=2
guidance_scale=1.0

. utils/parse_options.sh

if [ "$tokenizer" = "espeak" ]; then
      [ "$lang" = "default" ] && { echo "Error: lang is not set!" >&2; exit 1; }
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Download ZipVoice checkpoint, config, and tokens"
      hf_repo=k2-fsa/ZipVoice
      mkdir -p ${download_dir}
      for file in model.pt tokens.txt model.json; do
            uv run hf download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice/${file}
      done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Fine-tune ZipVoice-RecFM"

      [ -f "$checkpoint" ] || { echo "Error: checkpoint not found: $checkpoint" >&2; exit 1; }
      [ -f "$model_config" ] || { echo "Error: model_config not found: $model_config" >&2; exit 1; }
      [ -f "$token_file" ] || { echo "Error: token_file not found: $token_file" >&2; exit 1; }
      [ -f "$train_manifest" ] || { echo "Error: train_manifest not found: $train_manifest" >&2; exit 1; }
      [ -f "$dev_manifest" ] || { echo "Error: dev_manifest not found: $dev_manifest" >&2; exit 1; }

      uv run python3 -m zipvoice.bin.train_zipvoice_recfm \
            --world-size ${world_size} \
            --use-fp16 1 \
            --finetune 1 \
            --checkpoint ${checkpoint} \
            --base-lr ${base_lr} \
            --num-iters ${num_iters} \
            --save-every-n ${save_every_n} \
            --keep-last-k 5 \
            --max-duration ${max_duration} \
            --max-len ${max_len} \
            --num-buckets 10 \
            --model-config ${model_config} \
            --start-epoch 1 \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file ${token_file} \
            --dataset ${dataset} \
            --train-manifest ${train_manifest} \
            --dev-manifest ${dev_manifest} \
            --recfm-lambda ${recfm_lambda} \
            --exp-dir ${exp_dir}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Average ZipVoice-RecFM checkpoints"
      uv run python3 -m zipvoice.bin.generate_averaged_model \
            --iter ${num_iters} \
            --avg ${avg} \
            --model-name zipvoice_recfm \
            --exp-dir ${exp_dir}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Inference with 2-step ZipVoice-RecFM"
      uv run python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_recfm \
            --model-dir ${exp_dir} \
            --checkpoint-name ${infer_checkpoint} \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --test-list ${test_list} \
            --res-dir ${res_dir} \
            --num-step ${num_step} \
            --guidance-scale ${guidance_scale}
fi
