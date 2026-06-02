#!/usr/bin/env python3
# Copyright         2026  Xiaomi Corp.        (authors: OpenAI)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tune ZipVoice with a depth-2 Recursive Flow Matching objective.

This script is intended to start from an existing ZipVoice checkpoint.
It adds a new RecFM scale embedding and loads the checkpoint with strict=False,
so all existing parameters are reused and the new scale embedding is trained
from scratch.
"""

import copy
import json
import logging
import os
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.cut import Cut, CutSet
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import zipvoice.utils.diagnostics as diagnostics
from zipvoice.bin.train_zipvoice import (
    display_and_save_batch,
    get_params,
    get_parser,
    tokenize_text,
)
from zipvoice.dataset.datamodule import TtsDataModule
from zipvoice.models.zipvoice_recfm import ZipVoiceRecFM
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    RawPhonemeTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import (
    load_checkpoint,
    remove_checkpoints,
    resume_checkpoint,
    save_checkpoint,
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from zipvoice.utils.common import (
    AttributeDict,
    GradScaler,
    MetricsTracker,
    cleanup_dist,
    create_grad_scaler,
    get_adjusted_batch_count,
    get_parameter_groups_with_lrs,
    prepare_input,
    set_batch_count,
    setup_dist,
    setup_logger,
    str2bool,
    torch_autocast,
)
from zipvoice.utils.hooks import register_inf_check_hooks
from zipvoice.utils.lr_scheduler import FixedLRScheduler, LRScheduler
from zipvoice.utils.optim import ScaledAdam

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]


def add_recfm_arguments(parser):
    parser.description = "Fine-tune ZipVoice with RecFM."
    parser.add_argument(
        "--recfm-lambda",
        type=float,
        default=1.0,
        help="Weight for the RecFM cross-scale consistency loss.",
    )
    parser.add_argument(
        "--train-full-model",
        type=str2bool,
        default=False,
        help="If false, only train fm_decoder parameters.",
    )


def _sample_time_and_scale(
    batch_size: int,
    device: torch.device,
    is_training: bool,
) -> Tuple[Tensor, Tensor]:
    if is_training:
        t = torch.rand(batch_size, 1, 1, device=device)
        alpha_base = torch.rand(batch_size, 1, 1, device=device)
    else:
        t = (
            (torch.arange(batch_size, device=device) / batch_size)
            .unsqueeze(1)
            .unsqueeze(2)
        )
        alpha_base = 0.5 * torch.ones_like(t)

    alpha = t + alpha_base * (1.0 - t)
    alpha = alpha.clamp_min(1.0e-4)
    return t, alpha


def compute_fbank_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    features: Tensor,
    features_lens: Tensor,
    tokens: List[List[int]],
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """Compute the depth-2 RecFM loss."""

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batch_size, num_frames, _ = features.shape
    noise = torch.randn_like(features)
    t, alpha = _sample_time_and_scale(
        batch_size=batch_size,
        device=device,
        is_training=is_training,
    )
    with torch.set_grad_enabled(is_training):
        primary_v, secondary_v, speech_condition_mask, padding_mask = model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            t=t,
            alpha=alpha,
            condition_drop_ratio=params.condition_drop_ratio,
        )
        target_v = features - noise

        loss_mask = speech_condition_mask & (~padding_mask)

        primary_loss = torch.mean((primary_v[loss_mask] - target_v[loss_mask]) ** 2)
        secondary_loss = torch.mean(
            (secondary_v[loss_mask] - (alpha * target_v)[loss_mask]) ** 2
        )
        consistency_loss = torch.mean(
            (secondary_v[loss_mask] - (alpha * primary_v)[loss_mask]) ** 2
        )
        loss = primary_loss + secondary_loss + params.recfm_lambda * consistency_loss

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    num_frames = features_lens.sum().item()
    info["frames"] = num_frames
    info["loss"] = loss.detach().cpu().item() * num_frames
    info["primary_loss"] = primary_loss.detach().cpu().item() * num_frames
    info["secondary_loss"] = secondary_loss.detach().cpu().item() * num_frames
    info["consistency_loss"] = consistency_loss.detach().cpu().item() * num_frames
    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
        )
        loss, loss_info = compute_fbank_loss(
            params=params,
            model=model,
            features=features,
            features_lens=features_lens,
            tokens=tokens,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    model.train()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    tot_loss = MetricsTracker()
    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    try:
        num_batches = len(train_dl)
    except TypeError:
        num_batches = None

    pbar = tqdm(
        enumerate(train_dl),
        total=num_batches,
        desc=f"Epoch {params.cur_epoch}",
    )

    for batch_idx, batch in pbar:
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params) + 100000)

        if (
            params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(
                f"Epoch {params.cur_epoch}, global_batch_idx: {params.batch_idx_train},"
                f" validation: {valid_info}"
            )
            logging.info(
                f"Maximum memory allocated so far is "
                f"{torch.cuda.max_memory_allocated() // 1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

        params.batch_idx_train += 1
        batch_size = len(batch["text"])

        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
        )

        try:
            with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):
                loss, loss_info = compute_fbank_loss(
                    params=params,
                    model=model,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    is_training=True,
                )

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            cur_lr = max(scheduler.get_last_lr())
            postfix = {
                "iter": params.batch_idx_train,
                "bs": batch_size,
                "lr": f"{cur_lr:.2e}",
            }
            postfix.update(
                {
                    name: f"{value:.4g}"
                    for name, value in loss_info.norm_items()
                    if name
                    in (
                        "loss",
                        "primary_loss",
                        "secondary_loss",
                        "consistency_loss",
                    )
                }
            )
            if params.use_fp16:
                postfix["grad_scale"] = f"{scaler._scale.item():.3g}"
            pbar.set_postfix(postfix)
        except Exception as e:
            logging.info(f"Caught exception : {e}.")
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )
        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break
        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 1024.0 or (
                cur_grad_scale < 4096.0 and params.batch_idx_train % 400 == 0
            ):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if params.batch_idx_train % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, "
                f"batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                f"cur_lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

    loss_value = tot_loss["loss"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        tokens, features, features_lens = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_tokens=True,
            return_feature=True,
        )
        try:
            with torch_autocast(dtype=torch.float16, enabled=params.use_fp16):
                loss, loss_info = compute_fbank_loss(
                    params=params,
                    model=model,
                    features=features,
                    features_lens=features_lens,
                    tokens=tokens,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params)
            raise
        logging.info(
            f"Maximum memory allocated so far is "
            f"{torch.cuda.max_memory_allocated() // 1000000}MB"
        )


def run(rank, world_size, args):
    params = get_params()
    params.update(vars(args))
    params.valid_interval = params.save_every_n
    if params.num_iters > 0:
        params.num_epochs = 1000000

    with open(params.model_config, "r") as f:
        model_config = json.load(f)
    params.update(model_config["model"])
    params.update(model_config["feature"])

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    os.makedirs(f"{params.exp_dir}", exist_ok=True)
    copyfile(src=params.model_config, dst=f"{params.exp_dir}/model.json")
    copyfile(src=params.token_file, dst=f"{params.exp_dir}/tokens.txt")
    setup_logger(f"{params.exp_dir}/log/log-train")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    if torch.cuda.is_available():
        params.device = torch.device("cuda", rank)
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    if params.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=params.token_file)
    elif params.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=params.token_file)
    elif params.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=params.token_file, lang=params.lang)
    elif params.tokenizer == "raw_phoneme":
        tokenizer = RawPhonemeTokenizer(token_file=params.token_file, lang=params.lang)
    else:
        assert params.tokenizer == "simple"
        tokenizer = SimpleTokenizer(token_file=params.token_file)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    params.update(tokenizer_config)

    logging.info(params)
    logging.info("About to create RecFM model")

    model = ZipVoiceRecFM(
        **model_config["model"],
        **tokenizer_config,
    )

    if params.checkpoint is not None:
        logging.info(
            f"Loading pre-trained model from {params.checkpoint} with strict=False"
        )
        _ = load_checkpoint(filename=params.checkpoint, model=model, strict=False)
    else:
        logging.warning("No --checkpoint specified; RecFM model will train from scratch.")

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters : {num_param}")

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    if params.start_epoch > 1:
        checkpoints = resume_checkpoint(params=params, model=model, model_avg=model_avg)

    model = model.to(params.device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if not params.train_full_model:
        num_trainable = 0
        for name, p in model.named_parameters():
            if "fm_decoder" in name:
                p.requires_grad = True
                num_trainable += p.numel()
            else:
                p.requires_grad = False

        logging.info(
            "A total of {} trainable parameters ({:.3f}% of the whole model)".format(
                num_trainable, num_trainable / num_param * 100
            )
        )

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model,
            lr=params.base_lr,
            include_names=True,
        ),
        lr=params.base_lr,
        clipping_scale=2.0,
    )

    scheduler = FixedLRScheduler(optimizer)
    scaler = create_grad_scaler(enabled=params.use_fp16)

    if params.start_epoch > 1 and checkpoints is not None:
        if "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])
        if "scheduler" in checkpoints:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])
        if "grad_scaler" in checkpoints:
            logging.info("Loading grad scaler state dict")
            scaler.load_state_dict(checkpoints["grad_scaler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(512)
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    def remove_short_and_long_utt(c: Cut, min_len: float, max_len: float):
        if c.duration < min_len or c.duration > max_len:
            return False
        return True

    _remove_short_and_long_utt = partial(
        remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len
    )

    datamodule = TtsDataModule(args)
    if params.dataset == "emilia":
        train_cuts = CutSet.mux(
            datamodule.train_emilia_EN_cuts(),
            datamodule.train_emilia_ZH_cuts(),
            weights=[46000, 49000],
        )
        train_cuts = train_cuts.filter(_remove_short_and_long_utt)
        dev_cuts = CutSet.mux(
            datamodule.dev_emilia_EN_cuts(),
            datamodule.dev_emilia_ZH_cuts(),
            weights=[0.5, 0.5],
        )
    elif params.dataset == "libritts":
        train_cuts = datamodule.train_libritts_cuts()
        train_cuts = train_cuts.filter(_remove_short_and_long_utt)
        dev_cuts = datamodule.dev_libritts_cuts()
    else:
        assert params.dataset == "custom"
        train_cuts = datamodule.train_custom_cuts(params.train_manifest)
        train_cuts = train_cuts.filter(_remove_short_and_long_utt)
        dev_cuts = datamodule.dev_custom_cuts(params.dev_manifest)
        dev_cuts = dev_cuts.filter(_remove_short_and_long_utt)

    if params.tokenizer in ["emilia", "espeak", "dialog"]:
        if not hasattr(train_cuts[0].supervisions[0], "tokens") or not hasattr(
            dev_cuts[0].supervisions[0], "tokens"
        ):
            logging.warning(
                f"Using {params.tokenizer} tokenizer but tokens are not prepared,"
                f"will tokenize on-the-fly, which can slow down training significantly."
            )
    _tokenize_text = partial(tokenize_text, tokenizer=tokenizer)
    train_cuts = train_cuts.map(_tokenize_text)
    dev_cuts = dev_cuts.map(_tokenize_text)

    train_dl = datamodule.train_dataloaders(train_cuts)
    valid_dl = datamodule.dev_dataloaders(dev_cuts)

    if params.scan_oom:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
        )

    logging.info("Training started")

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")

        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.num_iters > 0 and params.batch_idx_train > params.num_iters:
            break

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
        save_checkpoint(
            filename=filename,
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

        if rank == 0:
            if params.best_train_epoch == params.cur_epoch:
                best_train_filename = params.exp_dir / "best-train-loss.pt"
                copyfile(src=filename, dst=best_train_filename)

            if params.best_valid_epoch == params.cur_epoch:
                best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                copyfile(src=filename, dst=best_valid_filename)

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    add_recfm_arguments(parser)
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
