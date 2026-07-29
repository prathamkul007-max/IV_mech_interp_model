"""Training the options-IV forecasting model on a single GPU (e.g. Colab T4)."""

import argparse
import json
import os
import time
from contextlib import nullcontext

import numpy as np
from tqdm.autonotebook import tqdm

import wandb

import torch
import torch.amp
import torch.nn as nn
from torch.utils.data import DataLoader

import gpt2.opts as opts
import gpt2.utils as utils
from gpt2.iv_dataset import IVDataset
from gpt2.iv_model import IVModel, IVModelConfig
from gpt2.meters import AverageMeter


def load_scaler(scaler_file: str) -> dict:
    with open(scaler_file) as f:
        return json.load(f)


def unscale_targets(values: torch.Tensor, scaler: dict) -> torch.Tensor:
    target_indices = scaler['target_indices']
    mean = torch.tensor([scaler['mean'][i] for i in target_indices], dtype=values.dtype, device=values.device)
    std = torch.tensor([scaler['std'][i] for i in target_indices], dtype=values.dtype, device=values.device)
    return values * std + mean


def train_model(args: argparse.Namespace) -> None:
    utils.set_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    print(f'Set float32 matmul precision to {args.matmul_precision}')

    checkpoints_dir = utils.ensure_dir(args.checkpoints_dir)

    scaler_params = load_scaler(args.scaler_file)
    args.num_features = len(scaler_params['mean'])
    args.num_targets = len(scaler_params['target_indices'])
    args.seq_length = scaler_params['seq_length']

    # dataset
    train_dataset = IVDataset(args.train_file)
    validation_dataset = IVDataset(args.valid_file)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=args.drop_last,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=args.drop_last,
    )

    # logging with wandb
    wandb_run = None
    if args.wandb_logging:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            id=args.wandb_resume_id,
            resume='must' if args.wandb_resume_id is not None else None,
        )

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # mixed precision training
    mp_dtype = torch.float32
    if device.type == 'cuda' and args.mixed_precision == 'fp16':
        mp_dtype = torch.float16
        print('Mixed precision training is enabled with fp16')
    elif device.type == 'cuda' and args.mixed_precision == 'bf16':
        if torch.cuda.is_bf16_supported():
            mp_dtype = torch.bfloat16
            print('Mixed precision training is enabled with bf16')
        else:
            mp_dtype = torch.float16
            print('bf16 is not supported on your hardware, fallback to fp16')
    autocast_context = torch.cuda.amp.autocast(enabled=(mp_dtype in (torch.float16, torch.bfloat16)), dtype=mp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(mp_dtype == torch.float16))

    # resume from previous checkpoint
    saved_states = None
    if args.from_checkpoint is None:
        model_config = IVModelConfig(
            num_features=args.num_features,
            num_targets=args.num_targets,
            seq_length=args.seq_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
        )
        model = IVModel(model_config)
    else:
        print(f'Loading states from checkpoint {args.from_checkpoint}')
        saved_states = torch.load(args.from_checkpoint, map_location=device)
        required_keys = ['model', 'optimizer', 'lr_scheduler', 'config']
        if scaler.is_enabled():
            required_keys.append('scaler')
        for key in required_keys:
            if key not in saved_states:
                raise ValueError(f'Missing key "{key}" in checkpoint')
        model_config = IVModelConfig(**saved_states['config'])
        model = IVModel(model_config)

    model.to(device)
    criterion = nn.MSELoss()
    learning_rate = args.learning_rate
    optimizer = utils.make_optimizer(
        model,
        device,
        args.optim_type,
        lr=learning_rate,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )
    if args.decay_method == 'noam':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.noam_decay(
                step, args.d_model, args.warmup_steps,
            ),
        )
    elif args.decay_method == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.cosine_decay(
                step, learning_rate, args.min_lr,
                args.warmup_steps,
                args.decay_steps, factor=1/learning_rate,
            ),
        )
    else:
        raise ValueError(f'Unsupported scheduler decay method: {args.decay_method}')

    initial_step = 0
    if saved_states is not None:
        model.load_state_dict(saved_states['model'])
        optimizer.load_state_dict(saved_states['optimizer'])
        lr_scheduler.load_state_dict(saved_states['lr_scheduler'])
        if scaler.is_enabled():
            scaler.load_state_dict(saved_states['scaler'])
        if 'global_step' in saved_states:
            initial_step = saved_states['global_step']

    raw_model = model
    if args.compile:
        print('Compiling the model')
        model = torch.compile(model)

    if args.do_test:
        valid_results = eval_model(
            model, device, criterion, validation_loader, args.valid_steps, autocast_context, scaler_params,
        )
        print('** Testing results **')
        print(f'MSE (scaled): {valid_results["loss"]}')
        print(f'RMSE (unscaled): {valid_results["rmse_unscaled"]}')
        return

    num_parameters = utils.count_model_param(model)
    print(f'Model has {num_parameters / 10 ** 6:0.2f}M parameters')

    train_iter = tqdm(
        range(initial_step, args.train_steps),
        desc='Training model',
        ncols=120,
    )

    global_step = initial_step
    batch_loss = 0.0
    batch_fb_time = 0.0
    wandb_accum_logs: list[dict] = []
    running_loss = AverageMeter('running_loss', device=device)

    model.train()
    optimizer.zero_grad()
    while global_step < args.train_steps:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            ts = time.perf_counter()
            inputs = inputs.to(device)
            targets = targets.to(device)

            with autocast_context:
                predictions = model(inputs)
                loss = criterion(predictions, targets)

            if args.gradient_accum_step > 1:
                loss /= args.gradient_accum_step
            batch_loss += loss.detach()

            scaler.scale(loss).backward()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_fb_time += time.perf_counter() - ts

            if (batch_idx + 1) % args.gradient_accum_step == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                wandb_accum_logs.append({
                    f'learning_rate/group_{group_id}': group_lr
                    for group_id, group_lr in enumerate(lr_scheduler.get_last_lr())
                })
                wandb_accum_logs[-1].update({
                    'loss/batch_loss': batch_loss,
                    'step': global_step,
                })

                lr_scheduler.step()
                running_loss.update(batch_loss)

                if (global_step + 1) % args.valid_interval == 0:
                    valid_results = eval_model(
                        model, device, criterion, validation_loader, args.valid_steps, autocast_context, scaler_params,
                    )
                    wandb_accum_logs[-1].update({
                        'loss/train': running_loss.average,
                        'loss/valid': valid_results['loss'],
                        'metrics/valid_rmse_unscaled': valid_results['rmse_unscaled'],
                    })
                    running_loss.reset()

                if (
                    len(wandb_accum_logs) >= args.wandb_logging_interval or
                    (len(wandb_accum_logs) > 0 and global_step + 1 >= args.train_steps)
                ):
                    if wandb_run is not None:
                        for log in wandb_accum_logs:
                            log['loss/batch_loss'] = float(log['loss/batch_loss'])
                            wandb_run.log(log)
                    wandb_accum_logs = []

                if (global_step + 1) % args.save_interval == 0:
                    save_checkpoint(raw_model, optimizer, lr_scheduler, scaler, model_config, global_step + 1, checkpoints_dir, args)

                train_iter.set_postfix({'loss': f'{batch_loss:0.4f}'})
                batch_loss = 0.0
                batch_fb_time = 0.0
                global_step += 1
                train_iter.update()
                if global_step >= args.train_steps:
                    break

        if global_step == args.train_steps and args.train_steps % args.save_interval != 0:
            save_checkpoint(raw_model, optimizer, lr_scheduler, scaler, model_config, global_step, checkpoints_dir, args)


def save_checkpoint(model, optimizer, lr_scheduler, scaler, model_config, global_step, checkpoints_dir, args):
    checkpoint_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'config': vars(model_config),
        'global_step': global_step,
    }
    if scaler.is_enabled():
        checkpoint_dict['scaler'] = scaler.state_dict()
    utils.ensure_num_saved_checkpoints(checkpoints_dir, 'iv_model', args.saved_checkpoint_limit)
    model_save_path = os.path.join(checkpoints_dir, f'iv_model-{global_step}.pt')
    torch.save(checkpoint_dict, model_save_path)


def main():
    parser = argparse.ArgumentParser(
        description='Train the options-IV forecasting model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opts.add_run_pretrain_iv_opts(parser)
    args = parser.parse_args()
    train_model(args)


@torch.no_grad()
def eval_model(
    model,
    device: torch.device,
    criterion,
    eval_loader: DataLoader,
    valid_steps: int,
    autocast_context=None,
    scaler_params: dict | None = None,
) -> dict[str, float]:
    evaluation_loss = AverageMeter('evaluation_loss', device=device)
    squared_error_sum = 0.0
    squared_error_count = 0
    if autocast_context is None:
        autocast_context = nullcontext()

    progress_bar = tqdm(
        range(valid_steps),
        total=valid_steps,
        desc='Evaluating model',
        ncols=120,
    )

    is_training = model.training
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(eval_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast_context:
            predictions = model(inputs)
            loss = criterion(predictions, targets)

        evaluation_loss.update(loss.detach())

        if scaler_params is not None:
            predictions_unscaled = unscale_targets(predictions.float(), scaler_params)
            targets_unscaled = unscale_targets(targets.float(), scaler_params)
            squared_error = (predictions_unscaled - targets_unscaled) ** 2
            squared_error_sum += squared_error.sum().item()
            squared_error_count += squared_error.numel()

        progress_bar.set_postfix({'loss': f'{loss:0.4f}'})
        progress_bar.update()
        if (batch_idx + 1) >= valid_steps:
            break

    model.train(is_training)

    rmse_unscaled = float(np.sqrt(squared_error_sum / squared_error_count)) if squared_error_count > 0 else float('nan')

    return {
        'loss': evaluation_loss.average,
        'rmse_unscaled': rmse_unscaled,
    }


if __name__ == '__main__':
    main()
