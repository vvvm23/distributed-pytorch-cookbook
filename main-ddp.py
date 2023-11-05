"""
run with torchrun or torch elastic

    torchrun --nnodes=<num_nodes> --nproc_per_node=<num_gpus_per_node> --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=<address>:<port> main-fsdp.py

"""

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from data import GPT2Tokenizer, get_dataset, get_tokenizer, transform_dataset
from models import TransformerDecoderLM
from utils import generate, prepare_batch


def init_mp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = f"cuda:{rank % torch.cuda.device_count()}"
    print(f"Worker {rank} starting up. Using device_id '{device_id}'.")

    return rank, device_id


def cleanup_mp():
    dist.destroy_process_group()


def main(args: SimpleNamespace):
    PRINT_FREQ = 8
    rank, device_id = init_mp()

    device = torch.device(device_id)
    tokenizer = get_tokenizer()
    tokenizer.pad_token_id = 2

    # initialise the GPT-style model
    sp_model = TransformerDecoderLM(
        dim=args.dim,
        head_dim=args.head_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.sequence_length,
    ).to(device)
    model = DDP(sp_model, device_ids=[device_id])

    model.train()

    # compile the model for faster speeds
    if not args.disable_compile:
        model = torch.compile(model)

    # define a simple optimiser
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # get our training and validation dataset (optionally slice using args.dataset_slice)
    train_dataset, validation_dataset = get_dataset(slice_size=args.dataset_slice)

    # preprocess the dataset (tokenising and dropping columns)
    train_dataset = transform_dataset(
        train_dataset,
        tokenizer,
        max_length=args.sequence_length,
        num_proc=args.num_workers,
    )
    validation_dataset = transform_dataset(
        validation_dataset,
        tokenizer,
        max_length=args.sequence_length,
        num_proc=args.num_workers,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    validation_sampler = DistributedSampler(validation_dataset, shuffle=False)

    # define our dataloaders on the dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=validation_sampler,
    )

    # AMP for gradients. Pass --disable_amp to disable
    scaler = torch.cuda.amp.GradScaler(enabled=not args.disable_amp)

    for ei in range(args.epochs):
        pb = tqdm(train_loader, disable=rank != 0)
        pb.set_description(f"[training] Epoch {ei+1}/{args.epochs} | loss: ?????")
        total_loss = 0.0
        model.train()
        for i, batch in enumerate(pb):
            optim.zero_grad()
            batch, targets = prepare_batch(batch, tokenizer.pad_token_id, device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=not args.disable_amp
            ):
                # get logits from model
                logits = model(**batch)

                # compute cross entropy loss (mask out padding)
                logits, targets = logits.view(-1, model.vocab_size), targets.view(-1)
                loss = F.cross_entropy(logits, targets, ignore_index=-100)

            # backwards pass (with optional gradients scaling) and parameter update
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_loss += loss.item()
            if i > 0 and not i % PRINT_FREQ:
                pb.set_description(
                    f"[training] Epoch {ei+1}/{args.epochs} | loss: {total_loss / PRINT_FREQ:.3f}"
                )
                total_loss = 0.0

        model.eval()
        with torch.no_grad():
            pb = tqdm(validation_loader, disable=rank != 0)
            pb.set_description(
                f"[validation] Epoch {ei+1}/{args.epochs} | loss: ?????, accuracy: ?????"
            )
            total_loss, total_accuracy = 0.0, 0.0
            for i, batch in enumerate(pb):
                batch, targets = prepare_batch(batch, tokenizer.pad_token_id, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**batch)

                    logits, targets = logits.view(-1, model.vocab_size), targets.view(
                        -1
                    )
                    loss = F.cross_entropy(logits, targets, ignore_index=-100)

                # computes accuracy
                target_mask = targets != -100
                accuracy = (
                    logits.argmax(dim=-1)[target_mask] == targets[target_mask]
                ).float().mean() * 100.0

                # TODO: may need to wrap below in a tensor
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(accuracy, op=dist.ReduceOp.AVG)

                total_loss += loss.item()
                total_accuracy += accuracy

                pb.set_description(
                    f"[validation] Epoch {ei+1}/{args.epochs} | loss: {total_loss / (i+1):.3f}, accuracy: {total_accuracy / (i+1):.2f}"
                )

            # generate some example outputs from the current model
            if rank == 0:
                print("Argmax sampling from model")
                print(generate(model, "The big brown cat ", tokenizer, device))
                print(generate(model, "One day, ", tokenizer, device))
                print(generate(model, "She said ", tokenizer, device))

            dist.barrier()

    # save trained model at end of training
    dist.barrier()
    if rank == 0:
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        save_id = "checkpoint-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pt"
        torch.save(model.state_dict(), checkpoint_dir / save_id)

    cleanup_mp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--head_dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dataset_slice", type=str, default="100%")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--disable_compile", action="store_true")

    args = parser.parse_args()
    main(args)
