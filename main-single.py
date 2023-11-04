#!/usr/bin/env python

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GPT2Tokenizer, get_dataset, get_tokenizer, transform_dataset
from models import TransformerDecoderLM


def prepare_batch(batch, pad_id, device):
    """
    Function that takes in a batch from the Huggingface dataset and processes
    it into a format compatible with the model.

    It takes a set of input_ids and a attention_mask, and returns the
    input_ids, position_ids, attention_mask and target labels, all set to the
    correct device.
    """

    input_ids, attention_mask = (
        batch["input_ids"],
        batch["attention_mask"][:, :-1],  # remove last index in mask
    )
    sequence_length = input_ids.shape[1]

    # create targets by shifting input_ids by one
    input_ids, targets = input_ids[:, :-1].clone(), input_ids[:, 1:].clone()

    # if target contains padding id, set to -100 (CrossEntropy ignore_index)
    targets[targets == pad_id] = -100

    # generate position ids from 0..sequence_length-1
    position_ids = (
        torch.arange(sequence_length - 1).unsqueeze(0).repeat(input_ids.shape[0], 1)
    )

    # format batch as a dictionary with matching keywords to model keywords
    batch = dict(
        input_ids=input_ids.to(device, non_blocking=True),
        position_ids=position_ids.to(device, non_blocking=True),
        mask=~attention_mask.to(dtype=torch.bool, device=device, non_blocking=True),
    )
    targets = targets.to(device, non_blocking=True)
    return batch, targets


@torch.inference_mode()
def generate(
    model: TransformerDecoderLM,
    prompt: str,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    max_new_tokens: int = 20,
):
    """
    Function that given a model, initial prompt, and tokenizer, generates
    sample sequences from the model.

    It uses simple argmax sampling which is equivalent to sampling with a
    temperature of 0.
    """
    batch = tokenizer([prompt], truncation=True, max_length=256, return_tensors="pt")
    input_ids = batch["input_ids"]
    sequence_length = input_ids.shape[1]
    position_ids = torch.arange(sequence_length).unsqueeze(0)

    input_ids, position_ids = input_ids.to(device), position_ids.to(device)
    for _ in range(max_new_tokens):
        logits = model(input_ids=input_ids, position_ids=position_ids)
        new_token = logits[0, -1].argmax(dim=-1)

        if new_token == tokenizer.eos_token_id:
            break

        input_ids = torch.cat(
            [
                input_ids,
                torch.tensor([new_token], dtype=input_ids.dtype)
                .to(device)
                .unsqueeze(0),
            ],
            dim=1,
        )
        position_ids = torch.cat(
            [
                position_ids,
                torch.tensor([sequence_length], dtype=position_ids.dtype)
                .to(device)
                .unsqueeze(0),
            ],
            dim=1,
        )

        sequence_length += 1

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main(args: SimpleNamespace):
    PRINT_FREQ = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    tokenizer.pad_token_id = 2

    # initialise the GPT-style model
    model = TransformerDecoderLM(
        dim=args.dim,
        head_dim=args.head_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.sequence_length,
    ).to(device)

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

    # define our dataloaders on the dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # AMP for gradients. Pass --disable_amp to disable
    scaler = torch.cuda.amp.GradScaler(enabled=not args.disable_amp)

    for ei in range(args.epochs):
        pb = tqdm(train_loader)
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
            pb = tqdm(validation_loader)
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

                total_loss += loss.item()
                total_accuracy += accuracy

                pb.set_description(
                    f"[validation] Epoch {ei+1}/{args.epochs} | loss: {total_loss / (i+1):.3f}, accuracy: {total_accuracy / (i+1):.2f}"
                )

            # generate some example outputs from the current model
            print("Argmax sampling from model")
            print(generate(model, "The big brown cat ", tokenizer, device))
            print(generate(model, "One day, ", tokenizer, device))
            print(generate(model, "She said ", tokenizer, device))


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
