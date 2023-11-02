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
    input_ids, attention_mask = (
        batch["input_ids"],
        batch["attention_mask"][:, :-1],
    )
    sequence_length = input_ids.shape[1]
    input_ids, targets = input_ids[:, :-1].clone(), input_ids[:, 1:].clone()
    targets[targets == pad_id] = -100
    position_ids = (
        torch.arange(sequence_length - 1).unsqueeze(0).repeat(input_ids.shape[0], 1)
    )

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
    batch_size = 64
    epochs = 4
    sequence_length = 256
    print_freq = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    tokenizer.pad_token_id = 2

    model = TransformerDecoderLM(
        dim=256,
        head_dim=32,
        heads=8,
        num_layers=8,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=sequence_length,
    ).to(device)

    model.train()
    model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_dataset, validation_dataset = get_dataset(slice_size="10%")

    train_dataset = transform_dataset(
        train_dataset, tokenizer, max_length=sequence_length
    )
    validation_dataset = transform_dataset(
        validation_dataset, tokenizer, max_length=sequence_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    scaler = torch.cuda.amp.GradScaler()

    for ei in range(epochs):
        pb = tqdm(train_loader)
        pb.set_description(f"[training] Epoch {ei+1}/{epochs} | loss: 0.000")
        total_loss = 0.0
        model.train()
        for i, batch in enumerate(pb):
            optim.zero_grad()
            batch, targets = prepare_batch(batch, tokenizer.pad_token_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch)

                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    targets.view(-1),
                    ignore_index=-100,
                )

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            # optim.step()

            total_loss += loss.item()
            if i > 0 and not i % print_freq:
                pb.set_description(
                    f"[training] Epoch {ei+1}/{epochs} | loss: {total_loss / print_freq:.3f}"
                )
                total_loss = 0.0

        model.eval()
        with torch.no_grad():
            pb = tqdm(validation_loader)
            pb.set_description(f"[validation] Epoch {ei+1}/{epochs} | loss: 0.000")
            total_loss, total_accuracy = 0.0, 0.0
            for i, batch in enumerate(pb):
                batch, targets = prepare_batch(batch, tokenizer.pad_token_id, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**batch)

                    logits, targets = logits.view(-1, model.vocab_size), targets.view(-1)
                    loss = F.cross_entropy(logits, targets, ignore_index=-100)

                accuracy = (logits.argmax(dim=-1) == targets).float().mean() * 100.0

                total_loss += loss.item()
                total_accuracy += accuracy

                pb.set_description(
                    f"[validation] Epoch {ei+1}/{epochs} | loss: {total_loss / (i+1):.3f}, accuracy: {total_accuracy / (i+1):.2f}"
                )

            print("Argmax sampling from model")
            print(generate(model, "The big brown cat ", tokenizer, device))
            print(generate(model, "One day, ", tokenizer, device))
            print(generate(model, "She said ", tokenizer, device))


if __name__ == "__main__":
    main(None)
