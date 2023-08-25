#!/usr/bin/env python

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_dataset, get_tokenizer, transform_dataset
from models import TransformerDecoderLM

def prepare_batch(batch, pad_id, device):
    sequence_length = batch["input_ids"].shape[1]
    input_ids, attention_mask = (
        batch["input_ids"],
        batch["attention_mask"][:, :-1],
    )
    input_ids, targets = input_ids[:, :-1].clone(), input_ids[:, 1:].clone()
    targets[targets == pad_id] = -100
    position_ids = (
        torch.arange(sequence_length - 1)
        .unsqueeze(0)
        .repeat(input_ids.shape[0], 1)
    )

    batch = dict(
        input_ids=input_ids.to(device, non_blocking=True),
        position_ids=position_ids.to(device, non_blocking=True),
        mask=~attention_mask.to(
            dtype=torch.bool, device=device, non_blocking=True
        ),
    )
    targets = targets.to(device, non_blocking=True)
    return batch, targets

def main(args: SimpleNamespace):
    batch_size = 32
    epochs = 4
    sequence_length = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = TransformerDecoderLM(
        dim=256,
        head_dim=32,
        heads=8,
        num_layers=8,
        vocab_size=tokenizer.vocab_size + 1,
        max_position_embeddings=sequence_length,
    ).to(device)
    model.train()
    # model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_dataset, validation_dataset = get_dataset()

    train_dataset = transform_dataset(
        train_dataset, tokenizer, max_length=sequence_length
    )
    validation_dataset = transform_dataset(validation_dataset, tokenizer, max_length=sequence_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    for ei in range(epochs):
        pb = tqdm(train_loader)
        pb.set_description(f"[training] Epoch {ei+1}/{epochs} | loss: 0.000")
        total_loss = 0.0
        model.train()
        for i, batch in enumerate(pb):
            optim.zero_grad()
            batch, targets = prepare_batch(batch, tokenizer.pad_token_id, device)
            logits = model(**batch)

            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size), targets.view(-1), ignore_index=-100
            )
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if i > 0 and not i % 32:
                pb.set_description(f"[training] Epoch {ei+1}/{epochs} | loss: {total_loss / 32:.3f}")
                total_loss = 0.0

        with torch.no_grad():
            model.eval()
            pb = tqdm(validation_loader)
            pb.set_description(f"[validation] Epoch {ei+1}/{epochs} | loss: 0.000")
            total_loss, total_accuracy = 0.0, 0.0
            for i, batch in enumerate(pb):
                batch, targets = prepare_batch(batch, tokenizer.pad_token_id, device)
                logits = model(**batch)

                logits, targets = logits.view(-1, model.vocab_size), targets.view(-1)
                loss = F.cross_entropy(
                    logits, targets, ignore_index=-100
                )

                accuracy = (logits.argmax(dim=-1) == targets).mean() * 100.

                total_loss += loss.item()
                accuracy += total_accuracy.item()

                pb.set_description(f"[validation] Epoch {ei+1}/{epochs} | loss: {total_loss / (i+1):.3f}, accuracy: {total_accuracy / (i+1):.2f}")



if __name__ == "__main__":
    main(None)
