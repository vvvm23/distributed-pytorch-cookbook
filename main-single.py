#!/usr/bin/env python

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_dataset, get_tokenizer, transform_dataset
from models import TransformerDecoderLM


def main(args: SimpleNamespace):
    batch_size = 8
    epochs = 1
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

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_dataset, validation_dataset = get_dataset()

    train_dataset = transform_dataset(
        train_dataset, tokenizer, max_length=sequence_length
    )
    # validation_dataset = transform_dataset(validation_dataset, tokenizer, max_length=sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    for ei in range(epochs):
        pb = tqdm(train_loader)
        pb.set_description(f"Epoch {ei+1}/{epochs} | loss: 0.0000")
        total_loss = 0.0
        for i, batch in enumerate(pb):
            optim.zero_grad()
            input_ids, attention_mask = (
                batch["input_ids"],
                batch["attention_mask"][:, -1],
            )
            input_ids, targets = input_ids[:, :-1].clone(), input_ids[:, 1:].clone()
            targets[targets == tokenizer.pad_token_id] = -100
            position_ids = (
                torch.arange(sequence_length - 1)
                .unsqueeze(0)
                .repeat(input_ids.shape[0], 1)
            )

            batch = dict(
                input_ids=input_ids.to(device),
                position_ids=position_ids.to(device),
                mask=~attention_mask.to(dtype=torch.bool, device=device),
            )
            targets = targets.to(device)
            logits = model(**batch)

            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size), targets.view(-1), ignore_index=-100
            )
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if i > 0 and not i % 32:
                pb.set_description(f"Epoch {ei+1}/{epochs} | loss: {total_loss / 32}")
                total_loss = 0.0


if __name__ == "__main__":
    main(None)
