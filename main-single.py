#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader

from data import get_dataset, get_tokenizer, transform_dataset
from models import TransformerDecoderLM

import argparse
from types import SimpleNamespace

def main(args: SimpleNamespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerDecoderLM(
        dim = 256,
        head_dim = 32,
        heads = 8,
        num_layers = 8,
        vocab_size = 10_000,
        max_position_embeddings = 512,
    ).to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    dataset = get_dataset()
    tokenizer = get_tokenizer()

    dataset = transform_dataset(dataset, tokenizer)

    # TODO: write training loop!

if __name__ == '__main__':
    main(None)