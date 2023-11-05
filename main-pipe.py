#!/usr/bin/env python

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe

from tqdm import tqdm
import tempfile
import itertools 

from data import GPT2Tokenizer, get_dataset, get_tokenizer, transform_dataset
from models import TransformerDecoderLM

from utils import prepare_batch, generate

def init_mp():
    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method="file://{}".format(tmpfile.name))
    )


class WrappedDecoderLayer(nn.Module):
    def __init__(self, model):
        self.model = model

    def forward(self, x: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None):
        y = self.model(x, mask)
        return y, mask

class WrappedEmbeddings(nn.Module):
    def __init__(self, model):
        self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        y = self.model(input_ids, position_ids)
        return y, mask

def build_pipeline(model: TransformerDecoderLM, num_stages: int):
    first_stage = [WrappedEmbeddings(model.embeddings)]
    mid_stages = [[] for _ in range(num_stages - 2)]
    last_stage = [model.norm_out, model.lm_head]

    num_decoder_layers = len(model.decoder.layers)
    min_layers_per_stage = num_decoder_layers // num_stages
    remaining_layers = num_decoder_layers % num_stages

    layers_per_stage = [
        min_layers_per_stage + (remaining_layers == num_stages - 1),
        *[min_layers_per_stage + (i < remaining_layers) for i in range(num_stages - 2)]
        min_layers_per_stage
    ]

    first_stage = first_stage + model.decoder.layers[:layers_per_stage[0]]
    last_stage = model.decoder.layers[layers_per_stage[-1]:] + last_stage

    cum_layers_per_stage = itertools.accumulate(layers_per_stage)

    for i, (layers_so_far, layers_in_stage) for enumerate(zip(cum_layers_per_stage, layers_per_stage[1:-1])):
        mid_stages = model.decoder.layers[layers_so_far:layers_so_far+layers_in_stage]

    first_stage = torch.nn.Sequential(*first_stage)
    mid_stages = [torch.nn.Sequential(*stage) for stage in mid_stages]
    last_stage = torch.nn.Sequential(*last_stage)

    return Pipe(torch.nn.Sequential(
        first_stage,
        *mid_stages,
        last_stage
    ), chunks=num_stages)

def main(args: SimpleNamespace):
    PRINT_FREQ = 8
    init_mp()

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
