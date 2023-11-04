from typing import Optional, Union

import datasets
from transformers import GPT2Tokenizer


def get_dataset(
    name: str = "roneneldan/TinyStories", slice_size: Optional[Union[str, int]] = None
):
    train_dataset = datasets.load_dataset(
        name, split=f"train[:{slice_size}]" if slice_size is not None else "train"
    )
    validation_dataset = datasets.load_dataset(name, split="validation")
    return train_dataset, validation_dataset


# TODO: remove all but most common 10k tokens
def get_tokenizer(name: str = "roneneldan/TinyStories-1M", max_length: int = 512):
    tokenizer = GPT2Tokenizer.from_pretrained(name, model_max_length=max_length)
    return tokenizer


def transform_dataset(dataset, tokenizer, max_length: int = 512, num_proc: int = 8):
    def _tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    dataset = dataset.map(_tokenize, batched=True, remove_columns=["text"], num_proc=num_proc)
    dataset.set_format("pt")
    return dataset
