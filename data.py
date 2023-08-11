# TODO: download TinyStories dataset from Huggingface and return torch-compatible dataset here
# don't need to return Dataloader, as this may be parallelism strategy dependent

import datasets

def get_dataset(name: str = "roneneldan/TinyStories"):
    dataset = datasets.load_dataset(name)
    return dataset