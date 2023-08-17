import datasets
from transformers import GPT2Tokenizer

def get_dataset(name: str = "roneneldan/TinyStories"):
    dataset = datasets.load_dataset(name)
    return dataset

# TODO: remove all but most common 10k tokens
def get_tokenizer(name: str = "EleutherAI/gpt-neo-1.3B", max_length: int = 512):
    tokenizer = GPT2Tokenizer.from_pretrained(name, model_max_length = max_length)
    return tokenizer

# TODO: implement dataset transforms for easy training later
# this includes tokenization and target building
def transform_dataset(dataset, tokenizer):
    pass