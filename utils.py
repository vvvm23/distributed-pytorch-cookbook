import torch
from models import TransformerDecoderLM
from data import GPT2Tokenizer

def prepare_batch(batch, pad_id: int, device):
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
