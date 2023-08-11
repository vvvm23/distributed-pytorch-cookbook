from math import sqrt
from typing import Callable, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    dim: int
    mult: int = 4
    activation: Callable = F.relu
    out_dim: Optional[int] = None
    bias: bool = True
    dropout: float = 0.0

    def __init__(self):
        super().__init__()

        self.out_dim = self.dim if self.out_dim is None else self.out_dim
        self.up_proj = nn.Linear(self.dim, self.dim * self.mult, bias=self.bias)
        self.down_proj = nn.Linear(self.dim * self.mult, self.out_dim, bias=self.bias)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.up_proj(x)
        x = self.activation(x)

        x = self.down_proj(x)
        x = self.activation(x)

        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    dim: int
    head_dim: int
    heads: int

    qkv_bias: bool = False
    dropout: float = 0.0

    def __init__(self):
        super().__init__()

        self.to_q = nn.Linear(self.dim, self.head_dim * self.heads, bias=self.qkv_bias)
        self.to_k = nn.Linear(self.dim, self.head_dim * self.heads, bias=self.qkv_bias)
        self.to_v = nn.Linear(self.dim, self.head_dim * self.heads, bias=self.qkv_bias)

        self.to_out = nn.Linear(self.head_dim * self.heads, self.dim, bias=True)
        self.dropout = nn.Dropout(self.dropout)
        self.attn_scale = 1 / sqrt(self.head_dim)

    # TODO: add attention mask. Think we can do by expanding two dims between existing, then adding to causal
    def forward(
        self, x: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_k(v)

        q = einops.rearrange(q, "N S (h d) -> N h S d")
        k = einops.rearrange(k, "N S (h d) -> N h d S")
        v = einops.rearrange(v, "N S (h d) -> N h S d")

        attn_logits = (q @ k) * self.scale
        attn_scores = F.softmax(attn_logits, dim=-1)

        # TODO: should be this tril?
        causal_mask = 1e9 * (torch.triu(torch.ones(x.shape[1], x.shape[1])) - 1.0)
        causal_mask = einops.repeat(
            causal_mask, "i j -> N h i j", N=x.shape[0], h=self.heads
        )

        attn_scores = attn_scores + causal_mask

        attn_outputs = attn_scores @ v
        attn_outputs = einops.rearrange(attn_outputs, "N h S d -> N S (h d)")

        output = self.to_out(attn_outputs)
        output = self.dropout(output)

        return output


class DecoderLayer(nn.Module):
    dim: int
    head_dim: int
    heads: int
    dropout: float = 0.0

    def __init__(self):
        super().__init__()

        self.attn = SelfAttention(
            dim=self.dim, head_dim=self.head_dim, heads=self.heads, dropout=self.dropout
        )
        self.norm1 = nn.LayerNorm(self.dim)

        self.fc = FeedForward(dim=self.dim, dropout=self.dropout)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(
        self, x: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        h = self.norm1(x)
        h = self.attn(h, mask)
        x = h + x

        h = self.norm2(x)
        h = self.fc(h)
        x = h + x

        return x


class TransformerDecoder(nn.Module):
    dim: int
    head_dim: int
    heads: int
    num_layers: int

    dropout: float = 0.0

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    dim=self.dim,
                    head_dim=self.head_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self, x: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        for layer in self.layers:
            x = layer(x, mask)

        return x


class TransformerDecoderLM(nn.Module):
    dim: int
    head_dim: int
    heads: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int

    dropout: float = 0.0

    def __init__(self):
        super().__init__()

        self.input_embeddings = nn.Embedding(self.vocab_size, self.dim)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.dim)

        self.decoder = TransformerDecoder(
            num_layers=self.num_layers,
            dim=self.dim,
            head_dim=self.head_dim,
            heads=self.heads,
            dropout=self.dropout,
        )
        self.norm_out = nn.LayerNorm(self.dim)

        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        x = self.input_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.decoder(x, mask=mask)
        x = self.norm_out(x)

        return self.lm_head(x)
