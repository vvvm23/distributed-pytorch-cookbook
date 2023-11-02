from math import sqrt
from typing import Callable, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        activation: Callable = F.relu,
        out_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.activation = activation
        self.out_dim = out_dim
        self.bias = bias
        self.dropout = dropout

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
    def __init__(
        self,
        dim: int,
        head_dim: int,
        heads: int,
        qkv_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.heads = heads
        self.qkv_bias = qkv_bias
        self.dropout = dropout

        self.to_q = nn.Linear(self.dim, self.head_dim * self.heads, bias=self.qkv_bias)
        self.to_k = nn.Linear(self.dim, self.head_dim * self.heads, bias=self.qkv_bias)
        self.to_v = nn.Linear(self.dim, self.head_dim * self.heads, bias=self.qkv_bias)

        self.to_out = nn.Linear(self.head_dim * self.heads, self.dim, bias=True)
        self.dropout = nn.Dropout(self.dropout)
        self.attn_scale = 1 / sqrt(self.head_dim)

    def forward(
        self, x: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = einops.rearrange(q, "N S (h d) -> N h S d", h=self.heads)
        k = einops.rearrange(k, "N S (h d) -> N h d S", h=self.heads)
        v = einops.rearrange(v, "N S (h d) -> N h S d", h=self.heads)

        attn_logits = (q @ k) * self.attn_scale

        # TODO: cache mask? slice to correct size
        # TODO: check direction of this mask
        causal_mask = 1e9 * (torch.tril(torch.ones(x.shape[1], x.shape[1])) - 1.0).to(
            device=x.device, dtype=attn_logits.dtype
        )
        causal_mask = einops.repeat(
            causal_mask, "i j -> N h i j", N=x.shape[0], h=self.heads
        )

        attn_logits = attn_logits + causal_mask
        if mask is not None:
            # attn_logits = attn_logits.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e9)
            attn_logits = attn_logits.masked_fill(mask[:, None, None, :], torch.finfo(attn_logits.dtype).min)

        attn_scores = F.softmax(attn_logits, dim=-1)

        attn_outputs = attn_scores @ v
        attn_outputs = einops.rearrange(attn_outputs, "N h S d -> N S (h d)")

        output = self.to_out(attn_outputs)
        output = self.dropout(output)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, head_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.heads = heads
        self.dropout = dropout

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
    def __init__(
        self, dim: int, head_dim: int, heads: int, num_layers: int, dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = dropout

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
    def __init__(
        self,
        dim: int,
        head_dim: int,
        heads: int,
        num_layers: int,
        vocab_size: int,
        max_position_embeddings: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.heads = heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout

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
