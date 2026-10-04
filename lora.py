import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    """
    LoRA wrapper for timm ViT's combined QKV projection.

    Original:
        qkv = Linear(dim, 3 * dim)

    LoRA is applied only to:
        Query (Q)
        Value (V)

    The original QKV layer remains frozen.
    """

    def __init__(self, qkv, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()

        self.qkv = qkv

        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

        # Freeze original pretrained QKV projection
        for param in self.qkv.parameters():
            param.requires_grad = False

        dim = qkv.in_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.dropout = nn.Dropout(dropout)

        # Query LoRA
        self.q_A = nn.Linear(dim, rank, bias=False)
        self.q_B = nn.Linear(rank, dim, bias=False)

        # Value LoRA
        self.v_A = nn.Linear(dim, rank, bias=False)
        self.v_B = nn.Linear(rank, dim, bias=False)

        # LoRA initialization
        nn.init.kaiming_uniform_(self.q_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.q_B.weight)

        nn.init.kaiming_uniform_(self.v_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.v_B.weight)

    def forward(self, x):

        # Original frozen QKV projection
        qkv = self.qkv(x)

        # Split Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        x_lora = self.dropout(x)

        # LoRA updates
        q_update = self.q_B(self.q_A(x_lora)) * self.scaling
        v_update = self.v_B(self.v_A(x_lora)) * self.scaling

        # Add LoRA updates
        q = q + q_update
        v = v + v_update

        # Reconstruct QKV in the format expected by timm
        return torch.cat([q, k, v], dim=-1)


def apply_lora(model, rank=8, alpha=16.0, dropout=0.0):
    """
    Insert LoRA into all transformer attention blocks.
    """

    if not hasattr(model, "blocks"):
        raise ValueError(
            "LoRA implementation expects a ViT-style model with model.blocks"
        )

    count = 0

    for block in model.blocks:

        if not hasattr(block, "attn") or not hasattr(block.attn, "qkv"):
            raise ValueError(
                "Transformer block does not contain attn.qkv"
            )

        block.attn.qkv = LoRAQKV(
            block.attn.qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        count += 1

    print(f"[LoRA] Injected LoRA into {count} transformer blocks.")
    return model