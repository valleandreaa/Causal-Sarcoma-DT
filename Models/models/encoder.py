import torch
import torch.nn as nn
from typing import List, Optional

class Encoder(nn.Module):
    """MLP encoder with optional embeddings for ordinal features."""
    def __init__(self, input_dim: int, hidden_dim: int, cat_cards: List[int]):
        super().__init__()
        self.use_embeddings = len(cat_cards) > 0
        embed_dim_total = 0
        if self.use_embeddings:
            # For each ordinal column, add 1 to the card (to account for missing=0)
            self.embeddings = nn.ModuleList([
                nn.Embedding(card + 1, min(50, (card + 1) // 2), padding_idx=0) for card in cat_cards
            ])
            embed_dim_total = sum(e.embedding_dim for e in self.embeddings)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + embed_dim_total, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

    def forward(self, x_num_oh: torch.Tensor, x_ord: Optional[torch.Tensor]):  # type: ignore[override]
        if self.use_embeddings and x_ord is not None:
            embeds = [emb(x_ord[:, i].long()) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num_oh] + embeds, dim=1)
        else:
            x = x_num_oh
        return self.mlp(x)