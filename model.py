# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq: int = 2048, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq = max_seq
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype,
                         device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cache[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cache[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, H, T, D)

        cos, sin = self.rope(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # Flash attention si dispo
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attn_mask is None),
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    """SwiGLU FFN."""

    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * mult
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") \
                     else nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") \
                     else nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult, dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class ExpertTransformer(nn.Module):
    """Expert spécialisé : ~25M params."""

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 ffn_mult: int = 4,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") \
                      else nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self,
                idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Seq {T} > max {self.max_seq_len}"

        x = self.tok_emb(idx)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self,
                 idx: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 0.8,
                 top_p: float = 0.9) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_local = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_local)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def build_expert(cfg) -> ExpertTransformer:
    return ExpertTransformer(
        vocab_size  = cfg.vocab_size,
        d_model     = cfg.d_model,
        n_heads     = cfg.n_heads,
        n_layers    = cfg.n_layers,
        ffn_mult    = cfg.ffn_mult,
        max_seq_len = cfg.max_seq_len,
        dropout     = cfg.dropout,
    )


if __name__ == "__main__":
    from config import CFG
    model = build_expert(CFG)
    print(f"Expert : {model.num_params() / 1e6:.2f}M paramètres")
    x = torch.randint(0, CFG.vocab_size, (2, 64))
    logits, loss = model(x, x)
    print(f"Logits : {logits.shape}  Loss : {loss.item():.4f}")
