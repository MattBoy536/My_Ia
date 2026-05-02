# router.py
"""Router : choisit les top-k experts pour un prompt donné."""
import torch
import torch.nn as nn
import pickle
import numpy as np
from pathlib import Path
from config import CFG
from super_tokenizer import SuperTokenizer


class Router(nn.Module):
    """Petit modèle qui prédit quels experts utiliser."""
    
    def __init__(self, vocab_size, n_experts):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, CFG.router_hidden)
        self.gru = nn.GRU(CFG.router_hidden, CFG.router_hidden, 
                         batch_first=True)
        self.head = nn.Linear(CFG.router_hidden, n_experts)
    
    def forward(self, x):
        h = self.embed(x)
        _, h = self.gru(h)
        return self.head(h.squeeze(0))
    
    def top_k(self, x, k=None):
        k = k or CFG.top_k_experts
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            values, indices = probs.topk(k, dim=-1)
        return indices, values


def train_router(clusters, tokenizer, device):
    """Entraîne le router à prédire le bon cluster."""
    print("[Router] Préparation données...")
    
    X, y = [], []
    for cluster_id, codes in clusters.items():
        for code in codes[:200]:  # échantillon
            ids = tokenizer.encode(code)[:CFG.max_seq_len]
            if len(ids) < CFG.max_seq_len:
                ids = ids + [0] * (CFG.max_seq_len - len(ids))
            X.append(ids)
            y.append(cluster_id)
    
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    
    # Mapping cluster_id → index
    unique_ids = sorted(set(y.tolist()))
    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
    y_mapped = torch.tensor([id_to_idx[c.item()] for c in y])
    
    n_experts = len(unique_ids)
    router = Router(tokenizer.vocab_size, n_experts).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=1e-3)
    
    print(f"[Router] Entraînement sur {len(X)} exemples...")
    batch_size = 64
    n_epochs = 3
    
    for ep in range(n_epochs):
        perm = torch.randperm(len(X))
        total_loss = 0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx].to(device), y_mapped[idx].to(device)
            logits = router(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {ep+1}: loss={total_loss/len(X)*batch_size:.3f}")
    
    torch.save({
        "state_dict": router.state_dict(),
        "id_to_idx": id_to_idx,
        "idx_to_id": {v: k for k, v in id_to_idx.items()},
        "n_experts": n_experts
    }, CFG.models_dir / "router.pt")
    
    print("[Router] Sauvegardé")
    return router
