# train.py
import json
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from config import CFG
from model import ExpertTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── DATASET ─────────────────────────────────────────────────────────────────

class CodeDataset(Dataset):
    def __init__(self, codes: list, tokenizer, max_len: int = None):
        self.max_len   = max_len or CFG.max_seq_len
        self.tokenizer = tokenizer
        self.samples   = []

        for code in codes:
            try:
                ids = tokenizer.encode(code, add_eos=True)
                # Découpe en chunks de max_len + 1
                for i in range(0, len(ids) - 1, self.max_len):
                    chunk = ids[i: i + self.max_len + 1]
                    if len(chunk) > 4:
                        self.samples.append(chunk)
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x     = torch.tensor(chunk[:-1], dtype=torch.long)
        y     = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


def collate_fn(batch, pad_id: int = 2):
    """Padding dynamique au sein du batch."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    x_padded = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    y_padded = torch.full((len(ys), max_len), -1,     dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_padded[i, :x.size(0)] = x
        y_padded[i, :y.size(0)] = y

    return x_padded, y_padded


# ─── ENTRAÎNEMENT D'UN EXPERT ────────────────────────────────────────────────

def train_one_expert(expert_id: int,
                     codes: list,
                     tokenizer,
                     device: str,
                     save_dir: Path) -> dict:

    save_path = save_dir / f"expert_{expert_id:04d}.pt"

    if save_path.exists():
        return {"id": expert_id, "status": "skipped"}

    if len(codes) < CFG.min_cluster_size:
        return {"id": expert_id, "status": "too_small", "n": len(codes)}

    try:
        # Dataset
        ds = CodeDataset(codes, tokenizer)
        if len(ds) == 0:
            return {"id": expert_id, "status": "empty"}

        def _collate(b):
            return collate_fn(b, pad_id=tokenizer.pad_id)

        loader = DataLoader(
            ds,
            batch_size=CFG.batch_size,
            shuffle=True,
            collate_fn=_collate,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        # Modèle
        model = ExpertTransformer(vocab_size=tokenizer.vocab_size).to(device)

        dtype = torch.bfloat16 if CFG.dtype == "bfloat16" else torch.float16
        if dtype == torch.bfloat16 and torch.cuda.is_bf16_supported():
            model = model.to(torch.bfloat16)

        if CFG.compile_model:
            try:
                model = torch.compile(model)
            except Exception:
                pass  # compile optionnel

        # Optimizer
        opt    = torch.optim.AdamW(
            model.parameters(),
            lr=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
            fused=True if "cuda" in device else False
        )

        total_steps = CFG.max_steps
        sched       = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=CFG.warmup_steps,
            num_training_steps=total_steps
        )
        scaler = GradScaler(enabled=(dtype != torch.bfloat16))

        # Entraînement
        model.train()
        step       = 0
        losses     = []
        accum_step = 0

        opt.zero_grad()

        while step < total_steps:
            for x, y in loader:
                if step >= total_steps:
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast(dtype=dtype):
                    _, loss = model(x, y)
                    loss    = loss / CFG.grad_accum

                scaler.scale(loss).backward()
                accum_step += 1

                if accum_step == CFG.grad_accum:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), CFG.grad_clip
                    )
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad()
                    accum_step = 0

                    losses.append(loss.item() * CFG.grad_accum)
                    step += 1

        # Calcul loss finale
        final_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1)

        # Sauvegarde
        state = model.state_dict()
        # Supprime le préfixe "_orig_mod." de torch.compile si présent
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

        torch.save({
            "state_dict": state,
            "vocab_size":  tokenizer.vocab_size,
            "final_loss":  final_loss,
            "n_examples":  len(codes)
        }, save_path)

        return {
            "id":        expert_id,
            "status":    "done",
            "loss":      final_loss,
            "n_examples": len(codes)
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning(f"OOM expert {expert_id}, skipped")
        return {"id": expert_id, "status": "oom"}

    except Exception as e:
        torch.cuda.empty_cache()
        logger.error(f"Expert {expert_id} erreur : {e}")
        return {"id": expert_id, "status": "error", "error": str(e)}


# ─── ENTRAÎNEMENT DE TOUS LES EXPERTS ────────────────────────────────────────

# train.py - Remplacer train_all_experts par cette version multi-GPU

import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def _train_worker(args):
    """Worker qui tourne sur un GPU spécifique."""
    expert_id, codes, tokenizer, gpu_id, save_dir = args
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    return train_one_expert(
        expert_id=expert_id,
        codes=codes,
        tokenizer=tokenizer,
        device=device,
        save_dir=save_dir
    )


def train_all_experts(clusters: dict,
                      tokenizer,
                      device: str,
                      save_dir: Path) -> list:
    """Distribue les experts sur les 2 GPUs en parallèle."""
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n_gpus = torch.cuda.device_count()
    logger.info(f"\n🚀 Multi-GPU détecté : {n_gpus} GPUs")
    
    if n_gpus < 2:
        logger.warning("Moins de 2 GPU → mode single GPU")
        return _train_all_single_gpu(clusters, tokenizer, device, save_dir)
    
    ids   = sorted(clusters.keys())
    total = len(ids)
    results = []
    start = time.time()
    
    logger.info(f"Distribution de {total} experts sur {n_gpus} GPUs")
    
    # Round-robin : expert i → gpu (i % n_gpus)
    tasks = [
        (eid, clusters[eid], tokenizer, eid % n_gpus, save_dir)
        for eid in ids
    ]
    
    # mp.set_start_method('spawn', force=True) → à mettre dans run_all
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(_train_worker, t): t[0] for t in tasks}
        
        done = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                eid = futures[future]
                logger.error(f"Expert {eid} crash : {e}")
                results.append({"id": eid, "status": "error", "error": str(e)})
            
            done += 1
            if done % 20 == 0:
                elapsed = time.time() - start
                eta = elapsed / done * (total - done)
                ok = sum(1 for r in results if r["status"] == "done")
                logger.info(
                    f"[{done}/{total}] OK={ok} "
                    f"Elapsed={elapsed/60:.0f}min ETA={eta/60:.0f}min"
                )
    
    return results


def _train_all_single_gpu(clusters, tokenizer, device, save_dir):
    """Fallback single GPU (code original)."""
    ids     = sorted(clusters.keys())
    total   = len(ids)
    results = []
    start   = time.time()

    for expert_id in tqdm(ids, desc="Experts"):
        result = train_one_expert(
            expert_id=expert_id,
            codes=clusters[expert_id],
            tokenizer=tokenizer,
            device=device,
            save_dir=save_dir
        )
        results.append(result)
        if "cuda" in device:
            torch.cuda.empty_cache()
    return results

# ─── ROUTER ──────────────────────────────────────────────────────────────────

class RouterModel(nn.Module):
    """
    Classifier léger : embedding moyen → top-k experts.
    """
    def __init__(self, vocab_size: int, n_experts: int,
                 hidden: int = None):
        super().__init__()
        hidden = hidden or CFG.router_hidden

        self.embed     = nn.EmbeddingBag(vocab_size, hidden,
                                          mode="mean", sparse=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb    = self.embed(x)
        logits = self.classifier(emb)
        return logits


class RouterDataset(Dataset):
    def __init__(self, clusters: dict, tokenizer, max_len: int = 256):
        self.samples = []
        for cluster_id, codes in clusters.items():
            for code in codes[:100]:  # max 100 par cluster
                try:
                    ids = tokenizer.encode(code)[:max_len]
                    if len(ids) > 2:
                        self.samples.append((ids, cluster_id))
                except Exception:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def router_collate(batch, pad_id: int = 2):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    padded  = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, :x.size(0)] = x
    return padded, torch.stack(ys)


def train_router(clusters: dict, tokenizer, device: str):
    """Entraîne le router de classification des experts."""
    router_path = CFG.models_dir / "router.pt"

    if router_path.exists():
        logger.info("Router déjà entraîné ✅")
        return

    n_experts = len(clusters)
    logger.info(f"Entraînement Router : {n_experts} classes...")

    ds     = RouterDataset(clusters, tokenizer)
    loader = DataLoader(
        ds,
        batch_size=128,
        shuffle=True,
        collate_fn=lambda b: router_collate(b, tokenizer.pad_id),
        num_workers=2
    )

    model = RouterModel(
        vocab_size=tokenizer.vocab_size,
        n_experts=n_experts
    ).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    for epoch in range(CFG.router_epochs):
        total_loss = 0.0
        correct    = 0
        total      = 0

        for x, y in tqdm(loader, desc=f"Router epoch {epoch+1}"):
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = crit(logits, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            correct    += (logits.argmax(-1) == y).sum().item()
            total      += y.size(0)

        acc = correct / total * 100
        logger.info(f"Epoch {epoch+1} | Loss {total_loss/len(loader):.4f} | Acc {acc:.1f}%")

    torch.save({
        "state_dict": model.state_dict(),
        "n_experts":  n_experts,
        "vocab_size": tokenizer.vocab_size
    }, router_path)

    logger.info(f"Router sauvegardé ✅")
