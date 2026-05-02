# inference.py
import logging
import torch
import torch.nn.functional as F
from functools import lru_cache
from pathlib import Path
from config import CFG
from model import ExpertTransformer
from train import RouterModel
from super_tokenizer import SuperTokenizer
from cluster import predict_cluster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoEInference:
    """
    Pipeline complet d'inférence :
    Prompt → Router → Top-K experts → Code généré
    """

    def __init__(self, device: str = None):
        self.device     = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer  = None
        self.router     = None
        self.n_experts  = 0
        self.models_dir = CFG.models_dir

        # Cache LRU : garde les N experts les plus utilisés en mémoire
        self._expert_cache: dict = {}
        self._cache_size = CFG.cache_size

    def load(self):
        """Charge tokenizer + router."""
        # Tokenizer
        self.tokenizer = SuperTokenizer()
        self.tokenizer.load(CFG.tokenizer_dir)
        logger.info("Tokenizer chargé ✅")

        # Router
        router_path = self.models_dir / "router.pt"
        if not router_path.exists():
            raise FileNotFoundError(f"Router introuvable : {router_path}")

        ckpt            = torch.load(router_path,
                                     map_location=self.device,
                                     weights_only=True)
        self.n_experts  = ckpt["n_experts"]
        vocab_size      = ckpt["vocab_size"]

        self.router     = RouterModel(
            vocab_size=vocab_size,
            n_experts=self.n_experts
        ).to(self.device)
        self.router.load_state_dict(ckpt["state_dict"])
        self.router.eval()
        logger.info(f"Router chargé : {self.n_experts} experts ✅")

    # ─── CACHE EXPERTS ───────────────────────────────────────────────────

    def _load_expert(self, expert_id: int):
        """Charge un expert avec cache LRU manuel."""
        if expert_id in self._expert_cache:
            return self._expert_cache[expert_id]

        path = self.models_dir / f"expert_{expert_id:04d}.pt"
        if not path.exists():
            return None

        try:
            ckpt  = torch.load(path, map_location=self.device,
                               weights_only=True)
            model = ExpertTransformer(
                vocab_size=ckpt.get("vocab_size", CFG.vocab_size)
            ).to(self.device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            dtype = torch.bfloat16 if (
                CFG.dtype == "bfloat16" and
                torch.cuda.is_available() and
                torch.cuda.is_bf16_supported()
            ) else torch.float32
            model = model.to(dtype)

            # Éviction si cache plein (LRU simplifié : supprime le premier)
            if len(self._expert_cache) >= self._cache_size:
                oldest = next(iter(self._expert_cache))
                del self._expert_cache[oldest]
                torch.cuda.empty_cache()

            self._expert_cache[expert_id] = model
            return model

        except Exception as e:
            logger.error(f"Erreur chargement expert {expert_id} : {e}")
            return None

    # ─── ROUTING ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _route(self, prompt: str, top_k: int = None) -> list:
        """Retourne les (expert_id, score) triés par score décroissant."""
        top_k = top_k or CFG.top_k_experts

        ids    = self.tokenizer.encode(prompt)[:256]
        x      = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.router(x)
        probs  = F.softmax(logits, dim=-1)[0]

        top_k  = min(top_k, self.n_experts)
        vals, idxs = torch.topk(probs, top_k)

        return [(int(i), float(v)) for i, v in zip(idxs, vals)]

    # ─── GÉNÉRATION ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 128,
                 temperature: float  = 0.8,
                 top_p: float        = 0.95,
                 strategy: str       = "best") -> str:
        """
        Génère du code à partir d'un prompt.

        strategy:
            "best"     → expert le plus confiant uniquement
            "ensemble" → moyenne pondérée logits top-5
            "vote"     → premier expert qui réussit
        """
        if self.tokenizer is None or self.router is None:
            raise RuntimeError("Appelle load() avant generate().")

        expert_scores = self._route(prompt)
        logger.debug(f"Top-3 experts : {expert_scores[:3]}")

        if strategy == "best":
            return self._generate_best(
                prompt, expert_scores[0][0],
                max_new_tokens, temperature, top_p
            )
        elif strategy == "ensemble":
            return self._generate_ensemble(
                prompt, expert_scores,
                max_new_tokens, temperature
            )
        elif strategy == "vote":
            return self._generate_vote(
                prompt, expert_scores,
                max_new_tokens, temperature, top_p
            )
        else:
            raise ValueError(f"Strategy inconnue : {strategy}")

    @torch.no_grad()
    def _generate_best(self, prompt: str, expert_id: int,
                       max_new: int, temperature: float,
                       top_p: float) -> str:
        model = self._load_expert(expert_id)
        if model is None:
            return f"# Expert {expert_id} non disponible"

        ids = self.tokenizer.encode(prompt)[:CFG.max_seq_len]
        x   = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = model.generate(x, max_new=max_new,
                             temperature=temperature, top_p=top_p)
        new_tokens = out[0, len(ids):].tolist()
        return prompt + self.tokenizer.decode(new_tokens)

    @torch.no_grad()
    def _generate_vote(self, prompt: str, expert_scores: list,
                       max_new: int, temperature: float,
                       top_p: float) -> str:
        for expert_id, score in expert_scores:
            result = self._generate_best(
                prompt, expert_id, max_new, temperature, top_p
            )
            if result and "non disponible" not in result:
                return f"# Expert {expert_id} (score={score:.3f})\n{result}"
        return "# Aucun expert disponible"

    @torch.no_grad()
    def _generate_ensemble(self, prompt: str, expert_scores: list,
                           max_new: int, temperature: float) -> str:
        """Moyenne pondérée des logits des top-5 experts."""
        ids       = self.tokenizer.encode(prompt)[:CFG.max_seq_len]
        generated = list(ids)

        dtype = (torch.bfloat16
                 if CFG.dtype == "bfloat16"
                 and torch.cuda.is_available()
                 else torch.float32)

        eos_id = self.tokenizer.eos_id

        for _ in range(max_new):
            x_cond     = torch.tensor(
                [generated[-CFG.max_seq_len:]],
                dtype=torch.long, device=self.device
            )
            all_logits = []

            for expert_id, score in expert_scores[:5]:
                model = self._load_expert(expert_id)
                if model is None:
                    continue

                with torch.autocast(
                    device_type=self.device.split(":")[0], dtype=dtype
                ):
                    logits, _ = model(x_cond)

                # Logits du dernier token, pondérés par score router
                all_logits.append(logits[:, -1, :].float() * score)

            if not all_logits:
                break

            # Moyenne pondérée
            combined = torch.stack(all_logits, dim=0).sum(dim=0) / temperature
            probs    = F.softmax(combined, dim=-1)

            # Top-p sampling
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cumulative  = torch.cumsum(sorted_p, dim=-1)
            remove_mask = (cumulative - sorted_p) > 0.95
            sorted_p[remove_mask] = 0.0
            sorted_p    = sorted_p / sorted_p.sum(dim=-1, keepdim=True)

            next_tok    = torch.multinomial(sorted_p, num_samples=1)
            next_tok    = sorted_idx.gather(-1, next_tok)
            tok         = next_tok[0].item()

            generated.append(tok)

            if tok == eos_id:
                break

        new_tokens = generated[len(ids):]
        return prompt + self.tokenizer.decode(new_tokens)

    def info(self) -> dict:
        experts_on_disk = len(list(self.models_dir.glob("expert_*.pt")))
        return {
            "experts_on_disk":   experts_on_disk,
            "experts_in_memory": len(self._expert_cache),
            "cache_size":        self._cache_size,
            "top_k":             CFG.top_k_experts,
            "device":            self.device,
            "vocab_size":        CFG.vocab_size
        }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    moe = MoEInference()
    moe.load()

    print(f"\n Info : {moe.info()}\n")

    tests = [
        "def trier_liste(lst):",
        "class Animal:\n    def __init__(self, nom):",
        "async def fetch(url):",
        "def fibonacci(n):",
    ]

    for prompt in tests:
        print(f"\n{'='*50}")
        print(f"PROMPT : {prompt}")
        print(f"{'─'*50}")
        result = moe.generate(prompt, max_new_tokens=80, strategy="best")
        print(result)
