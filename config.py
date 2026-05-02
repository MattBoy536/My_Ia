# config.py
from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class Config:
    # === DONNÉES ===
    n_samples        : int   = 500_000
    min_code_len     : int   = 50
    max_code_len     : int   = 50_000

    # === MODÈLE ===
    vocab_size       : int   = 8_000
    d_model          : int   = 512
    n_heads          : int   = 8
    n_layers         : int   = 6
    ffn_mult         : int   = 4
    max_seq_len      : int   = 512
    dropout          : float = 0.1

    # === ENTRAÎNEMENT ===
    batch_size       : int   = 16
    learning_rate    : float = 3e-4
    warmup_steps     : int   = 300
    max_steps        : int   = 3_000
    weight_decay     : float = 0.1
    grad_clip        : float = 1.0
    grad_accum       : int   = 2

    # === H100 ===
    dtype            : str   = "bfloat16"
    compile_model    : bool  = True
    tf32             : bool  = True
    parallel_experts : int   = 50

    # === CLUSTERING ===
    n_clusters       : int   = 2_000
    embedding_dim    : int   = 768
    min_cluster_size : int   = 50

    # === ROUTER ===
    top_k_experts    : int   = 20
    router_hidden    : int   = 512
    router_epochs    : int   = 10

    # === CACHE INFERENCE ===
    cache_size       : int   = 100   # experts en mémoire max

    # === CHEMINS ===
    data_dir         : Path  = field(default_factory=lambda: Path("./data"))
    models_dir       : Path  = field(default_factory=lambda: Path("./models"))
    cluster_dir      : Path  = field(default_factory=lambda: Path("./clusters"))
    tokenizer_dir    : Path  = field(default_factory=lambda: Path("./tokenizer"))
    log_dir          : Path  = field(default_factory=lambda: Path("./logs"))

    save_every       : int   = 500

    def __post_init__(self):
        # Crée tous les dossiers automatiquement
        for d in [self.data_dir, self.models_dir,
                  self.cluster_dir, self.tokenizer_dir,
                  self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Active TF32 si dispo
        if self.tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True

CFG = Config()
