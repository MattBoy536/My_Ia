# config.py
from dataclasses import dataclass
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

    # === ENTRAÎNEMENT (2x A100 40GB) ===
    batch_size       : int   = 32
    learning_rate    : float = 3e-4
    warmup_steps     : int   = 300
    max_steps        : int   = 3_000
    weight_decay     : float = 0.1
    grad_clip        : float = 1.0
    grad_accum       : int   = 1

    # === MULTI-GPU ===
    dtype            : str   = "bfloat16"
    compile_model    : bool  = True
    tf32             : bool  = True
    n_gpus           : int   = 2
    expert_timeout   : int   = 1800   # 30min max par expert

    # === CLUSTERING ===
    n_clusters       : int   = 2_000
    embedding_dim    : int   = 768
    min_cluster_size : int   = 50

    # === ROUTER ===
    top_k_experts    : int   = 20
    router_hidden    : int   = 512
    router_epochs    : int   = 5
    router_batch     : int   = 64

    # === INFERENCE ===
    cache_size       : int   = 1024
    temperature      : float = 0.8
    top_p            : float = 0.9
    max_gen_tokens   : int   = 256

    # === CHEMINS ===
    base_dir         : Path  = Path("./workspace")
    data_dir         : Path  = Path("./workspace/data")
    tokenizer_dir    : Path  = Path("./workspace/tokenizer")
    cluster_dir      : Path  = Path("./workspace/clusters")
    experts_dir      : Path  = Path("./workspace/experts")
    router_dir       : Path  = Path("./workspace/router")
    log_dir          : Path  = Path("./workspace/logs")

    # === HUGGINGFACE ===
    hf_dataset       : str   = "codeparrot/codeparrot-clean"
    hf_token_env     : str   = "HF_TOKEN"

    def __post_init__(self):
        for p in [self.base_dir, self.data_dir, self.tokenizer_dir,
                  self.cluster_dir, self.experts_dir, self.router_dir,
                  self.log_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def torch_dtype(self):
        return {
            "float32" : torch.float32,
            "float16" : torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]


CFG = Config()
