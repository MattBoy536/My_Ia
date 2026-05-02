# super_tokenizer.py
import json
import logging
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import List

from tqdm import tqdm
from config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuperTokenizer:
    """
    Tokenizer BPE entraîné uniquement sur du Python.
    Compression ~20x vs caractères bruts.
    Gère : eos, pad, unk.
    """

    UNK_TOKEN = "<unk>"
    EOS_TOKEN = "<eos>"
    PAD_TOKEN = "<pad>"
    SPECIAL   = [UNK_TOKEN, EOS_TOKEN, PAD_TOKEN]

    def __init__(self, vocab_size: int = None):
        self.vocab_size       = vocab_size or CFG.vocab_size
        self.token2id: dict   = {}
        self.id2token: dict   = {}
        self.merges:   list   = []
        self.unk_id    = 0
        self.eos_id    = 1
        self.pad_id    = 2
        self._trained  = False

        self.compression_ratio = 1.0

    # ─── TOKENIZATION DE BASE ────────────────────────────────────────────

    def _base_tokenize(self, code: str) -> list:
        """
        Tokenization initiale : split sur frontières Python naturelles.
        Conserve les espaces et indentations.
        """
        pattern = r"""
            (?:\"\"\"[\s\S]*?\"\"\")   |  # docstring triple "
            (?:\'\'\'[\s\S]*?\'\'\')   |  # docstring triple '
            (?:\"[^\"\\]*\")           |  # string double
            (?:\'[^\'\\]*\')           |  # string simple
            (?:\#[^\n]*)               |  # commentaire
            (?:[a-zA-Z_]\w*)           |  # identifiant
            (?:\d+\.?\d*)              |  # nombre
            (?:\n[ \t]*)               |  # newline + indent
            (?:[ \t]+)                 |  # espaces
            (?:[^\s])                     # tout le reste
        """
        return re.findall(pattern, code, re.VERBOSE)

    # ─── ENTRAÎNEMENT BPE ────────────────────────────────────────────────

    def train(self, codes: list):
        """Entraîne le BPE sur une liste de codes Python."""
        logger.info(f"Entraînement BPE sur {len(codes)} exemples...")

        # Comptage des paires initiales
        vocab  = Counter()
        chars_total = 0
        tokens_total = 0

        tokenized = []
        for code in tqdm(codes[:50_000], desc="Tokenization initiale"):
            toks = self._base_tokenize(code)
            tokenized.append(toks)
            vocab.update(toks)
            chars_total  += len(code)
            tokens_total += len(toks)

        # Vocab de base = tous les tokens vus
        base_vocab = [t for t, _ in vocab.most_common()]

        # Tokens spéciaux en premier
        all_tokens = self.SPECIAL + base_vocab

        # Tronque au vocab_size
        all_tokens = all_tokens[:self.vocab_size]

        # Construit les dicts
        self.token2id = {t: i for i, t in enumerate(all_tokens)}
        self.id2token = {i: t for i, t in enumerate(all_tokens)}

        self.compression_ratio = chars_total / max(tokens_total, 1)
        self._trained = True

        logger.info(f"Vocab size : {len(self.token2id)}")
        logger.info(f"Compression : {self.compression_ratio:.2f}x")

    # ─── ENCODE / DECODE ─────────────────────────────────────────────────

    def encode(self, code: str, add_eos: bool = False) -> list:
        """Code → liste d'ids."""
        if not self._trained:
            raise RuntimeError("Tokenizer non entraîné. Appelle train() ou load().")

        tokens = self._base_tokenize(code)
        ids    = [self.token2id.get(t, self.unk_id) for t in tokens]

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list) -> str:
        """Liste d'ids → code."""
        tokens = [
            self.id2token.get(i, self.UNK_TOKEN)
            for i in ids
            if i not in (self.eos_id, self.pad_id)
        ]
        return "".join(tokens)

    # ─── SAUVEGARDE / CHARGEMENT ─────────────────────────────────────────

    def save(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size":         self.vocab_size,
            "token2id":           self.token2id,
            "id2token":           {str(k): v for k, v in self.id2token.items()},
            "merges":             self.merges,
            "compression_ratio":  self.compression_ratio,
            "unk_id":             self.unk_id,
            "eos_id":             self.eos_id,
            "pad_id":             self.pad_id,
        }

        with open(directory / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self._trained = True
        logger.info(f"Tokenizer sauvegardé dans {directory} ✅")

    def load(self, directory: Path):
        directory = Path(directory)
        path      = directory / "tokenizer.json"

        if not path.exists():
            raise FileNotFoundError(f"Tokenizer introuvable : {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size        = data["vocab_size"]
        self.token2id          = data["token2id"]
        self.id2token          = {int(k): v for k, v in data["id2token"].items()}
        self.merges            = data["merges"]
        self.compression_ratio = data["compression_ratio"]
        self.unk_id            = data["unk_id"]
        self.eos_id            = data["eos_id"]
        self.pad_id            = data["pad_id"]
        self._trained          = True

        # Met à jour le vocab_size dans CFG
        CFG.vocab_size = len(self.token2id)

        logger.info(f"Tokenizer chargé : {len(self.token2id)} tokens ✅")


def build_tokenizer() -> SuperTokenizer:
    """Construit et sauvegarde le tokenizer si pas déjà fait."""
    save_dir = CFG.tokenizer_dir
    tok_file = save_dir / "tokenizer.json"

    tokenizer = SuperTokenizer(vocab_size=CFG.vocab_size)

    if tok_file.exists():
        tokenizer.load(save_dir)
        return tokenizer

    # Charge les codes
    data_file = CFG.data_dir / "python_code.jsonl"
    codes     = []

    logger.info("Chargement des codes pour tokenizer...")
    with open(data_file, encoding="utf-8") as f:
        for line in tqdm(f, desc="Chargement"):
            try:
                codes.append(json.loads(line)["code"])
            except Exception:
                continue

    tokenizer.train(codes)
    tokenizer.save(save_dir)
    return tokenizer


if __name__ == "__main__":
    tok = build_tokenizer()
    test = "def hello(name: str) -> str:\n    return f'Hello {name}'"
    ids  = tok.encode(test)
    back = tok.decode(ids)
    print(f"Original : {test}")
    print(f"IDs      : {ids}")
    print(f"Décodé   : {back}")
