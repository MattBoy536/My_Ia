# collect_data.py
import os
import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_valid_python(code: str) -> bool:
    import ast
    if not (CFG.min_code_len <= len(code) <= CFG.max_code_len):
        return False
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def collect_data():
    output_file = CFG.data_dir / "python_code.jsonl"

    # Vérifie si déjà fait
    if output_file.exists():
        count = sum(1 for _ in open(output_file, encoding="utf-8"))
        if count >= CFG.n_samples:
            logger.info(f"Données déjà collectées : {count} exemples ✅")
            return
        logger.info(f"Reprise : {count} exemples déjà là, on continue...")
        start_from = count
    else:
        start_from = 0

    # Auth HuggingFace automatique si token présent
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("HuggingFace auth OK ✅")
    else:
        logger.warning(
            "HF_TOKEN non défini → utilisation de codeparrot/github-code "
            "(pas besoin d'auth)"
        )

    logger.info(f"Collecte de {CFG.n_samples} exemples Python...")

    # Dataset sans auth requis
    try:
        dataset = load_dataset(
            "codeparrot/github-code",
            languages=["Python"],
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Erreur chargement dataset : {e}")
        raise

    collected = start_from
    skipped   = 0

    with open(output_file, "a", encoding="utf-8") as f:
        pbar = tqdm(total=CFG.n_samples, initial=start_from, desc="Collecte")

        for i, sample in enumerate(dataset):
            if collected >= CFG.n_samples:
                break

            # Skip les exemples déjà collectés si reprise
            if i < start_from:
                continue

            code = sample.get("code", sample.get("content", ""))

            if not is_valid_python(code):
                skipped += 1
                continue

            record = {"id": collected, "code": code, "size": len(code)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            collected += 1
            pbar.update(1)

        pbar.close()

    logger.info(f"Collecte terminée : {collected} exemples ✅")
    logger.info(f"Skippés (invalides) : {skipped}")


if __name__ == "__main__":
    collect_data()
