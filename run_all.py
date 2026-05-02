# run_all.py
import json
import logging
import time
import torch
from pathlib import Path
from config import CFG
from collect_data    import collect_data
from super_tokenizer import build_tokenizer
from cluster         import load_codes, cluster_codes
from train           import train_all_experts, train_router
from inference       import MoEInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CFG.log_dir / "run.log")
    ]
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU : {name} ({vram:.1f} GB VRAM)")
        return torch.device("cuda")
    logger.warning("Pas de GPU → CPU (très lent)")
    return torch.device("cpu")


def print_summary(results: list, t_total: float):
    done    = sum(1 for r in results if r["status"] == "done")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    oom     = sum(1 for r in results if r["status"] == "oom")
    errors  = sum(1 for r in results if r["status"] == "error")

    losses  = [r["loss"] for r in results
               if r["status"] == "done" and "loss" in r]
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    print("\n" + "="*60)
    print("           RÉSUMÉ ENTRAÎNEMENT")
    print("="*60)
    print(f"  Experts entraînés   : {done}")
    print(f"  Skippés             : {skipped}")
    print(f"  OOM                 : {oom}")
    print(f"  Erreurs             : {errors}")
    print(f"  Loss moyenne        : {avg_loss:.4f}")
    print(f"  Temps total         : {t_total/3600:.2f}h")
    print("="*60 + "\n")


def main():
    """Pipeline complet d'entraînement unsupervised MoE."""
    t_start = time.time()
    device  = get_device()

    logger.info("="*60)
    logger.info("         PIPELINE ENTRAÎNEMENT MoE")
    logger.info("="*60)

    # ── 1. COLLECTE DES DONNÉES ─────────────────────────────────────────
    logger.info("\n[1/6] Collecte des données Python...")
    t1 = time.time()
    try:
        collect_data()
        elapsed = time.time() - t1
        logger.info(f"✅ Collecte terminée en {elapsed/60:.1f} min")
    except Exception as e:
        logger.error(f"❌ Erreur collecte : {e}")
        raise

    # ── 2. CONSTRUCTION TOKENIZER ───────────────────────────────────────
    logger.info("\n[2/6] Construction tokenizer...")
    t2 = time.time()
    try:
        tokenizer = build_tokenizer()
        elapsed   = time.time() - t2
        logger.info(f"✅ Tokenizer prêt en {elapsed/60:.1f} min")
        logger.info(f"   Vocab size : {len(tokenizer.token2id)}")
        logger.info(f"   Compression : {tokenizer.compression_ratio:.2f}x")
    except Exception as e:
        logger.error(f"❌ Erreur tokenizer : {e}")
        raise

    # ── 3. CLUSTERING ───────────────────────────────────────────────────
    logger.info("\n[3/6] Clustering des codes...")
    t3 = time.time()
    try:
        codes    = load_codes()
        clusters = cluster_codes(codes, tokenizer)
        elapsed  = time.time() - t3
        logger.info(f"✅ Clustering en {elapsed/60:.1f} min")
        logger.info(f"   {len(clusters)} clusters créés")
        
        sizes = [len(v) for v in clusters.values()]
        logger.info(
            f"   Taille : min={min(sizes)} max={max(sizes)} "
            f"moy={sum(sizes)/len(sizes):.0f}"
        )
    except Exception as e:
        logger.error(f"❌ Erreur clustering : {e}")
        raise

    # ── 4. ENTRAÎNEMENT DES EXPERTS ─────────────────────────────────────
    logger.info("\n[4/6] Entraînement des experts...")
    logger.info(f"   Device : {device}")
    logger.info(f"   Config : batch={CFG.batch_size}, "
                f"seq_len={CFG.max_seq_len}, dtype={CFG.dtype}")

    t4 = time.time()
    try:
        results = train_all_experts(clusters, tokenizer, str(device),
                                    CFG.models_dir)
        elapsed = time.time() - t4
        logger.info(f"✅ Experts entraînés en {elapsed/3600:.2f}h")

        # Sauvegarde rapport
        report = {
            "total_experts": len(results),
            "completed": sum(1 for r in results if r["status"] == "done"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "oom": sum(1 for r in results if r["status"] == "oom"),
            "errors": sum(1 for r in results if r["status"] == "error"),
            "avg_loss": sum(
                r.get("loss", 0) for r in results
                if r["status"] == "done"
            ) / max(sum(1 for r in results if r["status"] == "done"), 1)
        }
        with open(CFG.log_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print_summary(results, time.time() - t_start)

    except KeyboardInterrupt:
        logger.warning("⚠️ Entraînement interrompu par utilisateur")
        raise
    except Exception as e:
        logger.error(f"❌ Erreur entraînement experts : {e}")
        raise

    # ── 5. ENTRAÎNEMENT DU ROUTER ───────────────────────────────────────
    logger.info("\n[5/6] Entraînement du Router...")
    t5 = time.time()
    try:
        train_router(clusters, tokenizer, str(device))
        elapsed = time.time() - t5
        logger.info(f"✅ Router entraîné en {elapsed/60:.1f} min")
    except Exception as e:
        logger.error(f"❌ Erreur router : {e}")
        raise

    # ── 6. TEST INFÉRENCE ───────────────────────────────────────────────
    logger.info("\n[6/6] Test inférence...")
    t6 = time.time()
    try:
        moe = MoEInference(device=str(device))
        moe.load()

        logger.info(f"Infos système : {moe.info()}")

        # Tests rapides
        test_prompts = [
            "def hello_world():",
            "class Person:\n    def __init__(self, name):",
            "def fibonacci(n):",
        ]

        logger.info("\n--- Tests de génération ---")
        for prompt in test_prompts:
            logger.info(f"\nPrompt : {prompt}")
            try:
                result = moe.generate(
                    prompt,
                    max_new_tokens=50,
                    strategy="best"
                )
                logger.info(f"Résultat : {result[:120]}...")
            except Exception as e:
                logger.warning(f"Test échoué : {e}")

        elapsed = time.time() - t6
        logger.info(f"\n✅ Tests inférence en {elapsed:.1f}s")

    except Exception as e:
        logger.warning(f"⚠️ Tests inférence échoués (pas bloquant) : {e}")

    # ── RÉSUMÉ FINAL ────────────────────────────────────────────────────
    t_total = time.time() - t_start

    logger.info("\n" + "="*60)
    logger.info("           PIPELINE COMPLET")
    logger.info("="*60)
    logger.info(f"  Collecte        : {(t2-t1)/60:6.1f} min")
    logger.info(f"  Tokenizer       : {(t3-t2)/60:6.1f} min")
    logger.info(f"  Clustering      : {(t4-t3)/60:6.1f} min")
    logger.info(f"  Experts         : {(t5-t4)/3600:6.2f}h")
    logger.info(f"  Router          : {(t6-t5)/60:6.1f} min")
    logger.info(f"  ─────────────────────────")
    logger.info(f"  TOTAL           : {t_total/3600:6.2f}h")
    logger.info("="*60)

    logger.info("\n✅ Pipeline complet réussi !")
    logger.info(f"Modèles sauvegardés dans : {CFG.models_dir}")
    logger.info(f"Logs dans : {CFG.log_dir}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Pipeline interrompu")
        exit_code = 130
    except Exception as e:
        logger.error(f"\n❌ Erreur fatale : {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    exit(exit_code)
