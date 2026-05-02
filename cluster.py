# cluster.py
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from config import CFG
from ast_analyzer import ASTAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_codes() -> list:
    """Charge tous les codes depuis le JSONL."""
    data_file = CFG.data_dir / "python_code.jsonl"
    codes     = []

    with open(data_file, encoding="utf-8") as f:
        for line in tqdm(f, desc="Chargement codes"):
            try:
                codes.append(json.loads(line)["code"])
            except Exception:
                continue

    logger.info(f"{len(codes)} codes chargés ✅")
    return codes


def build_feature_matrix(codes: list) -> np.ndarray:
    """Construit la matrice de features AST."""
    cache_path = CFG.cluster_dir / "features.npy"

    if cache_path.exists():
        logger.info("Features déjà calculées, chargement du cache ✅")
        return np.load(cache_path)

    analyzer = ASTAnalyzer()
    features = []

    for code in tqdm(codes, desc="Extraction features"):
        try:
            vec = analyzer.feature_vector(code)
            features.append(vec)
        except Exception:
            features.append([0] * len(ASTAnalyzer.FEATURE_KEYS))

    matrix = np.array(features, dtype=np.float32)
    np.save(cache_path, matrix)
    logger.info(f"Features sauvegardées : {matrix.shape} ✅")
    return matrix


def cluster_codes(codes: list, tokenizer=None) -> dict:
    """
    Cluster les codes en n_clusters groupes.
    Retourne dict : cluster_id → liste de codes
    """
    clusters_cache = CFG.cluster_dir / "clusters.pkl"

    if clusters_cache.exists():
        logger.info("Clusters déjà calculés, chargement ✅")
        with open(clusters_cache, "rb") as f:
            return pickle.load(f)

    logger.info(f"Clustering de {len(codes)} codes en {CFG.n_clusters} clusters...")

    # Features AST
    matrix = build_feature_matrix(codes)

    # Normalisation
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)

    # Sauvegarde scaler pour inférence
    with open(CFG.cluster_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # KMeans (MiniBatch = rapide sur 500K exemples)
    logger.info("KMeans clustering...")
    kmeans = MiniBatchKMeans(
        n_clusters=CFG.n_clusters,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        random_state=42,
        verbose=1
    )
    labels = kmeans.fit_predict(matrix)

    # Sauvegarde du modèle kmeans
    with open(CFG.cluster_dir / "kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    # Construction du dict clusters
    clusters: dict[int, list] = {}
    for code, label in zip(codes, labels):
        clusters.setdefault(int(label), []).append(code)

    # Stats
    sizes = [len(v) for v in clusters.values()]
    logger.info(f"Clusters créés : {len(clusters)}")
    logger.info(f"Taille min : {min(sizes)} | max : {max(sizes)} | moy : {np.mean(sizes):.0f}")

    # Sauvegarde
    with open(clusters_cache, "wb") as f:
        pickle.dump(clusters, f)

    logger.info("Clusters sauvegardés ✅")
    return clusters


def predict_cluster(code: str) -> int:
    """Prédit le cluster d'un nouveau code (pour inférence)."""
    analyzer  = ASTAnalyzer()
    scaler_p  = CFG.cluster_dir / "scaler.pkl"
    kmeans_p  = CFG.cluster_dir / "kmeans.pkl"

    if not scaler_p.exists() or not kmeans_p.exists():
        raise FileNotFoundError("Clustering non entraîné.")

    with open(scaler_p, "rb") as f:
        scaler = pickle.load(f)
    with open(kmeans_p, "rb") as f:
        kmeans = pickle.load(f)

    vec    = np.array([analyzer.feature_vector(code)], dtype=np.float32)
    vec    = scaler.transform(vec)
    label  = kmeans.predict(vec)[0]
    return int(label)


if __name__ == "__main__":
    codes    = load_codes()
    clusters = cluster_codes(codes)
    print(f"Total clusters : {len(clusters)}")
