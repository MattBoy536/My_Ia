# ast_analyzer.py
import ast
import json
import logging
from pathlib import Path
from tqdm import tqdm
from config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASTAnalyzer:
    """
    Extrait les features structurelles du code Python
    via l'AST pour aider le clustering.
    """

    FEATURE_KEYS = [
        "has_class", "has_async", "has_decorator",
        "has_comprehension", "has_lambda", "has_try_except",
        "has_with", "has_yield", "has_import", "has_type_hint",
        "n_functions", "n_classes", "n_loops", "n_conditions",
        "depth", "n_lines"
    ]

    def analyze(self, code: str) -> dict:
        """Retourne un dict de features pour un bout de code."""
        features = {k: 0 for k in self.FEATURE_KEYS}
        features["n_lines"] = code.count("\n") + 1

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return features

        features["depth"] = self._tree_depth(tree)

        for node in ast.walk(tree):
            t = type(node)

            if t == ast.ClassDef:
                features["has_class"]    = 1
                features["n_classes"]   += 1

            elif t in (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith):
                features["has_async"]    = 1

            elif t == ast.FunctionDef:
                features["n_functions"] += 1

                # Décorateurs
                if node.decorator_list:
                    features["has_decorator"] = 1

                # Type hints
                if node.returns or any(
                    a.annotation for a in node.args.args
                ):
                    features["has_type_hint"] = 1

            elif t in (ast.For, ast.While):
                features["n_loops"] += 1

            elif t == ast.If:
                features["n_conditions"] += 1

            elif t in (ast.ListComp, ast.DictComp,
                       ast.SetComp, ast.GeneratorExp):
                features["has_comprehension"] = 1

            elif t == ast.Lambda:
                features["has_lambda"] = 1

            elif t in (ast.Try,):
                features["has_try_except"] = 1

            elif t == ast.With:
                features["has_with"] = 1

            elif t in (ast.Yield, ast.YieldFrom):
                features["has_yield"] = 1

            elif t in (ast.Import, ast.ImportFrom):
                features["has_import"] = 1

        return features

    def _tree_depth(self, tree) -> int:
        """Calcule la profondeur max de l'AST."""
        def depth(node):
            children = list(ast.iter_child_nodes(node))
            if not children:
                return 0
            return 1 + max(depth(c) for c in children)
        try:
            return depth(tree)
        except RecursionError:
            return 0

    def feature_vector(self, code: str) -> list:
        """Retourne la feature list dans l'ordre FEATURE_KEYS."""
        d = self.analyze(code)
        return [d[k] for k in self.FEATURE_KEYS]


def analyze_all():
    """Analyse tous les fichiers collectés et sauvegarde les features."""
    input_file  = CFG.data_dir    / "python_code.jsonl"
    output_file = CFG.cluster_dir / "features.jsonl"

    if output_file.exists():
        n = sum(1 for _ in open(output_file))
        logger.info(f"Features déjà calculées : {n} ✅")
        return

    analyzer = ASTAnalyzer()
    total    = sum(1 for _ in open(input_file, encoding="utf-8"))

    with open(input_file,  encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total, desc="Analyse AST"):
            try:
                record   = json.loads(line)
                features = analyzer.analyze(record["code"])
                fout.write(json.dumps({
                    "id":       record["id"],
                    "features": features
                }) + "\n")
            except Exception as e:
                logger.debug(f"Skip ligne : {e}")

    logger.info("Analyse AST terminée ✅")


if __name__ == "__main__":
    analyze_all()
