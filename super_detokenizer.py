# super_detokenizer.py
"""Reconstruit du code Python à partir de super-tokens."""
from typing import List

class SuperDetokenizer:
    """
    Convertit super-tokens → squelette de code Python.
    Note : la reconstruction est approximative (templates).
    Les détails (noms variables, valeurs) doivent venir
    d'un modèle de raffinement séparé.
    """
    
    TEMPLATES = {
        "FUNC":      "def {name}({args}):\n    {body}",
        "ASYNC_FUNC":"async def {name}({args}):\n    {body}",
        "CLASS":     "class {name}({bases}):\n    {body}",
        "FOR":       "for {var} in {iter}:\n    {body}",
        "WHILE":     "while {cond}:\n    {body}",
        "IF":        "if {cond}:\n    {body}",
        "TRY":       "try:\n    {body}\nexcept Exception:\n    pass",
        "WITH":      "with {ctx} as {var}:\n    {body}",
        "IMPORT":    "import {module}",
        "RETURN":    "return {value}",
        "ASSIGN":    "{target} = {value}",
        "CALL":      "{func}({args})",
        "LIST_COMP": "[{expr} for {var} in {iter}]",
        "DICT_COMP": "{{{k}: {v} for {var} in {iter}}}",
        "LAMBDA":    "lambda {args}: {expr}",
    }
    
    def __init__(self):
        self.counter = 0
    
    def _placeholder(self, kind: str) -> str:
        self.counter += 1
        return f"{kind}_{self.counter}"
    
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruit un squelette de code."""
        self.counter = 0
        lines = []
        indent = 0
        
        for tok in tokens:
            tok_clean = tok.strip("<>")
            kind = tok_clean.split(":")[0]
            
            if kind in ("PAD", "BOS", "EOS", "UNK", "MASK"):
                continue
            
            line = self._render(kind, tok_clean)
            if line:
                lines.append("    " * indent + line)
        
        return "\n".join(lines)
    
    def _render(self, kind: str, full_tok: str) -> str:
        if kind == "FUNC":
            return f"def {self._placeholder('func')}(...):"
        if kind == "CLASS":
            return f"class {self._placeholder('Cls')}:"
        if kind == "FOR":
            return f"for {self._placeholder('i')} in ...:"
        if kind == "WHILE":
            return "while ...:"
        if kind == "IF":
            return "if ...:"
        if kind == "TRY":
            return "try:"
        if kind == "WITH":
            return f"with ... as {self._placeholder('ctx')}:"
        if kind == "IMPORT":
            mod = full_tok.split(":")[-1] if ":" in full_tok else "module"
            return f"import {mod}"
        if kind == "RETURN":
            return "return ..."
        if kind == "ASSIGN":
            return f"{self._placeholder('var')} = ..."
        if kind == "CALL":
            func = full_tok.split(":")[-1] if ":" in full_tok else "func"
            return f"{func}(...)"
        if kind == "LIST_COMP":
            return "[... for ... in ...]"
        if kind == "DICT_COMP":
            return "{... : ... for ... in ...}"
        if kind == "LAMBDA":
            return "lambda ...: ..."
        return f"# {kind}"
