"""Run the protocol's own metric code (OVO eval_utils, tberriel/OVO@e0d16f7, MIT) as the G4 reference.

The vendored file imports torch / seaborn / pandas at module level; the metric functions themselves are
numpy-only, so they are extracted verbatim by AST and executed — the reference is OVO's literal code.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "third_party" / "ovo_eval_utils.py"
_FUNCS = ("update_confmat", "get_iou", "iou_acc_from_confmat")


def _load() -> dict:
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _FUNCS]
    assert {n.name for n in keep} == set(_FUNCS), "OVO reference functions not found"
    mod = ast.Module(body=keep, type_ignores=[])
    from typing import Any, Dict, List, Tuple  # noqa: F401  (names used in the annotations)
    ns: dict = {"np": np, "List": List, "Tuple": Tuple, "Dict": Dict, "Any": Any}
    exec(compile(mod, str(_SRC), "exec"), ns)
    return ns


_NS = _load()


def ovo_metrics(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore: list[int]) -> dict:
    """mIoU / mAcc / f-mIoU exactly as OVO's eval_semantics computes them (unrounded)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.ulonglong)
    keep = np.asarray(gt) >= 0
    _NS["update_confmat"](cm, np.asarray(gt)[keep].tolist(), np.asarray(pred)[keep].tolist(), ignore)
    iou, iou_m, w, acc, acc_m = _NS["iou_acc_from_confmat"](cm, num_classes, ignore, True, False, None)
    return {
        "miou": float(np.mean(iou[iou_m])),
        "macc": float(np.mean(acc[acc_m])),
        "fmiou": float(np.sum(iou[iou_m] * w[iou_m]) / w[iou_m].sum()),
    }
