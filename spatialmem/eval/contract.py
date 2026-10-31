"""Prediction contract every mapper emits (docs/CONTRACT.md). Frozen 2026-10-27."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PREDICATES = ("on", "in", "next_to")
# Never scored as objects or relation endpoints, on either side (docs/CONTRACT.md, docs/RELATIONS.md).
STRUCTURAL = ("wall", "floor", "ceiling", "undefined")


@dataclass
class PredObject:
    id: int
    label: str
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    score: float = 1.0

    def __post_init__(self) -> None:
        self.aabb_min = np.asarray(self.aabb_min, dtype=np.float64).reshape(3)
        self.aabb_max = np.asarray(self.aabb_max, dtype=np.float64).reshape(3)
        if np.any(self.aabb_max < self.aabb_min):
            raise ValueError(f"object {self.id}: aabb_max < aabb_min")


@dataclass
class PredRelation:
    subj: int
    pred: str
    obj: int

    def __post_init__(self) -> None:
        if self.pred not in PREDICATES:
            raise ValueError(f"unknown predicate {self.pred!r}")

    def key(self) -> tuple[int, str, int]:
        """Canonical key: next_to is unordered."""
        if self.pred == "next_to":
            a, b = sorted((self.subj, self.obj))
            return (a, self.pred, b)
        return (self.subj, self.pred, self.obj)


@dataclass
class Prediction:
    scene: str
    objects: list[PredObject]
    relations: list[PredRelation] = field(default_factory=list)
    sem_points: np.ndarray | None = None
    sem_labels: np.ndarray | None = None

    def boxes(self) -> np.ndarray:
        """(N, 2, 3) array of [min, max]."""
        if not self.objects:
            return np.zeros((0, 2, 3))
        return np.stack([np.stack([o.aabb_min, o.aabb_max]) for o in self.objects])

    def to_json(self, path: str | Path | None = None) -> dict:
        """Objects + relations as JSON; semantic arrays (if any) go to a `<path>.sem.npz` sidecar."""
        d = {
            "scene": self.scene,
            "objects": [
                {"id": o.id, "label": o.label, "aabb_min": o.aabb_min.tolist(),
                 "aabb_max": o.aabb_max.tolist(), "score": o.score}
                for o in self.objects
            ],
            "relations": [{"subj": r.subj, "pred": r.pred, "obj": r.obj} for r in self.relations],
        }
        if path is not None:
            Path(path).write_text(json.dumps(d))
            if self.sem_points is not None:
                np.savez_compressed(str(path) + ".sem.npz", points=self.sem_points, labels=self.sem_labels)
        return d

    @classmethod
    def from_json(cls, d: dict | str | Path) -> "Prediction":
        sem = None
        if not isinstance(d, dict):
            side = Path(str(d) + ".sem.npz")
            sem = np.load(side) if side.exists() else None
            d = json.loads(Path(d).read_text())
        return cls(
            scene=d["scene"],
            objects=[PredObject(**{k: o[k] for k in ("id", "label", "aabb_min", "aabb_max", "score") if k in o})
                     for o in d["objects"]],
            relations=[PredRelation(**r) for r in d.get("relations", [])],
            sem_points=None if sem is None else sem["points"],
            sem_labels=None if sem is None else sem["labels"],
        )


# GT has the same shape; per-vertex labels live alongside.
@dataclass
class GroundTruth(Prediction):
    vertices: np.ndarray | None = None        # (V, 3) GT points for the semantic protocol
    vertex_labels: np.ndarray | None = None   # (V,) int protocol class index (-1 = unlabelled)
