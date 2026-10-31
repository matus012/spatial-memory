# Prediction contract — frozen 2026-10-27, before any score exists

Every mapper (ours, ConceptGraphs, OVO, GT-as-prediction) emits one `spatialmem.eval.contract.Prediction`
per scene. The scorer reads nothing else. Changing a field or a rule below = a dated entry here and in
`DEVIATIONS.md`.

```python
@dataclass
class PredObject:
    id: int                      # unique within the scene
    label: str                   # class name, from the scene's evaluation vocabulary (exact string)
    aabb_min: np.ndarray         # (3,) float64, world frame of the dataset, metres
    aabb_max: np.ndarray         # (3,) float64, aabb_max >= aabb_min componentwise
    score: float = 1.0           # confidence; not used by the metrics below

@dataclass
class PredRelation:
    subj: int                    # PredObject.id
    pred: str                    # "on" | "in" | "next_to"  (docs/RELATIONS.md)
    obj: int                     # PredObject.id

@dataclass
class Prediction:
    scene: str
    objects: list[PredObject]
    relations: list[PredRelation]
    sem_points: np.ndarray | None = None   # (P, 3) float, world frame — for the semantic protocol
    sem_labels: np.ndarray | None = None   # (P,) int, index into the protocol class list
```

GT has the same shape (`spatialmem.eval.contract.GroundTruth`: objects + relations + per-vertex labels).
JSON round-trip: `Prediction.to_json()` / `Prediction.from_json()`.

## Metrics (`spatialmem/eval/`)

**Objects, at τ ∈ {0.25, 0.5}.** IoU = 3D AABB IoU after padding extents < 2 cm to 2 cm (change log). Matching = one-to-one assignment that
**maximises the number of pairs with IoU ≥ τ**, ties broken by the larger IoU sum (Hungarian on weight
`1 + IoU/(N+1)` for eligible pairs, 0 otherwise; ineligible assigned pairs are discarded).
`TP` = matched pairs; precision = TP / #pred, recall = TP / #GT, F1 = 2PR / (P + R) (0 if P + R = 0).
Label accuracy@τ = fraction of matched pairs with `pred.label == gt.label` (exact string); undefined
(reported `nan`) when TP = 0. Structural / ignored classes are removed from **both** sides before
matching (same list as `docs/RELATIONS.md`).

**Relations.** Matching from objects @ τ = 0.25. A predicted relation `(s, p, o)` is a TP iff both
`s` and `o` are matched and `(m(s), p, m(o))` is a GT relation (`next_to` compared unordered).
Duplicates are counted once. precision = TP / #unique pred relations, recall = TP / #GT relations.
Relations touching a structural object are dropped with it.

**Semantic (Replica protocol).** Per-GT-point labels from `sem_points / sem_labels` transferred to the
GT points by the protocol's rule (OVO's eval code, `third_party/ovo_eval_utils.py`); confusion matrix over the protocol class
list; IoU_c = TP_c / (TP_c + FP_c + FN_c), Acc_c = TP_c / (TP_c + FN_c); mIoU and mAcc = means over
classes with TP_c + FP_c + FN_c > 0; f-mIoU = Σ_c (n_c / n) IoU_c with n_c = GT points of class c.
GT points whose class is ignored by the protocol are dropped before the matrix.

## Change log

- **2026-10-27b** (before any number on real data; code review findings 1, 6, 7; DEVIATIONS.md row 4):
  (1) **thin boxes** — before IoU every box extent < 2 cm is padded symmetrically to 2 cm, on both sides
  (`spatialmem.eval.boxes.MIN_EXTENT`); without it a flat GT object (picture, rug: one planar mesh segment)
  has IoU 0 with itself and could never be matched. (2) mIoU / mAcc averaging set corrected from "classes
  present in the GT" to "classes with TP + FP + FN > 0" — what OVO's code (and ours) does: a predicted but
  absent class counts with IoU 0. (3) structural filtering of both sides is done inside `score_objects` /
  `score_relations`; `ObjectScore.matches` holds object ids. (4) semantic arrays travel in a
  `<file>.sem.npz` sidecar; the mapper path is `spatialmem.eval.semantic.score_semantic` + `pooled`.
- **2026-10-27c — HSSD ground truth** (task 002, frozen before any HSSD number; DEVIATIONS.md row 6):
  (1) **objects** = Habitat rigid object instances of the scene (stage = walls / floor / ceiling = ID 0, not an
  object). **Decomposed parts** (`<hash>_part_<k>` templates, `objects/decomposed/`) of one parent template whose
  world AABBs are within 2 cm of each other form **one GT object** (connected components,
  `spatialmem.hssd.sim.group_parts`), rendered with one instance ID — a kitchen unit is one object, not its
  drawers and doors. (2) **AABB** = tight world AABB of the GLB vertices (union over a group's parts).
  (3) **label** = `semantics/objects.csv` `main_category` (the parent's, for parts), else the template's lexicon
  class (`object_config.json` `semantic_id` → `semantics/hssd-hab_semantic_lexicon.json`), else the WordNet
  synset head (`objects.csv` `wnsynsetkey`), else `undefined`; for a part group, a missing parent category
  falls back to the members' most common label (`spatialmem/hssd/labels.py`). `undefined` / `unknown` objects are
  rendered and ID'd but are not GT objects (3.9 % of placements in the 35 scenes). (4) up axis +Y; relations
  per `docs/RELATIONS.md` v2. (5) instance IDs 1..K in sorted-handle order of a group's first member.
- **2026-10-27d — HSSD instrument checks disclosed** (audit of task 002; no number changes): the 002 G4 chain
  check back-projects every 2nd pixel in u and v (`spatialmem.hssd.check.G4_STRIDE = 2`), and its ">= 500 points per
  object" threshold counts those subsampled points; G6 coverage counts full-resolution instance pixels (>= 200 in
  >= 1 frame).
- **2026-10-27e — HSSD canonical vocabulary + label procedure for all mappers** (task 003 step 1, ruling in
  DEVIATIONS.md row 7b; frozen before any mapper number).
  (1) **Canonical HSSD classes**: `configs/hssd_synonyms.yaml` groups + `configs/hssd_vocab_counts.json` (GT-object
  counts, 35 scenes) → `spatialmem.hssd.vocab.canonical` (canonical name = most frequent group member, ties
  alphabetical; no group mixes WALL_MOUNTED and other labels — asserted). 286 GT labels → **236 classes**. HSSD GT labels
  and predicted labels are canonicalised before object matching / label accuracy; relations are unaffected (the
  selfcheck recomputes them on canonical labels and they equal the exported ones, 35/35 scenes).
  (2) **Label of a predicted object** (every open-vocabulary method, baselines and ours): `label = argmax_c cos(f_obj,
  t_c)` where `f_obj` is the method's **own** per-object feature (as the method builds it) and `t_c` is the method's
  **own** text encoder's embedding of the prompt `"a photo of a {c}."` (one template for all methods; in `{c}` `_` and
  `-` become spaces) over the scene-set vocabulary: **Replica** = the 51 protocol classes
  (`third_party/ovo_replica_eval_info.yaml` `class_names_reduced`), **HSSD** = the 236 canonical classes
  (`spatialmem.hssd.vocab.vocabulary()`). No per-scene vocabulary, no thresholds, no "background" class.
  (3) **Relations of a prediction** = `spatialmem.gt.relations.compute_relations` (RELATIONS.md v2) on the method's
  boxes and labels (up = +z Replica, +y HSSD); a method's own scene-graph edges are not scored.
  (4) **Replica semantic metrics** for a published method additionally come from its own eval path (OVO `run_eval.py`
  default flags = Table II) to reproduce its paper number; our scorer is run on the same per-point predictions.
  *Record note (2026-10-28 01:10 (CET)):* this section was written at 2026-10-27 23:01 (CET) and named in commit
  `f4c76ee`, but the append failed silently (a preceding `ls` of a missing evidence dir aborted the `&&` chain) and the
  text only reached this file now, unchanged. Both baseline workers built their text embeddings from the task-file
  wording without the `_`/`-` → space rule; their labels are recomputed under (2) as written here before any score
  from our scorer is reported (003 report, finding F1).
- **2026-10-28f — Replica-dev ground truth** (task 006 step 1; scene list `configs/replica_dev.yaml` committed
  2faa726 before any dev number; this entry written 2026-10-28 23:35 (CET), still before any mapper or tuning number on a dev
  scene; DEVIATIONS.md row 12): the 10 Replica v1 scenes outside the protocol's 8 (apartment_0-2,
  frl_apartment_0-5, hotel_0). Objects, AABBs and relations v2 exactly as for the protocol scenes (`spatialmem.gt.replica`).
  Per-vertex semantic GT (no OVO txt exists for these scenes): each `mesh_semantic.ply` vertex takes the majority class of
  its faces (ties -> smallest class id), reduced to the protocol's 51 classes by the 001 map; unmapped -> -1 (ignored).
  Check on the protocol scenes, where OVO's txt exists: 98.59 % (room0) / 99.15 % (office0) index-wise agreement
  (001 measured 98.8-99.4 % between OVO's txt and mesh_semantic labels). Scored as split `replica_dev`
  (pooled like `replica`); GT-as-pred = 1.000 on every metric, k5 ceiling 0.9946-0.9971 per scene.
