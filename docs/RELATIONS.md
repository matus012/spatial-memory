# GT relation rules — frozen 2026-10-27, before any relation number exists

Relations are computed **geometrically from axis-aligned 3D boxes (AABBs)** in the scene's world frame,
by `spatialmem/gt/relations.py`. The same function is the reference any mapper may call on its own boxes,
so a relation scored as wrong is a box error or a predicate disagreement, never a rule mismatch.
Changing any rule or number below = a dated entry here and in `DEVIATIONS.md`.

## Frame and inputs

- `u` = the up axis of the dataset's world frame (Replica: determined from the data and asserted in
  `spatialmem/gt/replica.py` — floor below ceiling along `u`). `h1, h2` = the two horizontal axes.
- Box `A = [min, max]` per object. `top(A) = max_u`, `bottom(A) = min_u`.
- `footprint(A)` = the 2D rectangle `[min_h, max_h]`; `area(·)` its area; `∩` intersection.
- Endpoints: every GT object whose class is **not structural**. Structural (never an endpoint):
  `wall, floor, ceiling, undefined` plus any class the semantic protocol ignores
  (`third_party/ovo_replica_eval_info.yaml`). Objects with any box side < 1 cm are dropped (degenerate).

## Predicates (directed unless stated)

Let `gap = bottom(A) − top(B)`, `fo = area(footprint(A) ∩ footprint(B)) / area(footprint(A))`,
`ci = vol(A ∩ B) / vol(A)`.

1. **on(A, B)** — A rests on B's top surface:
   `−0.08 m ≤ gap ≤ 0.05 m` AND `fo ≥ 0.5` AND `top(A) > top(B)`.
   If several B qualify, keep only the one with the highest `top(B)` (ties → smaller `area(footprint(B))`,
   then smaller id). At most one `on` per A.
2. **in(A, B)** — A is inside B's box (shelf, cabinet, sofa):
   `ci ≥ 0.9` AND `vol(A) < vol(B)` AND NOT on(A, B).
   If several B qualify, keep only the smallest `vol(B)` (ties → smaller id). At most one `in` per A.
3. **next_to(A, B)** — **symmetric**, stored once as `(min id, max id)`:
   no on/in relation in either direction between A and B (any pair that *qualifies* under 1 or 2,
   before the one-per-A pruning), AND their `u`-intervals overlap
   (`min(top) − max(bottom) > 0`), AND the 2D distance between `footprint(A)` and `footprint(B)`
   (0 if they overlap) is `≤ 0.30 m`.

Predicate names in files and the contract: `"on"`, `"in"`, `"next_to"`.

## Known limits (accepted at demo grade; checked on the top-down figure + 10 sampled relations)

- A pillow on a sofa usually reads as `in(pillow, sofa)` (the backrest makes the sofa box taller than the
  pillow's bottom-to-seat gap). The rule is geometric, not semantic, by design.
- Dense clutter (books on a shelf) produces many `next_to` edges; they are scored like any other.
- AABBs of rotated objects are loose; OBB-based rules are a later option, not used here.

## v2 (2026-10-27) — wall-mounted classes

Added after task 001 found wall-mounted items mislabelled by the AABB-only rules ("wall-plug in table",
"picture on sofa", "pipe on wall-plug"). Source: `DEVIATIONS.md` row 5;
finding F1 of the Replica harness build. The v1 text above is kept unchanged as history; where it
conflicts with this section, v2 governs.

- A frozen set `WALL_MOUNTED` (in `spatialmem/gt/relations.py`). An object whose label is in `WALL_MOUNTED`
  is **neither subject nor support** of `on` or `in`: a pair (A, B) cannot qualify for `on` / `in` if
  either label is in the set.
- Because such a pair no longer qualifies `on` / `in`, it is no longer blocked from `next_to` by the
  "on/in-qualifying pair" clause of rule 3. The `next_to` rule itself is unchanged.
- Replica members: `wall-plug`, `switch`, `picture`, `panel`, `pipe`, `vent`.
- HSSD members (2026-10-27b, frozen before any HSSD relation number; chosen from the label vocabulary of the 35
  frozen scenes, `scripts/hssd_vocab.py`): `picture`, `panel`, `vent` (shared with Replica), `mirror`, `wall_lamp`,
  `wall_clock`, `light_switch`, `socket`, `electric_outlet`, `wall_panel`, `curtain`, `curtain_rod`, `window_shutter`,
  `towel_ring`, `towel_rack`, `wall_hook_rack`, `wall_board`, `photo_mount`, `showerhead`. Not members (as for Replica):
  windows, doors, blinds; ceiling-mounted items; `tap` (on a sink is correct); `picture_frame` and `clock` (mostly
  tabletop in HSSD). Adding the HSSD names changes no Replica relation (no Replica class among them beyond the three
  shared ones).
- HSSD up axis = +Y (Habitat world); `compute_relations(..., up=1)`.
