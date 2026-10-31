# data/ — manifest (task 001)

Everything under `data/` except this file is git-ignored and never redistributed (Replica: research,
non-commercial licence — THIRD_PARTY.md). Fetched 2026-10-27 on lpc by `scripts/fetch_replica.sh all`
(byte-range resumable), extracted by `scripts/extract_replica.sh`.

| source | URL | size (bytes) | local path |
|---|---|---|---|
| NICE-SLAM Replica (8 scenes: RGB-D `results/`, `traj.txt`, `<scene>_mesh.ply`, `cam_params.json`) | https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip | 12442855671 | `data/_dl/Replica.zip` → `data/nice/Replica/` |
| Replica v1 (17 split parts; only `<scene>/habitat/` of the 8 NICE scenes extracted) | https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.part{aa..aq} | 16 × 2000000000 + 1859047808 | `data/_dl/` → `data/replica_v1/<room_0>/habitat/` |
| OVO protocol per-vertex GT (original Replica class ids per NICE mesh vertex) | https://raw.githubusercontent.com/tberriel/OVO/e0d16f7a459ea5451540c50ac264b00aca3aeb85/data/input/replica_semantic_gt/<scene>.txt | see sha256 | `data/ovo_replica_gt/<scene>.txt` |

Integrity: `unzip` checked every member CRC; the 17 v1 parts were gunzipped as one stream (gzip CRC
over the whole archive) without error. Extracted: NICE 12 GB (2000 frames × 8 scenes), v1 habitat subset 567 MB.

## sha256

```
dc18265e213f8281444b768eed76f627bde8c3b925654934f84f78a87990a548  Replica.zip
96b64143d21d1630125c33f355e3f0c4ec5aef98ad88295316d9ac4f79d61d9a  replica_v1_0.tar.gz.partaa
eb930c383565894d75220d2562a93619bbb57a2793bf2d81c846eff58338104f  replica_v1_0.tar.gz.partab
641461fb7887829e529c2e2d885bbc9390711b8c558ba656f5e5d8072748ee7c  replica_v1_0.tar.gz.partac
17ff7bac3962289a894624529b001fd747a5fb26fef8070cf52fac156d57b91b  replica_v1_0.tar.gz.partad
528c9dff1ec11756a3ef6d4a1e3d8fc507f763979fc342019c6a77acbcafc514  replica_v1_0.tar.gz.partae
758e42e6c8378f78c1c719dd22ea2624f0ef0081a1d7ba88ef0c30f87c2d5f1c  replica_v1_0.tar.gz.partaf
3402380cfd29e79a361380334bd3fb98b798984a934bc780f7c2f36414925eb4  replica_v1_0.tar.gz.partag
87be7ceb51dd82227c8e4e14887c34b2252d2b0e968059896a16027266107119  replica_v1_0.tar.gz.partah
9240664291c7f1b5d9796a00d47f4016324af3ec0cc5f94f3d1ec7b252af3a9b  replica_v1_0.tar.gz.partai
932d5024985a9a2942553ea422ebbf709f4d254e08d1d35902fb100d4ebcc6b2  replica_v1_0.tar.gz.partaj
a7d1aef7489dde4e7f5fa92d8db3022ce36060da520847e4bb8faf903202c66d  replica_v1_0.tar.gz.partak
5158d3f502c9c145bb34e32b65db9a4caf6176709988fb56ea1119f60e51e6d3  replica_v1_0.tar.gz.partal
6975a2e4c5774801de8a0df5b93bb553c47e70110c8e532bda1c59c21b6bac72  replica_v1_0.tar.gz.partam
0c78039eb0b254ba88f925db06c62163f78e10f5dbf3dbf655725b692560aba6  replica_v1_0.tar.gz.partan
c43de2b75d578dcb8b89d5669d0f23e85813d68f38d539860bb12580a40de633  replica_v1_0.tar.gz.partao
f37fb2fcf2df99b327cbeb27190f7c29c4693fe6e9ed66e27c254bd44da00eed  replica_v1_0.tar.gz.partap
c2d520b6ed67714ce1ccacfa80e4a834f567307817c3179f3b3bc58bf033e18b  replica_v1_0.tar.gz.partaq
a55f1bf13fbb2057f3ab1a7f92180e9b19e400104f55c47130ab20cd3f72c4d2  ovo_replica_gt/office0.txt
39ca76b1f59873af6ffacb6f8dc64e5ab19924c24b3d0c117ece9f0fb80126c0  ovo_replica_gt/office1.txt
0dab4c76896e5b31c31a2d4b81a5336366ac0b7e12b7bdff0cd026daebcaf4dc  ovo_replica_gt/office2.txt
0056c42dc114308862a391dc40d31ba0c9577fe729ee9c80bd05803b6ceabfe8  ovo_replica_gt/office3.txt
a058ecb4da5ba51af99dd43f6b37d5901f03f195ac0086372754e7bee90f4783  ovo_replica_gt/office4.txt
0408668499ab5e6afd426eba4bee98d1d571918aedcd3014cb43c8efe521b343  ovo_replica_gt/room0.txt
a8a3f69c7eeddfda8c8ec32ee3218fbd9b469bbbe90a56156a8791f65bc82556  ovo_replica_gt/room1.txt
d4c3c54b79ad14688984fcc91b5e7f27dcef390d5ea8b5611c3aec2b7a3114e8  ovo_replica_gt/room2.txt
```

## HSSD (task 002) — on PERUN only

`hssd/hssd-hab` (Hugging Face, CC BY-NC 4.0, public, `gated: false`) at revision
`4369cb9876214c7fbebcf552eb532380e4d287e4`, fetched anonymously 2026-10-27 on the PERUN login node by
`scripts/hssd_fetch.py meta` + `objects` into `$PROJ/data/hssd-hab/`
(never on lpc except the 5 dev streams' renders under `data/hssd_runs/`, never redistributed).

- Scenes: `configs/hssd_scenes.yaml` (5 dev + 30 eval, seed 114, dev/eval disjoint by floorplan id).
- Per-scene assets: 35 scene instances + their stages + every referenced object template (incl. decomposed parts and
  openings): **15,079 files, 3,319,084,329 bytes** — sha256 of each in `data/MANIFEST_hssd.sha256` (tracked;
  its own sha256 `fe187f7cc53e8a437dc6839818c0fb6d35144898a545be6d14d2a30499ecdc5e`).
- Shared metadata (not in that list): `hssd-hab.scene_dataset_config.json` 186cbbd5fe8b53ddd1cc0dfb3c9084e75db8ec6cd05ba178a5b43890f36b25e2,
  `semantics/objects.csv` 15aa3d66f10bc29cbb6814ca6b800791e7cdde07b98575397ed0c44fa285bb0b,
  `semantics/hssd-hab_semantic_lexicon.json` 60764fd3a50f816a7dfa4ac885fe357a8ce406c008b5d8ad5a22f64bb52cfc5d.
- Rendered streams (RGB jpg / depth png / instance png / poses, 58,197 frames, 9.9 GB) on PERUN
  `runs/hssd/<scene>/`; the 5 dev scenes pulled to lpc `data/hssd_runs/dev/` (565 MB).
