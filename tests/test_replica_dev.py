"""Task 006 Replica-dev: scene list, per-vertex GT derivation, trajectory sampling, frame conversion."""
import math

import numpy as np

import spatialmem.gt.replica as R
from spatialmem.render.replica_dev import R_H2M, R_M2H, sample


def test_dev_scenes_disjoint_from_protocol():
    assert len(R.DEV_SCENES) == len(set(R.DEV_SCENES)) > 0
    assert not set(R.DEV_SCENES) & {R.v1_name(s) for s in R.SCENES}
    assert R.v1_name("room0") == "room_0" and R.v1_name("frl_apartment_3") == "frl_apartment_3"


def test_vertex_classes_majority_tie_and_orphan():
    # vertex 0: faces of classes 40, 40, 80 -> 40; vertex 1: 40 / 80 tie -> 40 (smallest); vertex 5: no face -> -1
    faces = np.array([[0, 1, 2, 3], [0, 2, 3, 4], [0, 1, 4, 2]])
    face_obj = np.array([10, 10, 20])
    obj_cls = {10: 40, 20: 80}
    lab, n_conflict = R.vertex_classes(6, faces, face_obj, obj_cls)
    assert lab.tolist() == [40, 40, 40, 40, 40, -1]
    assert n_conflict == 4          # vertices 0, 1, 2, 4 touch both classes
    lab, _ = R.vertex_classes(6, faces, np.array([10, 99, 99]), obj_cls)   # object 99 has no class -> -1
    assert lab[0] == -1 and lab[3] == -1      # majority -1; tie 40 / -1 -> smallest


def test_sample_rotation_cap_and_jumps():
    keys = [(np.zeros(3), 0.0, False), (np.array([1.0, 0, 0]), 0.0, False),
            (np.array([1.0, 0, 0]), 1.0, False),                       # 1 rad turn in place
            (np.array([9.0, 0, 0]), 1.0, True)]                        # island jump: no frames
    cap = math.radians(2.0)
    pos, yaw, st = sample(keys, 101, cap)
    assert st["n_jumps"] == 1 and st["path_m"] == 1.0 and len(pos) == 101
    assert np.abs(np.diff(yaw)).max() <= cap + 1e-9
    assert np.isclose(st["rot_frames"], 1.0 / cap)
    assert pos[:, 0].max() <= 1.0 + 1e-9                                # the jump target is never sampled mid-way
    try:
        sample(keys, 20, cap)                                          # 1 rad needs 29 frames at 2 deg
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_frame_conversion_is_z_up_to_y_up_rotation():
    assert np.allclose(R_M2H @ R_H2M, np.eye(4)) and math.isclose(np.linalg.det(R_M2H[:3, :3]), 1.0)
    assert np.allclose(R_M2H[:3, :3] @ [0, 0, 1], [0, 1, 0])      # mesh up -> Habitat up
