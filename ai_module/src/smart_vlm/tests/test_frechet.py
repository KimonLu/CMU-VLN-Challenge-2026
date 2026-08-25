import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'tools')))
from frechet import endpoint_distance, frechet_distance, resample  # noqa: E402
from score_regression import aabb_iou, load_gt_trajectory  # noqa: E402


def test_identical_zero():
    t = np.array([[0, 0], [1, 0], [2, 0], [3, 1]])
    assert frechet_distance(t, t) < 1e-9
    assert endpoint_distance(t, t) < 1e-9


def test_parallel_offset():
    t1 = np.array([[x, 0.0] for x in np.linspace(0, 5, 20)])
    t2 = t1 + [0.0, 0.7]
    assert abs(frechet_distance(t1, t2) - 0.7) < 1e-6


def test_reversed_direction_penalized():
    t1 = np.array([[x, 0.0] for x in np.linspace(0, 5, 20)])
    assert frechet_distance(t1, t1[::-1]) >= 5.0 - 1e-6


def test_resample_caps_length():
    t = np.zeros((5000, 2))
    assert len(resample(t, 400)) == 400
    assert len(resample(t[:100], 400)) == 100


def test_empty_inf():
    assert frechet_distance(np.zeros((0, 2)), np.zeros((3, 2))) == float('inf')


def test_extra_columns_ignored():
    t3 = np.array([[0, 0, 9], [1, 0, 9], [2, 0, 9]], dtype=float)
    assert frechet_distance(t3, t3[:, :2]) < 1e-9


def test_aabb_iou_identical_and_disjoint():
    assert abs(aabb_iou([0, 0, 0], [2, 2, 2],
                        [0, 0, 0], [2, 2, 2]) - 1.0) < 1e-9
    assert aabb_iou([0, 0, 0], [1, 1, 1],
                    [2, 0, 0], [1, 1, 1]) == 0.0


def test_aabb_iou_partial_overlap():
    # Two 2x2x2 cubes offset by one metre along x: intersection 4, union 12.
    assert abs(aabb_iou([0, 0, 0], [2, 2, 2],
                        [1, 0, 0], [2, 2, 2]) - 1 / 3) < 1e-9


def test_aabb_iou_rejects_invalid_size():
    assert aabb_iou([0, 0, 0], [0, 1, 1],
                    [0, 0, 0], [1, 1, 1]) == 0.0


def test_load_ascii_ply_without_plyfile(tmp_path):
    scene = tmp_path / 'scene'
    scene.mkdir()
    (scene / 'trajectory_q4.ply').write_text(
        'ply\nformat ascii 1.0\nelement vertex 2\n'
        'property float x\nproperty float y\nproperty float z\nend_header\n'
        '1 2 9\n3 4 8\n', encoding='ascii')
    got = load_gt_trajectory(str(tmp_path), 'scene', 4)
    np.testing.assert_allclose(got, [[1, 2], [3, 4]])
