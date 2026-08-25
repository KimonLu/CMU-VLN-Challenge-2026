import numpy as np
import pytest

from smart_vlm.projection import PanoProjector
from smart_vlm.semantic_map import MapObject, SemanticMap, _norm_label

IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
SMAP_CFG = {'merge_dist_m': 0.5, 'confirm_obs': 2, 'grid_res_m': 0.2,
            'terrain_obstacle_intensity': 0.15, 'obstacle_z_range': [0.1, 1.5],
            'min_lidar_pts': 8}


@pytest.fixture
def smap(proj_cfg, logger):
    return SemanticMap(dict(SMAP_CFG), PanoProjector(proj_cfg), logger)


def cluster(center, n=40, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.normal(0, spread, (n, 3))


def box_around(proj, pts, margin=8):
    pix, valid = proj.points_to_pixels(pts)
    pix = pix[valid]
    return (pix[:, 0].min() - margin, pix[:, 1].min() - margin,
            pix[:, 0].max() + margin, pix[:, 1].max() + margin)


PANO = np.full((640, 1920, 3), 120, dtype=np.uint8)


def test_integrate_keeps_foreground_cluster(smap):

    fg = cluster([2.0, 0.0, 0.5])
    bg = cluster([6.0, 0.0, 0.5], n=15, seed=1)
    pts = np.vstack([fg, bg])
    box = box_around(smap.proj, fg)
    smap.integrate(PANO, [{'label': 'chair', 'conf': 0.5, 'box': box}],
                   pts, IDENTITY)
    assert len(smap.objects) == 1
    assert abs(smap.objects[0].center[0] - 2.0) < 0.5


def test_merge_and_confirm(smap):
    pts = cluster([2.0, 0.0, 0.5])
    box = box_around(smap.proj, pts)
    det = {'label': 'chair', 'conf': 0.5, 'box': box}
    smap.integrate(PANO, [det], pts, IDENTITY)
    assert smap.confirmed() == []
    smap.integrate(PANO, [det], pts, IDENTITY)
    assert len(smap.objects) == 1
    assert smap.objects[0].n_obs == 2
    assert len(smap.confirmed()) == 1


def test_min_points_thin_class_relaxed(smap):
    pts = cluster([2.0, 0.0, 0.5], n=5)
    box = box_around(smap.proj, pts)
    smap.integrate(PANO, [{'label': 'chair', 'conf': 0.5, 'box': box}],
                   pts, IDENTITY)
    assert len(smap.objects) == 0
    smap.integrate(PANO, [{'label': 'window', 'conf': 0.5, 'box': box}],
                   pts, IDENTITY)
    assert len(smap.objects) == 1


def test_approximate_grounding_keeps_sparse_reflection(smap):

    smap.cfg['approx_min_lidar_pts'] = 1
    smap.cfg['approx_default_size_m'] = 0.3
    pts = np.asarray([[2.0, 0.0, 0.5]])
    box = box_around(smap.proj, pts)
    smap.integrate(PANO, [{'label': 'bowl', 'conf': 0.5, 'box': box}],
                   pts, IDENTITY)
    assert len(smap.objects) == 1
    assert np.allclose(smap.objects[0].center, pts[0])
    assert np.allclose(smap.objects[0].size, [0.3, 0.3, 0.3])


def test_integrate_wrapped_box_at_seam(smap):

    fg = cluster([-2.0, 0.0, 0.5])
    pix, valid = smap.proj.points_to_pixels(fg)
    pix = pix[valid]

    uc = 0.0
    du = (pix[:, 0] - uc + 960) % 1920 - 960
    box = (uc + du.min() - 8, pix[:, 1].min() - 8,
           uc + du.max() + 8, pix[:, 1].max() + 8)
    assert box[0] < 0 < box[2]
    smap.integrate(PANO, [{'label': 'door', 'conf': 0.5, 'box': box}],
                   fg, IDENTITY)
    assert len(smap.objects) == 1
    assert abs(smap.objects[0].center[0] + 2.0) < 0.5


def test_best_crop_padded_with_context(smap):


    pts = cluster([2.0, 0.0, 0.5])
    box = box_around(smap.proj, pts)
    smap.integrate(PANO, [{'label': 'chair', 'conf': 0.5, 'box': box}],
                   pts, IDENTITY)
    crop = smap.objects[0].best_crop
    assert crop.shape[0] >= 1.4 * (box[3] - box[1])
    assert crop.shape[1] >= 1.4 * (box[2] - box[0])


def mk(oid, label, center=(0, 0, 0), size=(1, 1, 1), n_obs=2, color='gray'):
    return MapObject(oid, label, color, np.asarray(center, float),
                     np.asarray(size, float), n_obs=n_obs)


def test_norm_label():
    assert _norm_label('Sofas') == 'sofa'
    assert _norm_label('couch') == 'sofa'
    assert _norm_label('paintings') == 'picture'
    assert _norm_label('potted plants') == 'plant'
    assert _norm_label('glass') == 'glass'


def test_by_label_matching(smap):
    smap.objects = [mk(0, 'coffee table'), mk(1, 'bookshelf'), mk(2, 'book'),
                    mk(3, 'painting'), mk(4, 'sofa'), mk(5, 'wall lamp')]
    ids = lambda q: {o.oid for o in smap.by_label(q)}
    assert ids('table') == {0}
    assert ids('book') == {2}
    assert ids('picture') == {3}
    assert ids('sofas') == {4}
    assert ids('couch') == {4}
    assert ids('wall lamp') == {5}
    assert ids('lamp') == {5}
    assert ids('') == set()


def test_scene_text_no_side_effect(smap):
    smap.objects = [mk(0, 'sofa'), mk(1, 'chair'), mk(2, 'table')]
    before = [o.oid for o in smap.objects]
    txt = smap.scene_text(relevant_words=['table'])
    assert '[2]' in txt.splitlines()[1]
    assert [o.oid for o in smap.objects] == before


def test_best_crop_is_deep_copy(smap):


    fg = cluster([2.0, 0.0, 0.5])
    box = box_around(smap.proj, fg)
    smap.integrate(PANO.copy(), [{'label': 'chair', 'conf': 0.5, 'box': box}],
                   fg, IDENTITY)
    crop = smap.objects[0].best_crop
    assert crop is not None and crop.size > 0
    assert crop.base is None


def test_saved_view_is_bounded_and_decodable(smap):
    smap.cfg['max_saved_views'] = 2
    for i in range(4):
        pano = np.full_like(PANO, 20 + i * 20)
        smap.integrate(pano, [], np.zeros((0, 3)), IDENTITY)
    assert len(smap._viewpoints) == 2
    view = smap.best_view('chair')
    assert view is not None and view.shape == PANO.shape

    assert abs(float(view.mean()) - 80.0) < 2.0
