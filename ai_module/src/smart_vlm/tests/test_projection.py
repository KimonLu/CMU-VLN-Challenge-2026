import numpy as np
import pytest

from smart_vlm.projection import PanoProjector, transform_points


def make_proj(cfg, az=False, el=False):
    c = dict(cfg)
    c['azimuth_flip'], c['elevation_flip'] = az, el
    return PanoProjector(c)


@pytest.mark.parametrize('az,el', [(False, False), (True, False),
                                   (False, True), (True, True)])
def test_roundtrip_points_to_pixels_to_ray(proj_cfg, az, el):

    proj = make_proj(proj_cfg, az, el)
    pts = np.array([[2.0, 0.5, 0.3], [-1.0, 2.0, -0.5], [0.5, -3.0, 1.0]])
    pix, valid = proj.points_to_pixels(pts)
    assert valid.all()
    for p, (u, v) in zip(pts, pix):
        ray = proj.pixel_to_ray(u, v)
        d = p / np.linalg.norm(p)
        assert np.dot(ray, d) > 0.999, f'ray mismatch: {ray} vs {d}'


def test_forward_point_center(proj_cfg):

    proj = PanoProjector(proj_cfg)
    pix, valid = proj.points_to_pixels(np.array([[3.0, 0.0, 0.0]]))
    assert valid[0]
    assert abs(pix[0, 0] - 960) < 1.0
    assert abs(pix[0, 1] - 320) < 1.0


def test_make_views_patch_position(proj_cfg):

    proj = PanoProjector(proj_cfg)
    pano = np.zeros((640, 1920, 3), dtype=np.uint8)
    pano[300:340, 940:980] = 255
    views = proj.make_views(pano)
    assert len(views) == 4
    v0, yaw0 = views[0]
    assert yaw0 == 0
    assert v0[310:330, 310:330].max() > 200
    v180 = views[2][0]
    assert v180[310:330, 310:330].max() == 0


def test_view_box_width_angular_scaling(proj_cfg):

    proj = PanoProjector(proj_cfg)
    size, hfov = 640, 105.0
    f = (size / 2) / np.tan(np.deg2rad(hfov) / 2)
    w_view = 100.0
    box = (size / 2 - w_view / 2, size / 2 - 20, size / 2 + w_view / 2, size / 2 + 20)
    u1, v1, u2, v2 = proj.view_box_to_pano_box(box, 0.0)
    ang_w = 2 * np.arctan((w_view / 2) / f)
    expect_w = ang_w / (2 * np.pi) * proj.W
    assert abs((u2 - u1) - expect_w) < 2.0

    assert abs((u2 - u1) - w_view) > 5.0


def test_view_box_center_matches_view_box_to_pano(proj_cfg):
    proj = PanoProjector(proj_cfg)
    box = (100, 200, 300, 400)
    u, v = proj.view_box_to_pano(box, np.pi / 2)
    u1, v1, u2, v2 = proj.view_box_to_pano_box(box, np.pi / 2)
    assert u1 <= u <= u2 and v1 <= v <= v2


def test_view_box_wraps_seam(proj_cfg):

    proj = PanoProjector(proj_cfg)
    box = (270, 300, 370, 340)
    u1, v1, u2, v2 = proj.view_box_to_pano_box(box, np.pi)
    assert (u2 - u1) < 200
    assert u1 < u2

    uc = (u1 + u2) / 2
    assert min(abs(uc), abs(uc - proj.W)) < 60


def test_transform_points_identity_and_shift():
    pts = np.array([[1.0, 2.0, 3.0]])
    out = transform_points(pts, (0, 0, 0, 0, 0, 0, 1))
    assert np.allclose(out, pts)
    out = transform_points(pts, (1.0, 0, 0, 0, 0, 0, 1))
    assert np.allclose(out, [[0.0, 2.0, 3.0]])

    s, c = np.sin(np.pi / 4), np.cos(np.pi / 4)
    out = transform_points(np.array([[0.0, 1.0, 0.0]]), (0, 0, 0, 0, 0, s, c))
    assert np.allclose(out, [[1.0, 0.0, 0.0]], atol=1e-9)
