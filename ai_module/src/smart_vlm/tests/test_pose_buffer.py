import math

from smart_vlm.pose_buffer import (PoseBuffer, TimedValueBuffer, ang_diff, keyframe_due,
                                   yaw_from_quat)

P = lambda x: (x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def test_query_nearest():
    buf = PoseBuffer()
    for t in (1.0, 1.1, 1.2):
        buf.push(t, P(t))
    assert buf.query(1.09)[0] == 1.1
    assert buf.query(1.14)[0] == 1.1
    assert buf.query(1.16)[0] == 1.2


def test_query_gap_returns_none():
    buf = PoseBuffer(max_gap_s=0.15)
    buf.push(1.0, P(1.0))
    assert buf.query(1.3) is None
    assert buf.query(0.7) is None
    assert buf.query(1.1) is not None


def test_empty_and_latest():
    buf = PoseBuffer()
    assert buf.query(1.0) is None
    assert buf.latest is None
    buf.push(1.0, P(1.0))
    buf.push(2.0, P(2.0))
    assert buf.latest[0] == 2.0


def test_out_of_order_dropped():
    buf = PoseBuffer()
    buf.push(2.0, P(2.0))
    buf.push(1.0, P(1.0))
    assert buf.query(1.0) == P(2.0) or buf.query(1.0) is None
    assert len(buf._t) == 1


def test_yaw_from_quat():
    s, c = math.sin(math.pi / 4), math.cos(math.pi / 4)
    assert abs(yaw_from_quat(0, 0, s, c) - math.pi / 2) < 1e-6
    assert abs(yaw_from_quat(0, 0, 0, 1)) < 1e-6


def test_ang_diff_wraps():
    assert abs(ang_diff(math.pi - 0.1, -math.pi + 0.1) - 0.2) < 1e-6
    assert abs(ang_diff(0.0, 2 * math.pi)) < 1e-6


def _pose(x, y, yaw):
    return (x, y, 0.0, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def test_keyframe_first_always_due():
    due, xy, yaw = keyframe_due(_pose(0, 0, 0), None, None, 0.5, 30)
    assert due and xy == (0, 0)


def test_keyframe_translation():
    _, xy, yaw = keyframe_due(_pose(0, 0, 0), None, None, 0.5, 30)
    due, _, _ = keyframe_due(_pose(0.4, 0, 0), xy, yaw, 0.5, 30)
    assert not due
    due, _, _ = keyframe_due(_pose(0.6, 0, 0), xy, yaw, 0.5, 30)
    assert due


def test_keyframe_rotation_only():

    _, xy, yaw = keyframe_due(_pose(0, 0, 0), None, None, 0.5, 30)
    due, _, _ = keyframe_due(_pose(0, 0, math.radians(20)), xy, yaw, 0.5, 30)
    assert not due
    due, _, _ = keyframe_due(_pose(0, 0, math.radians(40)), xy, yaw, 0.5, 30)
    assert due


def test_pose_buffer_covers_delayed_panorama():

    b = PoseBuffer()
    for i in range(1800):
        b.push(i * 0.005, (i,))
    assert b.query(1.5) == (300,)


def test_timed_value_buffer_nearest_and_gap():
    b = TimedValueBuffer(maxlen=4, max_gap_s=0.3)
    b.push(1.0, 'a'); b.push(1.2, 'b'); b.push(1.4, 'c')
    assert b.query(1.26) == 'b'
    assert b.query(2.0) is None
