import sys
import types
from types import MethodType

import numpy as np


def _install_ros_stubs():

    if 'rclpy' in sys.modules:
        return
    try:
        __import__('rclpy')
        return
    except ModuleNotFoundError:
        pass

    def module(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    dummy = type('Dummy', (), {})
    module('rclpy', ok=lambda: True)
    module('rclpy.node', Node=dummy)
    module('rclpy.executors', MultiThreadedExecutor=dummy)
    module('rclpy.callback_groups', ReentrantCallbackGroup=dummy)
    for pkg, names in {
            'std_msgs.msg': ('String', 'Int32'),
            'nav_msgs.msg': ('Odometry',),
            'sensor_msgs.msg': ('Image', 'PointCloud2'),
            'geometry_msgs.msg': ('Pose2D',),
            'visualization_msgs.msg': ('Marker',)}.items():
        module(pkg, **{name: dummy for name in names})


_install_ros_stubs()
import smart_vlm.main_node as main_node  # noqa: E402
from smart_vlm.main_node import SmartVLM  # noqa: E402


class Logger:
    def info(self, *args, **kwargs):
        pass

    warn = info


class Grid:
    def __init__(self, path):
        self.path = path
        self.calls = []

    def astar(self, start, goal, penalty_zones=None):
        self.calls.append((tuple(start), tuple(goal), penalty_zones))
        return self.path

    def recovery_approach(self, start, goal, standoff=1.0, penalty_zones=None):
        return None, None


def harness(path=()):
    h = types.SimpleNamespace()
    h.cfg = {
        'exploration': {'final_approach_radius_m': 3.0,
                        'final_waypoint_step_m': 1.0,
                        'final_reach_dist_m': 0.5,
                        'final_replan_attempts': 1,
                        'reach_dist_m': 1.0,
                        'stuck_timeout_s': 10.0,
                        'stuck_min_move_m': 0.3,
                        'goal_progress_enabled': True,
                        'progress_timeout_s': 10.0,
                        'progress_min_delta_m': 0.15,
                        'failure_memory_enabled': True,
                        'failure_memory_replan_attempts': 2,
                        'approach_replan_attempts': 2},
        'timing': {'waypoint_timeout_s': 25,
                   'final_approach_timeout_s': 70}}
    h.pose = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    h.gm = Grid(path)
    h.log = Logger()
    h.get_logger = lambda: h.log
    h._follow_final_path = MethodType(SmartVLM._follow_final_path, h)
    h._failure_penalty = MethodType(SmartVLM._failure_penalty, h)
    h._remember_failure = MethodType(SmartVLM._remember_failure, h)
    h.calls = []
    return h


def test_goto_lateral_drift_is_no_goal_progress(monkeypatch):

    h = harness()
    h.pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    h.pub_wp = types.SimpleNamespace(publish=lambda msg: None)
    clock = [100.0]

    def sleep(dt):
        clock[0] += 1.0


        y = 0.05 * (clock[0] - 100.0)
        h.pose = (0.0, y, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(main_node.time, 'time', lambda: clock[0])
    monkeypatch.setattr(main_node.time, 'sleep', sleep)
    assert not SmartVLM.goto(
        h, (5.0, 0.0), timeout=30.0, stuck_check=True, reach=0.5)
    assert h._last_goto_result['status'] == 'no_goal_progress'
    assert h._last_goto_result['moved'] > 0.3
    assert h._last_goto_result['elapsed'] < 30.0


def test_goto_goal_progress_resets_progress_window(monkeypatch):

    h = harness()
    h.pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    h.pub_wp = types.SimpleNamespace(publish=lambda msg: None)
    clock = [100.0]

    def sleep(dt):
        clock[0] += 1.0
        x = min(5.0, 0.10 * (clock[0] - 100.0))
        h.pose = (x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(main_node.time, 'time', lambda: clock[0])
    monkeypatch.setattr(main_node.time, 'sleep', sleep)
    assert SmartVLM.goto(
        h, (2.0, 0.0), timeout=30.0, stuck_check=True, reach=0.5)
    assert h._last_goto_result['status'] == 'reached'


def test_follow_waypoints_final_goal_not_unconditionally_repeated():
    h = harness()

    def goto(xy, timeout=25, stuck_check=False, reach=None):
        h.calls.append((tuple(xy), timeout, stuck_check, reach))
        h.pose = (xy[0], xy[1], 0.0, 0.0, 0.0, 0.0, 1.0)
        return True

    h.goto = goto
    ok = SmartVLM.follow_waypoints(
        h, [(0.0, 0.0), (3.0, 0.0), (5.0, 0.0)],
        main_node.time.time() + 100, final_start_idx=1)
    assert ok
    assert h.calls[0][2:] == (False, None)
    assert all(c[2:] == (True, 0.5) for c in h.calls[1:])
    assert sum(c[0] == (5.0, 0.0) for c in h.calls) == 1
    assert h.gm.calls == []


def test_follow_waypoints_respects_planner_final_boundary():
    h = harness()
    first = True

    def goto(xy, timeout=25, stuck_check=False, reach=None):
        nonlocal first
        h.calls.append((tuple(xy), stuck_check, reach))
        if first:
            first = False
            return False
        h.pose = (xy[0], xy[1], 0.0, 0.0, 0.0, 0.0, 1.0)
        return True

    h.goto = goto
    wps = [(0.5, 0.0), (8.0, 0.0), (3.0, 0.0),
           (2.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    SmartVLM.follow_waypoints(
        h, wps, main_node.time.time() + 100, final_start_idx=2)
    assert h.gm.calls == []
    assert h.calls[0][1:] == (False, None)
    assert all(c[1:] == (True, 0.5) for c in h.calls[2:])


def test_follow_waypoints_final_failure_replans_once_with_penalties():
    retry_path = [(2.5, 0.0), (3.5, 0.0), (4.5, 0.0), (5.0, 0.0)]
    h = harness(retry_path)
    failed = False

    def goto(xy, timeout=25, stuck_check=False, reach=None):
        nonlocal failed
        h.calls.append((tuple(xy), timeout, stuck_check, reach))
        if tuple(xy) == (3.0, 0.0) and not failed:
            failed = True
            h.pose = (2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            return False
        h.pose = (xy[0], xy[1], 0.0, 0.0, 0.0, 0.0, 1.0)
        return True

    h.goto = goto
    zones = [((1.0, 1.0), (2.0, 2.0), 1.2, 50.0)]
    ok = SmartVLM.follow_waypoints(
        h, [(0.0, 0.0), (3.0, 0.0), (5.0, 0.0)],
        main_node.time.time() + 100, penalty_zones=zones,
        final_start_idx=1)
    assert ok
    assert len(h.gm.calls) == 1
    start, goal, got_zones = h.gm.calls[0]
    assert start == (2.5, 0.0) and goal == (5.0, 0.0)
    assert got_zones is not zones
    assert got_zones[:1] == zones
    assert len(got_zones) == 2
    assert all(c[2:] == (True, 0.5) for c in h.calls[2:])


def test_final_failure_adds_episode_local_corridor_penalty_before_replan():

    h = harness([(2.5, 1.0), (5.0, 0.0)])
    failed = False

    def goto(xy, timeout=25, stuck_check=False, reach=None):
        nonlocal failed
        h.calls.append(tuple(xy))
        if not failed:
            failed = True

            h.pose = (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            h._last_goto_result = {'status': 'no_goal_progress'}
            return False
        h.pose = (xy[0], xy[1], 0.0, 0.0, 0.0, 0.0, 1.0)
        return True

    h.goto = goto
    original = [((8.0, 8.0), (9.0, 9.0), 1.2, 50.0)]
    assert SmartVLM.follow_waypoints(
        h, [(3.0, 0.0), (5.0, 0.0)],
        main_node.time.time() + 100, penalty_zones=original,
        final_start_idx=0)

    _, _, replanning_zones = h.gm.calls[0]
    assert replanning_zones is not original
    assert original == [((8.0, 8.0), (9.0, 9.0), 1.2, 50.0)]
    assert replanning_zones[:1] == original
    assert len(replanning_zones) == 2
    start, end, width, cost = replanning_zones[-1]
    assert np.allclose(start, (2.0, 0.0))
    assert np.allclose(end, (2.8, 0.0))
    assert width > 0 and cost > 0


def test_follow_waypoints_replan_no_path_directs_goal_once():
    h = harness(None)
    failed = False

    def goto(xy, timeout=25, stuck_check=False, reach=None):
        nonlocal failed
        h.calls.append(tuple(xy))
        if not failed:
            failed = True
            h.pose = (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            return False
        h.pose = (xy[0], xy[1], 0.0, 0.0, 0.0, 0.0, 1.0)
        return True

    h.goto = goto
    assert SmartVLM.follow_waypoints(
        h, [(5.0, 0.0)], main_node.time.time() + 100,
        final_start_idx=0)
    assert len(h.gm.calls) == 1
    assert h.calls == [(5.0, 0.0), (5.0, 0.0)]


def test_final_failure_switches_to_alternate_standoff():

    h = harness()
    h.gm.astar = lambda start, goal, penalty_zones=None: [tuple(start), tuple(goal)]
    failed = False

    def goto(xy, timeout=25, stuck_check=False, reach=None):
        nonlocal failed
        h.calls.append(tuple(xy))
        if not failed:
            failed = True
            h.pose = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            return False
        h.pose = (xy[0], xy[1], 0.0, 0.0, 0.0, 0.0, 1.0)
        return True

    h.goto = goto
    assert SmartVLM.follow_waypoints(
        h, [(3.0, 0.0)], main_node.time.time() + 100,
        final_start_idx=0, final_alternatives=[(4.0, 1.0), (4.0, -1.0)])
    assert h.calls[0] == (3.0, 0.0)
    assert h.calls[-1] == (4.0, 1.0)


def test_follow_waypoints_final_timeout_is_bounded_by_hard_abort(monkeypatch):
    h = harness()
    h.pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    timeouts = []

    def goto(xy, timeout=25, stuck_check=False, reach=None):
        timeouts.append(timeout)
        h.pose = (xy[0], xy[1], 0.0, 0.0, 0.0, 0.0, 1.0)
        return True

    h.goto = goto
    monkeypatch.setattr(main_node.time, 'time', lambda: 100.0)
    assert SmartVLM.follow_waypoints(
        h, [(2.0, 0.0)], t_abort=108.0, final_start_idx=0)
    assert timeouts == [8.0]


def test_follow_waypoints_final_points_share_one_budget(monkeypatch):
    h = harness()
    clock = [100.0]
    timeouts = []

    def goto(xy, timeout=25, stuck_check=False, reach=None):
        timeouts.append(timeout)
        clock[0] += 20.0
        h.pose = (xy[0], xy[1], 0.0, 0.0, 0.0, 0.0, 1.0)
        return True

    h.goto = goto
    monkeypatch.setattr(main_node.time, 'time', lambda: clock[0])
    SmartVLM.follow_waypoints(
        h, [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        t_abort=200.0, final_start_idx=0)
    assert timeouts == [25, 25, 25, 10.0]


def test_relation_final_target_never_early_stops():

    h = types.SimpleNamespace()
    h.answerer = types.SimpleNamespace(can_ground=lambda *args: True)
    h.targets_found = lambda parsed: True
    rel = {'constraints': [{'action': 'stop_at', 'target': 'cabinet',
                           'anchors': ['picture'], 'relation': 'above'}]}
    plain = {'constraints': [{'action': 'stop_at', 'target': 'chair',
                             'anchors': [], 'relation': 'none'}]}
    assert SmartVLM._early_stop_ok(h, rel, 'instruction_following') is False
    assert SmartVLM._early_stop_ok(h, plain, 'instruction_following') is True
