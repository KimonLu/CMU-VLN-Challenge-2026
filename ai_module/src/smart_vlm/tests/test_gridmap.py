import numpy as np
import yaml
from pathlib import Path

from smart_vlm.exploration import (FREE, OBSTACLE, UNKNOWN, Explorer, GridMap,
                                   bootstrap_waypoints, decimate,
                                   decimate_final_approach, line_waypoints)


def test_far_point_growth_no_wraparound():

    gm = GridMap(res=0.2)
    gm.add_obstacles(np.array([[150.0, 150.0]]))
    assert (gm.grid == OBSTACLE).sum() == 1
    ij = np.argwhere(gm.grid == OBSTACLE)[0]
    assert np.allclose(gm.world(ij), [150.0, 150.0], atol=0.3)


def test_far_negative_point_growth():
    gm = GridMap(res=0.2)
    gm.add_obstacles(np.array([[-100.0, -55.0]]))
    ij = np.argwhere(gm.grid == OBSTACLE)[0]
    assert np.allclose(gm.world(ij), [-100.0, -55.0], atol=0.3)


def test_growth_preserves_content():
    gm = GridMap(res=0.2)
    gm.add_obstacles(np.array([[1.0, 1.0]]))
    gm.add_obstacles(np.array([[80.0, -3.0]]))
    ws = [gm.world(ij) for ij in np.argwhere(gm.grid == OBSTACLE)]
    assert any(np.allclose(w, [1.0, 1.0], atol=0.3) for w in ws)
    assert any(np.allclose(w, [80.0, -3.0], atol=0.3) for w in ws)


def test_terrain_obstacle_not_overwritten_by_free():
    gm = GridMap(res=0.2)
    pt = np.array([[2.0, 2.0, 0.0]])
    gm.update_terrain(pt, np.array([0.5]), obs_thresh=0.15)
    gm.update_terrain(pt, np.array([0.01]), obs_thresh=0.15)
    i, j = gm._idx([[2.0, 2.0]])[0]
    assert gm.grid[i, j] == OBSTACLE


def test_frontiers_found_and_filtered():
    gm = GridMap(res=0.2)
    gm.grid[50:60, 50:60] = FREE
    fs = gm.frontiers(min_cluster=5)
    assert len(fs) >= 1
    gm2 = GridMap(res=0.2)
    gm2.grid[:, :] = OBSTACLE
    gm2.grid[50:60, 50:60] = FREE
    assert gm2.frontiers() == []


def test_astar_goes_through_gap():
    gm = GridMap(res=0.2)
    gm.grid[:, :] = FREE
    gm.grid[90:110, 100] = OBSTACLE
    gm.grid[99:102, 100] = FREE
    start = gm.world((100, 80))
    goal = gm.world((100, 120))
    path = gm.astar(start, goal)
    assert path is not None

    for p in path:
        i, j = gm._idx([p])[0]
        if j == 100:
            assert 95 <= i <= 106


def test_astar_penalty_corridor_detours():
    gm = GridMap(res=0.2)
    gm.grid[:, :] = FREE
    start, goal = gm.world((100, 80)), gm.world((100, 120))
    z = (tuple(gm.world((90, 100))), tuple(gm.world((110, 100))), 1.0, 50.0)
    path = gm.astar(start, goal, penalty_zones=[z])
    assert path is not None

    for p in path:
        i, j = gm._idx([p])[0]
        if j == 100:
            assert not (95 <= i <= 105)


def test_nearest_free():
    gm = GridMap(res=0.2)
    gm.grid[:, :] = OBSTACLE
    gm.grid[105:115, 95:105] = FREE
    p = gm.nearest_free(gm.world((100, 100)), max_r=3.0)
    i, j = gm._idx([p])[0]
    assert gm.grid[i, j] == FREE


def test_bootstrap_translation_is_disabled_for_360_camera():
    cfg = {'init_sweep_enabled': False, 'init_spin_radius_m': 1.5}
    assert bootstrap_waypoints((2.0, -1.0), cfg) == []


def test_bootstrap_translation_legacy_switch_is_explicit():
    cfg = {'init_sweep_enabled': True, 'init_spin_radius_m': 1.5}
    pts = bootstrap_waypoints((2.0, -1.0), cfg)
    assert len(pts) == 4
    assert np.allclose(pts[0], (3.5, -1.0))
    assert np.allclose(pts[-1], (2.0, -2.5))


def test_reachable_standoffs_are_free_path_ranked_and_clear():
    gm = GridMap(res=0.2)
    gm.grid[:, :] = FREE
    target = gm.world((100, 100))
    start = gm.world((100, 75))

    gm.grid[96:105, 95:98] = OBSTACLE
    ranked = gm.reachable_standoffs(
        start, target, min_r=0.8, max_r=2.0,
        preferred_r=1.2, min_clearance=0.35)
    assert ranked
    best = ranked[0]
    assert best['path']
    i, j = gm._idx([best['point']])[0]
    assert gm.grid[i, j] == FREE
    assert 0.7 <= np.linalg.norm(best['point'] - target) <= 2.1
    assert best['clearance'] >= 0.35
    assert all(a['score'] <= b['score'] for a, b in zip(ranked, ranked[1:]))


def test_reachable_standoffs_return_empty_when_target_is_enclosed():
    gm = GridMap(res=0.2)
    gm.grid[:, :] = FREE
    target = gm.world((100, 100))
    start = gm.world((100, 70))
    gm.grid[88:113, 88:113] = OBSTACLE
    assert gm.reachable_standoffs(start, target, min_r=0.8, max_r=2.0) == []


def test_decimate_step_and_endpoint():
    path = [(i * 0.5, 0.0) for i in range(11)]
    wps = decimate(path, step=2.0)
    assert wps[-1] == (5.0, 0.0)
    assert len(wps) == 3
    assert wps[0] == (2.0, 0.0)


def test_decimate_final_approach_densifies_only_last_radius():
    path = [(x, 0.0) for x in np.linspace(0.0, 10.0, 51)]
    wps = decimate_final_approach(
        path, step=2.5, final_step=1.0, final_radius=3.0)
    assert wps[-1] == (10.0, 0.0)
    gaps = [np.linalg.norm(np.asarray(b) - np.asarray(a))
            for a, b in zip(wps, wps[1:])]
    assert any(g > 1.5 for g in gaps)
    final = [p for p in wps if p[0] >= 7.0]
    assert all(np.linalg.norm(np.asarray(b) - np.asarray(a)) <= 1.01
               for a, b in zip(final, final[1:]))
    assert all(a != b for a, b in zip(wps, wps[1:]))


def test_decimate_final_approach_short_path_is_all_fine():
    path = [(x, 0.0) for x in np.linspace(0.0, 2.4, 13)]
    wps = decimate_final_approach(
        path, step=2.5, final_step=1.0, final_radius=3.0)
    assert wps[-1] == (2.4, 0.0)
    assert all(np.linalg.norm(np.asarray(b) - np.asarray(a)) <= 1.21
               for a, b in zip(wps, wps[1:]))


def test_decimate_final_approach_uses_remaining_arc_length():

    path = [(0.2, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
            (2.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    wps = decimate_final_approach(
        path, step=2.5, final_step=1.0, final_radius=3.0)
    assert wps[0] == (3.0, 0.0)
    assert wps[-1] == (0.0, 0.0)


def test_decimate_final_approach_degenerate_inputs():
    path = [(x, 0.0) for x in range(6)]
    assert decimate_final_approach([], 2.0, 1.0, 3.0) == []
    assert decimate_final_approach(
        [(0.0, 0.0)], 2.0, 1.0, 3.0) == [(0.0, 0.0)]
    assert decimate_final_approach(path, 2.0, 1.0, 0.0) == decimate(path, 2.0)
    assert decimate_final_approach(path, 2.0, 0.0, 3.0) == decimate(path, 2.0)


def test_line_waypoints_keeps_small_steps_and_endpoint():
    wps = line_waypoints((0.0, 0.0), (3.2, 0.0), step=1.0)
    assert wps[-1] == (3.2, 0.0)
    assert len(wps) == 4
    assert all(np.linalg.norm(np.asarray(b) - np.asarray(a)) <= 1.0
               for a, b in zip([(0.0, 0.0)] + wps, wps))


def test_final_approach_config_contract():
    cfg = yaml.safe_load(open(Path(__file__).parents[1] / 'config' / 'params.yaml'))
    ex, timing = cfg['exploration'], cfg['timing']
    assert ex['final_waypoint_step_m'] == 1.0
    assert ex['final_approach_radius_m'] == 3.0
    assert ex['final_reach_dist_m'] == 0.5
    assert ex['final_replan_attempts'] == 1
    assert ex['goal_progress_enabled'] is False
    assert ex['failure_memory_enabled'] is False
    assert ex['init_sweep_enabled'] is False
    assert ex['approach_min_radius_m'] == 0.8
    assert ex['approach_preferred_radius_m'] == 1.2
    assert ex['approach_max_radius_m'] == 2.0
    assert ex['approach_replan_attempts'] == 2
    assert ex['progress_timeout_s'] == 10
    assert ex['progress_min_delta_m'] == 0.15
    assert timing['final_approach_timeout_s'] == 70


def test_explorer_blacklist(logger):
    gm = GridMap(res=0.2)
    gm.grid[50:60, 50:60] = FREE
    ex = Explorer(gm, {}, logger)
    g1 = ex.next_goal(np.array([10.0, 10.0]))
    assert g1 is not None
    ex.give_up_current()
    g2 = ex.next_goal(np.array([10.0, 10.0]))
    if g2 is not None:
        assert np.linalg.norm(g2 - g1) > 1.0


def test_explorer_reached_frontier_is_retired(logger):
    gm = GridMap(res=0.2)
    gm.grid[50:60, 50:60] = FREE
    ex = Explorer(gm, {}, logger)
    g1 = ex.next_goal(np.array([10.0, 10.0]))
    assert g1 is not None

    ex.complete_current()

    assert ex.current_goal is None
    assert any(np.allclose(g1, old) for old in ex.blacklist)
    g2 = ex.next_goal(np.array([10.0, 10.0]))
    if g2 is not None:
        assert np.linalg.norm(g2 - g1) > 1.0


def test_garbage_far_point_dropped_no_memory_explosion():


    gm = GridMap(res=0.2)
    n0 = gm.grid.size
    gm.add_obstacles(np.array([[5000.0, 5000.0], [1.0, 1.0]]))
    assert gm.grid.size == n0
    ws = [gm.world(ij) for ij in np.argwhere(gm.grid == OBSTACLE)]
    assert any(np.allclose(w, [1.0, 1.0], atol=0.3) for w in ws)
    assert not any(np.allclose(w, [5000.0, 5000.0], atol=1.0) for w in ws)


def test_garbage_terrain_point_dropped():
    gm = GridMap(res=0.2)
    pts = np.array([[2.0, 2.0, 0.0], [-9000.0, 0.0, 0.0]])
    gm.update_terrain(pts, np.array([0.5, 0.5]), obs_thresh=0.15)
    assert gm.grid.size == int(40 / 0.2) ** 2
    i, j = gm._idx([[2.0, 2.0]])[0]
    assert gm.grid[i, j] == OBSTACLE
