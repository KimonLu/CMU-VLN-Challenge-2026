"""Deterministic spatial-relation utilities for mapped objects."""

import numpy as np


def hdist(a, b):
    return float(np.linalg.norm(a.center[:2] - b.center[:2]))


def closest(cands, anchor):
    return min(cands, key=lambda o: hdist(o, anchor)) if cands else None


def farthest(cands, anchor):
    return max(cands, key=lambda o: hdist(o, anchor)) if cands else None


def near(a, b, slack=1.5):
    thr = max(slack, 0.5 * (max(a.size[:2]) + max(b.size[:2])))
    return hdist(a, b) < thr


def between(x, a, b, corridor=1.0):
    p, p1, p2 = x.center[:2], a.center[:2], b.center[:2]
    v = p2 - p1
    L = np.linalg.norm(v)
    if L < 1e-6:
        return False
    t = float(np.dot(p - p1, v) / (L * L))
    d = float(np.linalg.norm(p - (p1 + t * v)))
    return 0.15 < t < 0.85 and d < corridor


def on_top(a, b, z_tol=0.15, xy_slack=0.1):
    a_bottom = a.center[2] - a.size[2] / 2
    b_top = b.center[2] + b.size[2] / 2
    if abs(a_bottom - b_top) > z_tol:
        return False
    dx = abs(a.center[0] - b.center[0])
    dy = abs(a.center[1] - b.center[1])
    return dx < b.size[0] / 2 + xy_slack and dy < b.size[1] / 2 + xy_slack


def above(a, b, min_gap=0.2):
    dz = (a.center[2] - a.size[2] / 2) - (b.center[2] + b.size[2] / 2)
    dx = abs(a.center[0] - b.center[0])
    dy = abs(a.center[1] - b.center[1])
    horiz = dx < (a.size[0] + b.size[0]) / 2 and dy < (a.size[1] + b.size[1]) / 2
    return horiz and dz > -0.05 and a.center[2] > b.center[2] + min_gap


def below(a, b, min_gap=0.2):
    return above(b, a, min_gap)


def apply_relation(relation, cands, anchors):


    if not cands:
        return []
    if relation in ('none', None, ''):
        return cands
    if relation == 'closest' and anchors:
        return [closest(cands, anchors[0])]
    if relation == 'farthest' and anchors:
        return [farthest(cands, anchors[0])]
    if relation == 'near' and anchors:
        return [c for c in cands if any(near(c, a) for a in anchors)]
    if relation == 'between' and len(anchors) >= 2:
        r = [c for c in cands if between(c, anchors[0], anchors[1])]
        return r or cands
    if relation == 'on' and anchors:
        return [c for c in cands if any(on_top(c, a) for a in anchors)]
    if relation == 'above' and anchors:
        return [c for c in cands if any(above(c, a) for a in anchors)]
    if relation == 'below' and anchors:
        return [c for c in cands if any(below(c, a) for a in anchors)]
    if relation == 'with' and anchors:   # "table with kettle on it"
        return [c for c in cands if any(on_top(a, c) for a in anchors)]
    return cands


def relation_table(cands, anchors):

    lines = []
    for c in cands:
        for a in anchors:
            lines.append(f'dist(obj{c.oid}, obj{a.oid}) = {hdist(c, a):.2f} m; '
                         f'near={near(c, a)}; on={on_top(c, a)}; '
                         f'above={above(c, a)}; below={below(c, a)}')
    return '\n'.join(lines)
