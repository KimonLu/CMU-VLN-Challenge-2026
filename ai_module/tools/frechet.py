"""离散 Fréchet 距离(纯 numpy,迭代 DP)。用于指令题轨迹 vs 真值 ply 对比。"""
import numpy as np


def resample(traj, max_pts=400):
    """降采样到 <=max_pts 点,控制 O(n*m) 规模。traj: (N,2+)."""
    traj = np.asarray(traj, dtype=float)
    if len(traj) <= max_pts:
        return traj
    idx = np.linspace(0, len(traj) - 1, max_pts).astype(int)
    return traj[idx]


def frechet_distance(p, q, max_pts=400):
    """离散 Fréchet 距离(米)。p, q: (N,>=2) 只用前两列(xy)。"""
    p = resample(np.asarray(p, dtype=float)[:, :2], max_pts)
    q = resample(np.asarray(q, dtype=float)[:, :2], max_pts)
    if len(p) == 0 or len(q) == 0:
        return float('inf')
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)   # (n,m)
    n, m = d.shape
    ca = np.empty((n, m))
    ca[0, 0] = d[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(d[i, j],
                           min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]))
    return float(ca[-1, -1])


def endpoint_distance(p, q):
    """两轨迹终点的欧氏距离(米)。"""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if len(p) == 0 or len(q) == 0:
        return float('inf')
    return float(np.linalg.norm(p[-1, :2] - q[-1, :2]))
