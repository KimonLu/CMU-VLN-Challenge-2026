"""全局栅格图 + frontier 探索 + A*(报告 §6)。纯 CPU。"""
import numpy as np
import heapq

UNKNOWN, FREE, OBSTACLE = 0, 1, 2


def bootstrap_waypoints(origin_xy, cfg):
    """返回可选的旧式起步平移扫描点。

    比赛相机本身已有 360° 水平视野。四向各平移一次不仅不会补充朝向覆盖，
    在新增四场景的 20 题中还稳定制造约 3×8s/题的 timeout。因此默认关闭；
    保留显式开关只用于后续消融，而不是删除可回退行为。
    """
    if not cfg.get('init_sweep_enabled', False):
        return []
    origin = np.asarray(origin_xy, dtype=float)
    radius = float(cfg.get('init_spin_radius_m', 1.5))
    return [tuple(origin + radius * np.array([np.cos(ang), np.sin(ang)]))
            for ang in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]


def decimate(path, step=2.0):
    """A* 细路径 → 稀疏航点(保留拐点节奏与终点)。"""
    out, acc = [], 0.0
    for i in range(1, len(path)):
        acc += float(np.linalg.norm(np.asarray(path[i]) - np.asarray(path[i - 1])))
        if acc >= step or i == len(path) - 1:
            out.append(tuple(path[i]))
            acc = 0.0
    return out


def decimate_final_approach(path, step=2.0, final_step=1.0,
                            final_radius=3.0, return_start=False):
    """A* 路径分段抽稀：常规段用 ``step``，末段用更密的
    ``final_step``。

    ``final_radius`` 按沿路径到终点的剩余弧长计算，而非直线距离，
    因此家具区内的绕行路径也能稳定保留末段航点。
    ``return_start=True`` 时同时返回细航点后缀在输出中的起始下标。
    """
    if len(path) < 2:
        out = [tuple(path[0])] if path else []
        return (out, 0) if return_start else out
    if final_radius <= 0 or final_step <= 0:
        out = decimate(path, step)
        start = max(0, len(out) - 1)
        return (out, start) if return_start else out

    pts = [np.asarray(p, dtype=float) for p in path]
    remaining = np.zeros(len(pts), dtype=float)
    for i in range(len(pts) - 2, -1, -1):
        remaining[i] = remaining[i + 1] + float(
            np.linalg.norm(pts[i + 1] - pts[i]))

    # 第一个进入末段半径的原始 A* 点，同时作为粗/细两段边界。
    fine_start = next((i for i, d in enumerate(remaining)
                       if d <= final_radius), len(path) - 1)
    coarse = decimate(path[:fine_start + 1], step)
    fine = decimate(path[fine_start:], final_step)
    out = coarse + fine
    start = min(len(coarse), max(0, len(out) - 1))
    return (out, start) if return_start else out


def line_waypoints(start_xy, goal_xy, step=1.0):
    """当全局 A* 暂无路径时，仍以小步长把目标交给局部规划器。

    直线只是 waypoint 提示而非碰撞路径；系统局部规划器仍负责避障。这保证
    q5 式密集家具区不会因 A* 返回 None 而从远处一步直冲终点。
    """
    start = np.asarray(start_xy, dtype=float)
    goal = np.asarray(goal_xy, dtype=float)
    dist = float(np.linalg.norm(goal - start))
    if dist < 1e-9:
        return [tuple(goal)]
    n = max(1, int(np.ceil(dist / max(step, 1e-6))))
    return [tuple(start + (goal - start) * (i / n))
            for i in range(1, n + 1)]


class GridMap:
    """滚动扩张的全局 2D 栅格。"""

    def __init__(self, res=0.2, init_span=40.0, max_span=400.0):
        self.res = res
        n = int(init_span / res)
        self.grid = np.zeros((n, n), dtype=np.uint8)
        self.origin = np.array([-init_span / 2, -init_span / 2])  # 世界坐标
        self.max_span = max_span
        self._center0 = self.origin + init_span / 2

    def _sane(self, xy):
        """跨度硬上限的点过滤。SLAM 未收敛时错误配准的远点若不丢弃,
        _ensure 会为覆盖它分配巨型栅格,数秒吃爆内存(2026-07-08 两次事故)。"""
        return np.abs(np.atleast_2d(xy) - self._center0).max(axis=1) \
            < self.max_span / 2

    def _idx(self, xy):
        return ((np.atleast_2d(xy) - self.origin) / self.res).astype(int)

    def _ensure(self, xy):
        """按需非对称扩张,保证 xy(单点或 (N,2))全部落进栅格。"""
        ij = self._idx(xy)
        margin = 25
        lo, hi = ij.min(0), ij.max(0)
        shape = np.array(self.grid.shape)
        pad_lo = np.where(lo < 0, -lo + margin, 0)
        pad_hi = np.where(hi >= shape, hi - shape + 1 + margin, 0)
        if pad_lo.any() or pad_hi.any():
            g = np.zeros(tuple(shape + pad_lo + pad_hi), dtype=np.uint8)
            g[pad_lo[0]:pad_lo[0] + shape[0],
              pad_lo[1]:pad_lo[1] + shape[1]] = self.grid
            self.grid = g
            self.origin = self.origin - pad_lo * self.res

    def _idx_safe(self, xy):
        """_ensure 后的越界兜底(理论不触发)。"""
        ij = self._idx(xy)
        return np.clip(ij, 0, np.array(self.grid.shape) - 1)

    def update_terrain(self, pts, intensities, obs_thresh):
        """terrain_map_ext 一帧:低代价=FREE,高代价=OBSTACLE(OBSTACLE 不被 FREE 覆盖)。"""
        if len(pts) == 0:
            return
        m = self._sane(pts[:, :2])
        pts, intensities = pts[m], np.asarray(intensities)[m]
        if len(pts) == 0:
            return
        self._ensure(pts[:, :2])
        ij = self._idx_safe(pts[:, :2])
        obs = intensities > obs_thresh
        i, j = ij[:, 0], ij[:, 1]
        free_m = ~obs & (self.grid[i, j] != OBSTACLE)
        self.grid[i[free_m], j[free_m]] = FREE
        self.grid[i[obs], j[obs]] = OBSTACLE

    def add_obstacles(self, pts_xy):
        if len(pts_xy) == 0:
            return
        pts_xy = np.atleast_2d(pts_xy)[self._sane(pts_xy)]
        if len(pts_xy) == 0:
            return
        self._ensure(pts_xy)
        ij = self._idx_safe(pts_xy)
        self.grid[ij[:, 0], ij[:, 1]] = OBSTACLE

    def world(self, ij):
        return np.asarray(ij) * self.res + self.origin + self.res / 2

    # ---------- frontier ----------
    def frontiers(self, min_cluster=5):
        from scipy import ndimage
        g = self.grid
        free = g == FREE
        unknown_nb = ndimage.maximum_filter((g == UNKNOWN).astype(np.uint8), 3) > 0
        frontier = free & unknown_nb
        labels, n = ndimage.label(frontier)
        out = []
        for k in range(1, n + 1):
            ij = np.argwhere(labels == k)
            if len(ij) >= min_cluster:
                out.append({'centroid': self.world(ij.mean(0)), 'size': len(ij)})
        return out

    # ---------- A*(8 邻域,OBSTACLE 膨胀 + 额外代价区)----------
    def astar(self, start_xy, goal_xy, inflate=0.3, penalty_zones=None):
        from scipy import ndimage
        cost = np.ones_like(self.grid, dtype=np.float32)
        obs = ndimage.maximum_filter(
            (self.grid == OBSTACLE).astype(np.uint8),
            max(1, int(2 * inflate / self.res) + 1)) > 0
        cost[obs] = np.inf
        cost[self.grid == UNKNOWN] = 3.0          # 允许穿未知但有代价
        for z in (penalty_zones or []):           # z: ((x1,y1),(x2,y2),width,cost)
            self._paint_corridor(cost, *z)
        s = tuple(self._idx(start_xy)[0]); t = tuple(self._idx(goal_xy)[0])
        if not (0 <= t[0] < cost.shape[0] and 0 <= t[1] < cost.shape[1]):
            return None
        h = lambda a: np.hypot(a[0] - t[0], a[1] - t[1])
        openq = [(h(s), 0.0, s, None)]
        came, gscore = {}, {s: 0.0}
        while openq:
            _, g, cur, par = heapq.heappop(openq)
            if cur in came:
                continue
            came[cur] = par
            if cur == t:
                path = []
                while cur:
                    path.append(self.world(cur)); cur = came[cur]
                return path[::-1]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == dj == 0:
                        continue
                    nb = (cur[0] + di, cur[1] + dj)
                    if not (0 <= nb[0] < cost.shape[0] and 0 <= nb[1] < cost.shape[1]):
                        continue
                    c = cost[nb]
                    if not np.isfinite(c) or nb in came:
                        continue
                    ng = g + c * np.hypot(di, dj)
                    if ng < gscore.get(nb, np.inf):
                        gscore[nb] = ng
                        heapq.heappush(openq, (ng + h(nb), ng, nb, cur))
        return None

    def _paint_corridor(self, cost, p1, p2, width, c):
        n = max(2, int(np.hypot(*(np.array(p2) - p1)) / self.res) * 2)
        w = max(1, int(width / self.res))
        for t in np.linspace(0, 1, n):
            xy = np.array(p1) + t * (np.array(p2) - np.array(p1))
            i, j = self._idx(xy)[0]
            if 0 <= i < cost.shape[0] and 0 <= j < cost.shape[1]:
                sl = (slice(max(0, i - w), i + w + 1), slice(max(0, j - w), j + w + 1))
                finite = np.isfinite(cost[sl])
                cost[sl][finite] = np.maximum(cost[sl][finite], c)

    def nearest_free(self, xy, max_r=2.0):
        """物体旁最近可达点(报告 §9.4 goto 偏移)。"""
        best, bd = None, np.inf
        for r in np.arange(0.4, max_r, self.res):
            for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                p = np.array(xy) + r * np.array([np.cos(ang), np.sin(ang)])
                i, j = self._idx(p)[0]
                if 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1] \
                        and self.grid[i, j] == FREE:
                    d = np.linalg.norm(p - xy)
                    if d < bd:
                        best, bd = p, d
            if best is not None:
                return best
        return np.asarray(xy)

    def reachable_standoffs(self, start_xy, target_xy, min_r=0.8,
                            max_r=2.0, preferred_r=1.2,
                            min_clearance=0.35, penalty_zones=None,
                            angles=24, max_path_evals=12):
        """为物体目标生成按路径代价排序的安全停靠点。

        ``nearest_free`` 只看目标周围第一个 FREE 栅格，容易选到桌椅/墙体错误
        一侧。本方法先按自由空间余量筛选环形候选，再只对最有希望的少量候选
        跑 A*，最后综合真实路径长度、障碍余量和期望停靠半径排序。

        返回元素为 ``{'point','path','score','clearance','radius'}``；无可靠候选
        时返回空列表，由调用者执行旧兼容回退。
        """
        from scipy import ndimage

        start = np.asarray(start_xy, dtype=float)
        target = np.asarray(target_xy, dtype=float)
        min_r = max(float(min_r), self.res)
        max_r = max(float(max_r), min_r)
        preferred_r = float(np.clip(preferred_r, min_r, max_r))
        clearance = ndimage.distance_transform_edt(
            self.grid != OBSTACLE) * self.res

        # 0.4m 的径向间隔兼顾候选多样性与规划时间；按栅格去重防止
        # 低分辨率地图上相邻角度落到同一 cell。
        radial_step = max(0.4, 2 * self.res)
        radii = list(np.arange(min_r, max_r + 1e-6, radial_step))
        if not radii or radii[-1] < max_r - 0.15:
            radii.append(max_r)
        candidates = {}
        for radius in radii:
            for ang in np.linspace(0, 2 * np.pi, int(angles), endpoint=False):
                point = target + radius * np.array([np.cos(ang), np.sin(ang)])
                i, j = self._idx(point)[0]
                if not (0 <= i < self.grid.shape[0]
                        and 0 <= j < self.grid.shape[1]):
                    continue
                if self.grid[i, j] != FREE or clearance[i, j] < min_clearance:
                    continue
                # 先用廉价启发式挑出至多 max_path_evals 个候选，避免每个
                # instruction 对几十个点反复跑 Python A*。
                # 半径偏离必须足以抵消“越远离目标、离机器人越近”的路径
                # 捷径，否则排序会系统性选择 max_r，终点虽好走却离目标太远。
                pre_score = (float(np.linalg.norm(point - start))
                             + 0.35 / max(float(clearance[i, j]), 0.05)
                             + 1.5 * abs(radius - preferred_r))
                old = candidates.get((i, j))
                item = (pre_score, point, float(clearance[i, j]), float(radius))
                if old is None or pre_score < old[0]:
                    candidates[(i, j)] = item

        ranked = []
        shortlist = sorted(candidates.values(), key=lambda x: x[0])[
            :max(1, int(max_path_evals))]
        for _, point, clear, radius in shortlist:
            path = self.astar(start, point, penalty_zones=penalty_zones)
            if not path:
                continue
            path_len = (float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                        if len(path) > 1 else 0.0)
            score = (path_len + 0.35 / max(clear, 0.05)
                     + 1.5 * abs(radius - preferred_r))
            ranked.append({'point': np.asarray(point), 'path': path,
                           'score': score, 'clearance': clear,
                           'radius': radius})
        return sorted(ranked, key=lambda item: item['score'])

class Explorer:
    """frontier 选择 + 卡死处理。由 main_node 以 ~1Hz 驱动。"""

    def __init__(self, gridmap: GridMap, cfg, logger):
        self.gm = gridmap
        self.cfg = cfg
        self.log = logger
        self.blacklist = []           # 失败的 frontier 质心
        self.current_goal = None

    def next_goal(self, robot_xy):
        fs = self.gm.frontiers()
        fs = [f for f in fs
              if all(np.linalg.norm(f['centroid'] - b) > 1.0 for b in self.blacklist)]
        if not fs:
            return None
        best, best_s = None, -np.inf
        for f in fs:
            path = self.gm.astar(robot_xy, f['centroid'])
            cost = sum(np.linalg.norm(np.diff(path, axis=0), axis=1).tolist()) \
                if path else np.linalg.norm(f['centroid'] - robot_xy) * 3
            s = f['size'] / (1.0 + cost)
            if s > best_s:
                best, best_s = f, s
        self.current_goal = best['centroid']
        return self.current_goal

    def give_up_current(self):
        if self.current_goal is not None:
            self.blacklist.append(self.current_goal)
            self.log.warn(f'Frontier blacklisted: {self.current_goal}')
            self.current_goal = None

    def complete_current(self):
        """Retire a reached frontier so it cannot be selected in a tight loop.

        Mapping updates arrive asynchronously.  Without this short-lived
        spatial retirement, a frontier already within ``reach_dist_m`` can be
        selected hundreds of times before the grid changes, consuming most of
        the question deadline without moving the robot.
        """
        if self.current_goal is not None:
            self.blacklist.append(self.current_goal)
            self.current_goal = None
