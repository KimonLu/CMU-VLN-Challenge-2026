"""语义物体库:2D 检测 + 激光深度 → 3D 物体,增量合并(报告 §7.3–7.4)。"""
import numpy as np
from dataclasses import dataclass, field
from .perception import dominant_color
from .projection import transform_points

_SYNONYMS = {
    'couch': 'sofa',
    'refrigerator': 'fridge',
    'painting': 'picture',
    'television': 'tv',
    'potted plant': 'plant',
    'houseplant': 'plant',
    'garbage can': 'trash can',
    'trashcan': 'trash can',
    'bedside table': 'nightstand',
    'computer monitor': 'monitor',
}


def _norm_label(s):
    """小写、去尾 s、同义词归一。"""
    s = (s or '').lower().strip()
    if s.endswith('s') and not s.endswith('ss'):
        s = s[:-1]
    return _SYNONYMS.get(s, s)


@dataclass
class MapObject:
    oid: int
    label: str
    color: str
    center: np.ndarray          # map 系 (3,)
    size: np.ndarray            # (l, w, h)
    n_obs: int = 1
    weight: float = 1.0         # 累计激光点数
    best_crop: object = None    # BGR ndarray
    best_crop_area: float = 0.0

    def brief(self):
        c = self.center
        s = self.size
        return (f'[{self.oid}] {self.color} {self.label}, '
                f'center=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), '
                f'size={s[0]:.1f}x{s[1]:.1f}x{s[2]:.1f}, seen {self.n_obs} times')


class SemanticMap:
    THIN_CLASSES = {'window', 'door', 'picture', 'painting', 'mirror', 'curtain', 'TV'}

    def __init__(self, cfg, projector, logger):
        self.cfg = cfg
        self.proj = projector
        self.log = logger
        self.objects: list[MapObject] = []
        self._next_id = 0
        # 直接视觉计数使用的候选全景。JPEG 压缩后只保留少量高信息帧，
        # 避免保存 1920x640 原图导致内存随探索时长线性增长。
        self._viewpoints = []
        self._view_seq = 0

    def _remember_view(self, pano_bgr, detections):
        import cv2
        self._view_seq += 1
        counts = {}
        conf_sum = 0.0
        for det in detections:
            label = _norm_label(det.get('label'))
            counts[label] = counts.get(label, 0) + 1
            conf_sum += float(det.get('conf', 0.0))
        ok, encoded = cv2.imencode('.jpg', pano_bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return
        item = {'jpeg': encoded, 'counts': counts,
                'score': len(detections) + 0.1 * conf_sum,
                'seq': self._view_seq}
        self._viewpoints.append(item)
        cap = int(self.cfg.get('max_saved_views', 6))
        # 检测丰富的帧优先；完全无检测时保留较新的不同位姿帧。
        self._viewpoints.sort(key=lambda v: (v['score'], v['seq']), reverse=True)
        del self._viewpoints[cap:]

    def best_view(self, label=''):
        """返回最可能覆盖目标类别的一张 BGR 全景；没有帧时返回 None。"""
        import cv2
        if not self._viewpoints:
            return None
        q = _norm_label(label)
        best = max(self._viewpoints,
                   key=lambda v: (v['counts'].get(q, 0), v['score'], v['seq']))
        return cv2.imdecode(best['jpeg'], cv2.IMREAD_COLOR)

    # ---------- 主入口:处理一个关键帧 ----------
    def integrate(self, pano_bgr, detections, scan_map_pts, odom_pose):
        """detections 来自 4 视图,box 已换算为全景像素域 (u1,v1,u2,v2)。
        scan_map_pts: (N,3) map 系点云快照;odom_pose: 与图像同步的位姿。"""
        self._remember_view(pano_bgr, detections)
        pts_sensor = transform_points(scan_map_pts, odom_pose)
        pix, valid = self.proj.points_to_pixels(pts_sensor)
        pix, pts_map = pix[valid], scan_map_pts[valid]

        for det in detections:
            u1, v1, u2, v2 = det['box']
            # u 可越出 [0,W)(跨 360° 接缝),用相对中心的 wrap 距离判断
            W = self.proj.W
            uc, halfw = (u1 + u2) / 2, (u2 - u1) / 2
            du = (pix[:, 0] - uc + W / 2) % W - W / 2
            m = (np.abs(du) <= halfw) & (pix[:, 1] >= v1) & (pix[:, 1] <= v2)
            in_pts = pts_map[m]
            min_pts = self.cfg['min_lidar_pts']
            if _norm_label(det['label']) in {
                    _norm_label(c) for c in self.THIN_CLASSES}:
                min_pts = max(3, min_pts // 2)
            approximate = len(in_pts) < min_pts
            if len(in_pts) < self.cfg.get('approx_min_lidar_pts', min_pts):
                continue
            if approximate:
                # 玻璃/小物体常只有 1–7 个激光点。公开队伍的实践表明，先保留
                # 粗位置供关系推理、后续多视角再收敛，比永久丢弃更稳健。
                core = in_pts
                center = np.median(core, axis=0)
                size = np.full(3, self.cfg.get(
                    'approx_default_size_m', 0.3), dtype=float)
            else:
                core = self._nearest_cluster(in_pts, odom_pose)
                if core is None:
                    continue
                lo = np.percentile(core, 5, axis=0)
                hi = np.percentile(core, 95, axis=0)
                center = (lo + hi) / 2
                size = np.maximum(hi - lo, 0.05)
            # 外扩上下文再裁:紧框小图放大后 VLM 认不出(SJTU 实测,
            # 粉枕头紧裁后连人都难辨);带上周边环境复核才有线索。
            # copy:切片视图会把整张 pano 钉在内存,物体一多泄漏数百 MB
            margin = self.cfg.get('context_crop_margin_ratio', 0.25)
            pu, pv = margin * (u2 - u1), margin * (v2 - v1)
            crop = pano_bgr[max(0, int(v1 - pv)):max(0, int(v2 + pv)),
                            max(0, int(u1 - pu)):min(int(W), max(0, int(u2 + pu)))].copy()
            self._merge(det['label'], dominant_color(crop), center, size,
                        len(core), crop)

    def _nearest_cluster(self, pts, odom_pose):
        """按到相机距离直方图取最近主簇,剔除打到背景的穿透点(报告 §7.3)。"""
        cam = np.array(odom_pose[:3])
        d = np.linalg.norm(pts - cam, axis=1)
        if len(pts) < 12:
            # 点太少不足以分簇(薄物体类常见):取中位距离 ±1m,防单点穿透拉偏
            m = np.abs(d - np.median(d)) < 1.0
            return pts[m] if m.sum() >= 3 else None
        hist, edges = np.histogram(d, bins=24)
        # 从最近的达标 bin 起,聚合连续非空区间
        nz = np.flatnonzero(hist >= max(1, int(0.05 * len(pts))))
        if len(nz) == 0:
            return None
        first = int(nz[0])
        last = first
        while last + 1 < len(hist) and hist[last + 1] > 0:
            last += 1
        m = (d >= edges[first]) & (d <= edges[last + 1])
        return pts[m] if m.sum() >= 3 else None

    def _merge(self, label, color, center, size, weight, crop):
        thr = max(self.cfg['merge_dist_m'], 0.5 * float(np.max(size)))
        for o in self.objects:
            if (_norm_label(o.label) == _norm_label(label)
                    and np.linalg.norm(o.center - center) < thr):
                w = o.weight + weight
                o.center = (o.center * o.weight + center * weight) / w
                o.size = (o.size * o.weight + size * weight) / w
                o.weight = w
                o.n_obs += 1
                area = crop.shape[0] * crop.shape[1]
                if area > o.best_crop_area:
                    o.best_crop, o.best_crop_area = crop, area
                if o.color == 'unknown':
                    o.color = color
                return
        self.objects.append(MapObject(self._next_id, label, color, center,
                                      size, best_crop=crop,
                                      best_crop_area=crop.shape[0] * crop.shape[1]))
        self._next_id += 1

    # ---------- 查询 ----------
    def confirmed(self):
        return [o for o in self.objects if o.n_obs >= self.cfg['confirm_obs']]

    def by_label(self, label, confirmed_only=True):
        """类别查询:归一化(同义词/复数)后全等,或查询词==label 的尾词
        (query "table" 命中 "coffee table";"book" 不命中 "bookshelf")。"""
        src = self.confirmed() if confirmed_only else self.objects
        q = _norm_label(label)
        if not q:
            return []
        out = []
        for o in src:
            ol = _norm_label(o.label)
            if ol == q or _norm_label(ol.split()[-1]) == q:
                out.append(o)
        return out

    def scene_text(self, relevant_words=None, limit=80):
        objs = self.confirmed() or list(self.objects)
        if relevant_words:
            rw = [w.lower() for w in relevant_words]
            objs = sorted(objs, key=lambda o: (not any(
                w in o.label.lower() or o.label.lower() in w for w in rw)))
        lines = [o.brief() for o in objs[:limit]]
        return 'Objects in scene (map frame, meters):\n' + '\n'.join(lines)
