"""位姿时间缓冲 + 关键帧判据(报告 §7.1 时间同步)。

不 import rclpy/numpy 以便脱离 ROS 单测;时间戳用 float 秒。
"""
import bisect
import math
import threading
from collections import deque


def stamp_to_sec(stamp):
    """builtin_interfaces/Time → float 秒。"""
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def ang_diff(a, b):
    """两角最小夹角,弧度,[0, pi]。"""
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def keyframe_due(pose, last_xy, last_yaw, trans_m, rot_deg):
    """关键帧判据:平移或转角任一超限。pose: (x,y,z,qx,qy,qz,qw)。
    返回 (due, xy, yaw) — xy/yaw 供触发后更新 last_*。"""
    x, y = pose[0], pose[1]
    yaw = yaw_from_quat(*pose[3:7])
    if last_xy is None:
        return True, (x, y), yaw
    trans = math.hypot(x - last_xy[0], y - last_xy[1])
    rot = ang_diff(yaw, last_yaw)
    return (trans >= trans_m or rot >= math.radians(rot_deg)), (x, y), yaw


class PoseBuffer:
    """按 header.stamp 存位姿历史,查询与图像时间戳最近的位姿。线程安全。"""

    def __init__(self, maxlen=4096, max_gap_s=0.20):
        self.max_gap = max_gap_s
        self._t = deque(maxlen=maxlen)
        self._p = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, t, pose):
        with self._lock:
            if self._t and t <= self._t[-1]:
                return                      # 乱序/重复样本丢弃
            self._t.append(float(t))
            self._p.append(tuple(pose))

    def query(self, t):
        """最近邻查询;与最近样本时间差 > max_gap 返回 None(调用方应丢帧)。"""
        with self._lock:
            if not self._t:
                return None
            ts = list(self._t)
            i = bisect.bisect_left(ts, t)
            best = min([k for k in (i - 1, i) if 0 <= k < len(ts)],
                       key=lambda k: abs(ts[k] - t))
            if abs(ts[best] - t) > self.max_gap:
                return None
            return self._p[best]

    @property
    def latest(self):
        with self._lock:
            return self._p[-1] if self._p else None


class TimedValueBuffer:
    """低频大对象的时间戳最近邻缓冲（用于 registered_scan）。"""

    def __init__(self, maxlen=128, max_gap_s=0.30):
        self.max_gap = max_gap_s
        self._t = deque(maxlen=maxlen)
        self._v = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, t, value):
        with self._lock:
            if self._t and t <= self._t[-1]:
                return
            self._t.append(float(t))
            self._v.append(value)

    def query(self, t):
        with self._lock:
            if not self._t:
                return None
            ts = list(self._t)
            i = bisect.bisect_left(ts, t)
            candidates = [k for k in (i - 1, i) if 0 <= k < len(ts)]
            best = min(candidates, key=lambda k: abs(ts[k] - t))
            if abs(ts[best] - t) > self.max_gap:
                return None
            return self._v[best]
