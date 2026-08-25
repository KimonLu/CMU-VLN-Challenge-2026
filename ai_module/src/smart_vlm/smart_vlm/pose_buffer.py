"""Thread-safe timestamp buffers for delayed sensor synchronization."""

import bisect
import math
import threading
from collections import deque


def stamp_to_sec(stamp):

    return stamp.sec + stamp.nanosec * 1e-9


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def ang_diff(a, b):

    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def keyframe_due(pose, last_xy, last_yaw, trans_m, rot_deg):


    x, y = pose[0], pose[1]
    yaw = yaw_from_quat(*pose[3:7])
    if last_xy is None:
        return True, (x, y), yaw
    trans = math.hypot(x - last_xy[0], y - last_xy[1])
    rot = ang_diff(yaw, last_yaw)
    return (trans >= trans_m or rot >= math.radians(rot_deg)), (x, y), yaw


class PoseBuffer:


    def __init__(self, maxlen=4096, max_gap_s=0.20):
        self.max_gap = max_gap_s
        self._t = deque(maxlen=maxlen)
        self._p = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, t, pose):
        with self._lock:
            if self._t and t <= self._t[-1]:
                return
            self._t.append(float(t))
            self._p.append(tuple(pose))

    def query(self, t):

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
