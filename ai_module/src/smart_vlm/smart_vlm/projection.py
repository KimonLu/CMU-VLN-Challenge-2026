"""全景图 ⇄ 3D 几何(报告 §7.1)。

约定:图像 1920x640,等距柱状投影,HFOV=360°,VFOV=120°。
camera frame 与 sensor(激光)frame 对齐(官方已标定重映射)。
!! 方位角零点与符号必须用 rosbag 实测校验(config: azimuth_flip / elevation_flip)。
"""
import numpy as np
import cv2


class PanoProjector:
    def __init__(self, cfg):
        self.W = cfg['img_w']
        self.H = cfg['img_h']
        self.vfov = np.deg2rad(cfg['vfov_deg'])
        self.az_sign = -1.0 if cfg.get('azimuth_flip') else 1.0
        self.el_sign = -1.0 if cfg.get('elevation_flip') else 1.0
        self._view_maps = None

    # ---------- 激光点(sensor 系) -> 全景像素 ----------
    def points_to_pixels(self, pts):
        """pts: (N,3) sensor 系。返回 (N,2) 像素坐标 + 有效掩码。"""
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r_xy = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x) * self.az_sign          # 方位角 [-pi, pi]
        phi = np.arctan2(z, r_xy) * self.el_sign         # 俯仰角
        u = (0.5 - theta / (2 * np.pi)) * self.W % self.W
        v = (0.5 - phi / self.vfov) * self.H
        valid = (v >= 0) & (v < self.H) & (r_xy > 0.05)
        return np.stack([u, v], axis=1), valid

    # ---------- 全景像素 -> 单位方向向量(sensor 系) ----------
    def pixel_to_ray(self, u, v):
        theta = (0.5 - u / self.W) * 2 * np.pi * self.az_sign
        phi = (0.5 - v / self.H) * self.vfov * self.el_sign
        return np.array([np.cos(phi) * np.cos(theta),
                         np.cos(phi) * np.sin(theta),
                         np.sin(phi)])

    # ---------- 全景 -> 4 个 90°(重叠15°)虚拟针孔视图 ----------
    def make_views(self, pano, out_size=640, hfov_deg=105.0):
        """返回 [(view_img, yaw_center_rad), ...] 共 4 个,朝向 0/90/180/270°。"""
        if self._view_maps is None:
            self._view_maps = [self._build_map(np.deg2rad(yaw), out_size,
                                               np.deg2rad(hfov_deg))
                               for yaw in (0, 90, 180, 270)]
        views = []
        for (mx, my), yaw in zip(self._view_maps,
                                 np.deg2rad([0, 90, 180, 270])):
            views.append((cv2.remap(pano, mx, my, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_WRAP), yaw))
        return views

    def _build_map(self, yaw, size, hfov):
        f = (size / 2) / np.tan(hfov / 2)
        i, j = np.meshgrid(np.arange(size), np.arange(size))  # j=row, i=col
        # 针孔相机系: x 前, y 左, z 上
        xc = np.full_like(i, f, dtype=np.float64)
        yc = -(i - size / 2)
        zc = -(j - size / 2)
        # 旋转到全景 yaw 朝向
        x = xc * np.cos(yaw) - yc * np.sin(yaw)
        y = xc * np.sin(yaw) + yc * np.cos(yaw)
        z = zc
        theta = np.arctan2(y, x) * self.az_sign
        phi = np.arctan2(z, np.sqrt(x * x + y * y)) * self.el_sign
        u = ((0.5 - theta / (2 * np.pi)) * self.W) % self.W
        v = (0.5 - phi / self.vfov) * self.H
        return u.astype(np.float32), v.astype(np.float32)

    def view_pts_to_pano(self, pts_uv, yaw, out_size=640, hfov_deg=105.0):
        """视图像素 (N,2) -> 全景像素 (N,2)(逐点精确映射)。"""
        pts = np.atleast_2d(np.asarray(pts_uv, dtype=np.float64))
        f = (out_size / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
        xc = np.full(len(pts), f)
        yc = -(pts[:, 0] - out_size / 2)
        zc = -(pts[:, 1] - out_size / 2)
        x = xc * np.cos(yaw) - yc * np.sin(yaw)
        y = xc * np.sin(yaw) + yc * np.cos(yaw)
        theta = np.arctan2(y, x) * self.az_sign
        phi = np.arctan2(zc, np.hypot(x, y)) * self.el_sign
        u = ((0.5 - theta / (2 * np.pi)) * self.W) % self.W
        v = (0.5 - phi / self.vfov) * self.H
        return np.stack([u, v], axis=1)

    def view_box_to_pano(self, box, yaw, out_size=640, hfov_deg=105.0):
        """把视图内 bbox(x1,y1,x2,y2)中心映射回全景像素(用于关联激光点)。"""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        u, v = self.view_pts_to_pano([(cx, cy)], yaw, out_size, hfov_deg)[0]
        return float(u), float(v)

    def view_box_to_pano_box(self, box, yaw, out_size=640, hfov_deg=105.0):
        """视图 bbox -> 全景 bbox(4 角点+中心精确映射,宽度按角分辨率换算)。
        返回 (u1,v1,u2,v2);u 可越出 [0,W)(跨 360° 接缝),消费方需 wrap 判断。"""
        x1, y1, x2, y2 = box
        pts = self.view_pts_to_pano(
            [(x1, y1), (x2, y1), (x1, y2), (x2, y2),
             ((x1 + x2) / 2, (y1 + y2) / 2)], yaw, out_size, hfov_deg)
        uc = pts[4, 0]
        du = (pts[:4, 0] - uc + self.W / 2) % self.W - self.W / 2  # 相对中心去 wrap
        u1p, u2p = uc + du.min(), uc + du.max()
        v1p = max(0.0, float(pts[:, 1].min()))
        v2p = min(float(self.H), float(pts[:, 1].max()))
        return float(u1p), v1p, float(u2p), v2p


def transform_points(pts_map, odom_pose):
    """map 系点 -> sensor 系。odom_pose: (x, y, z, qx, qy, qz, qw)。"""
    from scipy.spatial.transform import Rotation as R
    t = np.array(odom_pose[:3])
    rot = R.from_quat(odom_pose[3:])
    return rot.inv().apply(pts_map - t)
