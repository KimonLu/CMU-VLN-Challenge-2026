#!/usr/bin/env python3
"""投影符号标定(报告 §7.1 TODO(calibrate)):自动判定 azimuth_flip / elevation_flip。

原理:把 /registered_scan 变换回 sensor 系并按 4 种 flip 组合投到全景图,
正确组合下"点云深度结构"与"图像内容"对齐 → 两个指标都应最高:
  1. MI  — 投影点深度 vs 图像灰度的互信息
  2. EDGE — 粗栅格深度图边缘 vs 图像边缘的相关系数
并为每种组合输出叠加可视化 PNG,供人眼终审(以人眼为准)。

用法(ai_module 容器内,仿真已跑通或有 bag):
  python3 tools/calibrate_projection.py --bag /path/to/dev_bag
  python3 tools/calibrate_projection.py --live 30
结论回填 ai_module/src/smart_vlm/config/params.yaml 的 projection 段。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'smart_vlm'))
from smart_vlm.projection import PanoProjector, transform_points  # noqa: E402
import bag_io  # noqa: E402

TOPICS = ('/camera/image', '/registered_scan', '/state_estimation')
COMBOS = [(False, False), (True, False), (False, True), (True, True)]


def mutual_info(a, b, bins=24):
    h, _, _ = np.histogram2d(a, b, bins=bins)
    p = h / max(h.sum(), 1)
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def edge_correlation(pix, depth, gray, ds=4):
    """粗栅格深度图 Sobel vs 图像 Sobel 的掩码相关系数。"""
    import cv2
    H, W = gray.shape[0] // ds, gray.shape[1] // ds
    dmap = np.full((H, W), np.nan, np.float32)
    u = np.clip((pix[:, 0] / ds).astype(int), 0, W - 1)
    v = np.clip((pix[:, 1] / ds).astype(int), 0, H - 1)
    for ui, vi, di in zip(u, v, depth):          # 同格取最近深度
        if np.isnan(dmap[vi, ui]) or di < dmap[vi, ui]:
            dmap[vi, ui] = di
    mask = ~np.isnan(dmap)
    if mask.sum() < 100:
        return 0.0
    dfill = np.where(mask, dmap, np.nanmedian(dmap)).astype(np.float32)
    ge = cv2.Sobel(cv2.resize(gray, (W, H)).astype(np.float32), cv2.CV_32F, 1, 1)
    de = cv2.Sobel(dfill, cv2.CV_32F, 1, 1)
    ge, de = np.abs(ge)[mask], np.abs(de)[mask]
    if ge.std() < 1e-6 or de.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(ge, de)[0, 1])


def overlay_png(pano, pix, depth, path):
    import cv2
    img = pano.copy()
    dn = np.clip((depth - 0.3) / 8.0 * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(dn.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    for (u, v), c in zip(pix.astype(int), colors):
        cv2.circle(img, (u % pano.shape[1], min(v, pano.shape[0] - 1)),
                   1, tuple(int(x) for x in c), -1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', default=None)
    ap.add_argument('--live', type=float, default=None, metavar='SECONDS')
    ap.add_argument('--samples', type=int, default=10)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), 'out'))
    args = ap.parse_args()

    if args.bag:
        src = bag_io.read_bag(args.bag, TOPICS)
    elif args.live:
        from sensor_msgs.msg import Image, PointCloud2
        from nav_msgs.msg import Odometry
        src = bag_io.live_source({'/camera/image': Image,
                                  '/registered_scan': PointCloud2,
                                  '/state_estimation': Odometry}, args.live)
    else:
        raise SystemExit('--bag <path> 或 --live <秒数> 必须给一个')

    samples = bag_io.collect_synced_samples(src, args.samples)
    if len(samples) < 3:
        raise SystemExit(f'同步样本不足({len(samples)}<3):确认三个 topic 都在发布,'
                         '或加长 --live 时间/换更长的 bag')
    print(f'采到 {len(samples)} 组同步样本,开始评测 4 种 flip 组合…')

    import cv2
    os.makedirs(args.out, exist_ok=True)
    results = []
    for az, el in COMBOS:
        proj = PanoProjector({'img_w': 1920, 'img_h': 640, 'vfov_deg': 120,
                              'azimuth_flip': az, 'elevation_flip': el})
        mi_d, mi_g, edges = [], [], []
        for i, (pano, scan, pose) in enumerate(samples):
            pts = transform_points(scan, pose)
            pix, valid = proj.points_to_pixels(pts)
            pix, pts = pix[valid], pts[valid]
            if len(pix) == 0:
                continue
            depth = np.linalg.norm(pts, axis=1)
            gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
            u = np.clip(pix[:, 0].astype(int), 0, gray.shape[1] - 1)
            v = np.clip(pix[:, 1].astype(int), 0, gray.shape[0] - 1)
            mi_d.append(depth)
            mi_g.append(gray[v, u].astype(float))
            edges.append(edge_correlation(pix, depth, gray))
            if i == 0:
                overlay_png(pano, pix, depth, os.path.join(
                    args.out, f'calib_az{int(az)}_el{int(el)}.png'))
        mi = mutual_info(np.concatenate(mi_d), np.concatenate(mi_g))
        ec = float(np.mean(edges))
        results.append({'az': az, 'el': el, 'mi': mi, 'edge': ec})
        print(f'  azimuth_flip={az!s:5}  elevation_flip={el!s:5}  '
              f'MI={mi:.4f}  EDGE={ec:.4f}')

    best_mi = max(results, key=lambda r: r['mi'])
    best_ed = max(results, key=lambda r: r['edge'])
    print(f'\n叠加图已存 {args.out}/calib_az*_el*.png — 人眼确认:'
          '激光点应贴合图像中物体轮廓(墙角/家具边缘)')
    if (best_mi['az'], best_mi['el']) == (best_ed['az'], best_ed['el']):
        print('两指标一致,推荐写入 params.yaml:')
        print(f"  azimuth_flip: {str(best_mi['az']).lower()}")
        print(f"  elevation_flip: {str(best_mi['el']).lower()}")
    else:
        print(f"两指标不一致(MI 推 az={best_mi['az']},el={best_mi['el']};"
              f"EDGE 推 az={best_ed['az']},el={best_ed['el']})——以叠加图人眼为准")


if __name__ == '__main__':
    main()
