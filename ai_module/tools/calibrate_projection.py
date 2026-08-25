#!/usr/bin/env python3
"""Estimate panorama projection signs from synchronized sensor samples."""



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

    import cv2
    H, W = gray.shape[0] // ds, gray.shape[1] // ds
    dmap = np.full((H, W), np.nan, np.float32)
    u = np.clip((pix[:, 0] / ds).astype(int), 0, W - 1)
    v = np.clip((pix[:, 1] / ds).astype(int), 0, H - 1)
    for ui, vi, di in zip(u, v, depth):
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
        raise SystemExit('provide either --bag <path> or --live <seconds>')

    samples = bag_io.collect_synced_samples(src, args.samples)
    if len(samples) < 3:
        raise SystemExit(f'not enough synchronized samples ({len(samples)}<3); '
                         'verify all three topics or use a longer capture')
    print(f'collected {len(samples)} synchronized samples; evaluating four flip combinations')

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
    print(f'\noverlays saved to {args.out}/calib_az*_el*.png; '
          'verify that LiDAR points align with visible object boundaries')
    if (best_mi['az'], best_mi['el']) == (best_ed['az'], best_ed['el']):
        print('both metrics agree; recommended params.yaml values:')
        print(f"  azimuth_flip: {str(best_mi['az']).lower()}")
        print(f"  elevation_flip: {str(best_mi['el']).lower()}")
    else:
        print(f"metrics disagree (MI: az={best_mi['az']}, el={best_mi['el']}; "
              f"EDGE: az={best_ed['az']}, el={best_ed['el']}); inspect the overlays")


if __name__ == '__main__':
    main()
