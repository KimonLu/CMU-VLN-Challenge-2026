#!/usr/bin/env python3
"""Estimate a terrain obstacle threshold from intensity samples."""



import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_io  # noqa: E402

TOPIC = '/terrain_map_ext'


def otsu_threshold(vals, bins=256):
    hist, edges = np.histogram(vals, bins=bins)
    p = hist.astype(float) / max(hist.sum(), 1)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu_cum = np.cumsum(p * mids)
    mu_t = mu_cum[-1]
    w1 = 1 - w0
    var = w0 * w1 * (mu_cum / np.maximum(w0, 1e-12)
                     - (mu_t - mu_cum) / np.maximum(w1, 1e-12)) ** 2
    return float(mids[np.argmax(var)])


def ascii_hist(vals, bins=40, width=60):
    hist, edges = np.histogram(vals, bins=bins)
    top = hist.max()
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = '#' * int(h / max(top, 1) * width)
        print(f'  {lo:8.3f}-{hi:8.3f} |{bar} {h}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', default=None)
    ap.add_argument('--live', type=float, default=None, metavar='SECONDS')
    ap.add_argument('--max-frames', type=int, default=150)
    args = ap.parse_args()

    if args.bag:
        src = bag_io.read_bag(args.bag, (TOPIC,))
    elif args.live:
        from sensor_msgs.msg import PointCloud2
        src = bag_io.live_source({TOPIC: PointCloud2}, args.live)
    else:
        raise SystemExit('provide either --bag <path> or --live <seconds>')

    chunks, frames = [], 0
    for _, msg, _ in src:
        _, inten = bag_io.pc2_to_xyz_intensity(msg)
        if len(inten):
            chunks.append(inten)
            frames += 1
        if frames >= args.max_frames:
            break
    if not chunks:
        raise SystemExit(f'no data received on {TOPIC}; verify the simulator or bag contents')
    vals = np.concatenate(chunks)
    print(f'{frames} frames, {len(vals)} points\n')
    print('quantiles:')
    for q in (1, 5, 25, 50, 75, 90, 95, 99):
        print(f'  p{q:02d} = {np.percentile(vals, q):.4f}')
    print(f'  min = {vals.min():.4f}  max = {vals.max():.4f}\n')
    print('histogram:')
    ascii_hist(vals)
    thr = otsu_threshold(vals)
    print(f'\nrecommended Otsu threshold: {thr:.3f}')
    print('For a clearly bimodal histogram, select the valley between ground and obstacle modes; '
          'otherwise validate the Otsu value in simulation.')
    print(f'params.yaml value:\n  terrain_obstacle_intensity: {thr:.3f}')


if __name__ == '__main__':
    main()
