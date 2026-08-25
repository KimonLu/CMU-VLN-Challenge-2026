#!/usr/bin/env python3
"""terrain_map intensity 标定(报告 §6.1 TODO(calibrate)):
采集 /terrain_map_ext 的 intensity 分布,给出分位数表 + Otsu 建议阈值。

用法(ai_module 容器内):
  python3 tools/terrain_intensity_hist.py --bag /path/to/dev_bag
  python3 tools/terrain_intensity_hist.py --live 30
结论回填 params.yaml 的 mapping.terrain_obstacle_intensity。
注意:采集时机器人应既看到开阔地面又看到障碍(先动几步再采)。
"""
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
        print(f'  {lo:8.3f}–{hi:8.3f} |{bar} {h}')


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
        raise SystemExit('--bag <path> 或 --live <秒数> 必须给一个')

    chunks, frames = [], 0
    for _, msg, _ in src:
        _, inten = bag_io.pc2_to_xyz_intensity(msg)
        if len(inten):
            chunks.append(inten)
            frames += 1
        if frames >= args.max_frames:
            break
    if not chunks:
        raise SystemExit(f'没收到 {TOPIC} 数据:确认仿真已启动/bag 含该 topic')
    vals = np.concatenate(chunks)
    print(f'{frames} 帧,共 {len(vals)} 点\n')
    print('分位数表:')
    for q in (1, 5, 25, 50, 75, 90, 95, 99):
        print(f'  p{q:02d} = {np.percentile(vals, q):.4f}')
    print(f'  min = {vals.min():.4f}  max = {vals.max():.4f}\n')
    print('直方图:')
    ascii_hist(vals)
    thr = otsu_threshold(vals)
    print(f'\nOtsu 建议阈值: {thr:.3f}')
    print('若直方图呈明显双峰(地面≈低值/障碍≈高值),取谷底;'
          '否则从 Otsu 值起在仿真里试跑验证(过低→到处是墙,过高→撞障碍)。')
    print(f'写入 params.yaml:\n  terrain_obstacle_intensity: {thr:.3f}')


if __name__ == '__main__':
    main()
