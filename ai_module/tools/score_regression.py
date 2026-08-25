#!/usr/bin/env python3
"""离线回归打分:tools/out/*.json(run_one_question 产出)→ results.csv。

无 ROS 依赖,本机 venv 可跑:
  python3 tools/score_regression.py --results tools/out --questions questions

打分口径(近似,非官方):
  numerical  : 与 answer_key.yaml 精确匹配 → 1/0
  instruction: 轨迹 vs trajectory_q4/q5.ply 的离散 Fréchet + 终点距离,
               score ≈ 6 × (0.5×exp(-frechet/3) + 0.5×exp(-endpoint/1.5))
  objref     : 无机器真值 → 打印 marker 参数,与 questions.pdf 人工比对
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frechet import endpoint_distance, frechet_distance  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


def load_gt_trajectory(questions_dir, scene, q):
    import numpy as np
    ply = os.path.join(questions_dir, scene, f'trajectory_q{q}.ply')
    if not os.path.exists(ply):
        return None
    # The challenge trajectories are small ASCII PLY vertex tables.  Parsing
    # this narrow, documented subset avoids requiring the non-runtime
    # ``plyfile`` package merely to run the scorer in the submission image.
    properties = []
    header_lines = 0
    is_ascii = False
    in_vertex = False
    with open(ply, encoding='ascii') as fp:
        for line in fp:
            header_lines += 1
            fields = line.strip().split()
            if fields[:2] == ['format', 'ascii']:
                is_ascii = True
            elif fields[:2] == ['element', 'vertex']:
                in_vertex = True
            elif fields and fields[0] == 'element':
                in_vertex = False
            elif in_vertex and fields[:1] == ['property']:
                properties.append(fields[-1])
            elif fields[:1] == ['end_header']:
                break
        else:
            raise ValueError(f'PLY header incomplete: {ply}')
    if not is_ascii or 'x' not in properties or 'y' not in properties:
        raise ValueError(f'unsupported trajectory PLY: {ply}')
    data = np.loadtxt(ply, skiprows=header_lines, ndmin=2)
    return data[:, [properties.index('x'), properties.index('y')]]


def score_instruction(frechet_m, endpoint_m):
    return 6.0 * (0.5 * math.exp(-frechet_m / 3.0)
                  + 0.5 * math.exp(-endpoint_m / 1.5))


def aabb_iou(center_a, size_a, center_b, size_b):
    """Axis-aligned 3D box IoU used only as a development proxy."""
    import numpy as np
    ca, sa = np.asarray(center_a, float), np.asarray(size_a, float)
    cb, sb = np.asarray(center_b, float), np.asarray(size_b, float)
    if ca.shape != (3,) or cb.shape != (3,) or sa.shape != (3,) or sb.shape != (3,):
        return 0.0
    if not (np.isfinite(ca).all() and np.isfinite(cb).all()
            and np.isfinite(sa).all() and np.isfinite(sb).all()):
        return 0.0
    if (sa <= 0).any() or (sb <= 0).any():
        return 0.0
    lo = np.maximum(ca - sa / 2, cb - sb / 2)
    hi = np.minimum(ca + sa / 2, cb + sb / 2)
    inter = float(np.prod(np.maximum(hi - lo, 0.0)))
    union = float(np.prod(sa) + np.prod(sb) - inter)
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(
        os.path.dirname(__file__), 'out'), help='run_one_question 输出目录')
    ap.add_argument('--questions', default=os.path.join(
        os.path.dirname(__file__), '..', '..', 'questions'),
        help='官方 questions/ 目录')
    ap.add_argument('--out', default=None, help='results.csv 路径')
    args = ap.parse_args()

    key_path = os.path.join(os.path.dirname(__file__), 'answer_key.yaml')
    keys = yaml.safe_load(open(key_path)) if yaml else {}
    answer_key = keys.get('numerical', {})
    object_key = keys.get('object_reference', {})

    rows = []
    # A full run is stored as <run>/<scene>/qN/*.json.  Recursive discovery
    # also keeps the historical flat tools/out layout compatible.
    pattern = os.path.join(args.results, '**', '*.json')
    for path in sorted(glob.glob(pattern, recursive=True)):
        r = json.load(open(path))
        scene, q, qtype = r['scene'], r['q'], r['qtype']
        row = {'scene': scene, 'q': q, 'qtype': qtype,
               'duration_s': round(r.get('duration', 0), 1),
               'answer': '', 'expected': '', 'frechet_m': '',
               'endpoint_m': '', 'center_error_m': '', 'iou_3d': '',
               'score': '', 'max_score': '', 'notes': ''}
        if qtype == 'numerical':
            got, exp = r.get('numerical'), answer_key.get(scene)
            row.update(answer=got, expected=exp, max_score=1,
                       score=int(got == exp and got is not None))
        elif qtype == 'object_reference':
            mk = r.get('marker')
            gt = object_key.get(scene, {}).get(q)
            answer = (f"{mk['ns']} c={mk['center']} s={mk['size']}"
                      if mk else 'NO MARKER')
            if mk and gt:
                import numpy as np
                err = float(np.linalg.norm(
                    np.asarray(mk['center'], float) - np.asarray(gt['center'], float)))
                iou = aabb_iou(mk['center'], mk['size'],
                               gt['center'], gt['size'])
                label_ok = str(gt['label']).lower() in str(mk['ns']).lower()
                row.update(max_score=2, score=round(2.0 * iou, 2),
                           answer=answer,
                           expected=f"{gt['label']} object_id={gt['object_id']}",
                           center_error_m=round(err, 3), iou_3d=round(iou, 4),
                           notes=('3D AABB IoU 代理分,非官方口径; label=' +
                                  ('OK' if label_ok else 'MISMATCH')))
            else:
                row.update(max_score=2, score=0, answer=answer,
                           notes='无 Marker 或缺少开发集真值')
        else:
            traj = r.get('trajectory') or []
            gt = load_gt_trajectory(args.questions, scene, q)
            if gt is None or len(traj) < 2:
                row.update(max_score=6, score=0, notes='轨迹缺失')
            else:
                xy = [(p[1], p[2]) for p in traj]        # [t,x,y] → (x,y)
                f = frechet_distance(xy, gt)
                e = endpoint_distance(xy, gt)
                row.update(frechet_m=round(f, 2), endpoint_m=round(e, 2),
                           max_score=6, score=round(score_instruction(f, e), 2),
                           notes='近似分,非官方口径')
        rows.append(row)

    out = args.out or os.path.join(args.results, 'results.csv')
    if rows:
        with open(out, 'w', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    total = sum(float(r['score']) for r in rows if r['score'] != '')
    maxsc = sum(float(r['max_score']) for r in rows
                if r['max_score'] != '' and r['score'] != '')
    print(f'{len(rows)} 条结果 → {out}')
    print(f'可自动打分部分: {total:.1f} / {maxsc:.0f} (指代题需人工比对)')
    for r in rows:
        print(f"  {r['scene']} q{r['q']} [{r['qtype'][:5]}] "
              f"score={r['score']} {r['notes']}")


if __name__ == '__main__':
    main()
