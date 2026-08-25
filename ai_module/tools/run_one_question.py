#!/usr/bin/env python3
"""模拟评测节点:发一道题 + 录响应与轨迹(在 ai_module 容器内运行,需 rclpy)。

前置:仿真已启动(对应场景)、smart_vlm 已启动。
用法(容器内):
  python3 tools/run_one_question.py --scene arabic_room --q 1
  python3 tools/run_one_question.py --question "How many chairs" --qtype numerical

q 与题型对应:1=numerical, 2/3=object_reference, 4/5=instruction_following。
行为:1Hz 重复发布问题;numerical/objref 收到答案后 10s 结束,
instruction 录满 --duration(默认 600s,可 Ctrl-C 提前结束,结果仍会写盘)。
输出:tools/out/<scene>_q<q>.json
"""
import argparse
import json
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

QTYPES = {1: ('numerical', 0), 2: ('object_reference', 0),
          3: ('object_reference', 1), 4: ('instruction_following', 0),
          5: ('instruction_following', 1)}


def load_question(questions_file, scene, q):
    data = json.load(open(questions_file))
    qtype, idx = QTYPES[q]
    for item in data:
        if item['scene'] == scene:
            return qtype, item['questions'][qtype][idx]
    raise SystemExit(f'scene {scene} 不在 {questions_file}')


class Evaluator(Node):
    def __init__(self, question, qtype):
        super().__init__('question_evaluator')
        self.qtype = qtype
        self.result = {'question': question, 'numerical': None, 'marker': None,
                       'trajectory': []}
        self.t0 = time.time()
        self.t_answer = None
        self._last_traj_t = 0.0
        msg = String()
        msg.data = question
        self.pub = self.create_publisher(String, '/challenge_question', 5)
        self.create_timer(1.0, lambda: self.pub.publish(msg))
        self.create_subscription(Int32, '/numerical_response', self.on_num, 5)
        self.create_subscription(Marker, '/selected_object_marker',
                                 self.on_marker, 5)
        # sensor QoS:仿真的 /state_estimation 是 BEST_EFFORT 发布,
        # 默认 RELIABLE 订阅不兼容会一条都收不到(轨迹恒空)
        self.create_subscription(Odometry, '/state_estimation', self.on_odom,
                                 qos_profile_sensor_data)

    def on_num(self, msg):
        if self.result['numerical'] is None:
            self.result['numerical'] = int(msg.data)
            self.t_answer = time.time()
            self.get_logger().info(f'numerical_response: {msg.data}')

    def on_marker(self, msg):
        p, s = msg.pose.position, msg.scale
        self.result['marker'] = {'ns': msg.ns, 'id': msg.id,
                                 'center': [p.x, p.y, p.z],
                                 'size': [s.x, s.y, s.z]}
        if self.t_answer is None:
            self.t_answer = time.time()
        self.get_logger().info(f'marker: {msg.ns}')

    def on_odom(self, msg):
        t = time.time() - self.t0
        if t - self._last_traj_t >= 0.5:                 # 2Hz 降采样
            p = msg.pose.pose.position
            self.result['trajectory'].append(
                [round(t, 2), round(p.x, 3), round(p.y, 3)])
            self._last_traj_t = t

    def done(self, duration):
        t = time.time()
        if t - self.t0 >= duration:
            return True
        if self.qtype != 'instruction_following' and self.t_answer:
            return t - self.t_answer > 10.0              # 收到答案后再录 10s
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default=None)
    ap.add_argument('--q', type=int, choices=range(1, 6), default=1)
    ap.add_argument('--question', default=None, help='直接给题面(配 --qtype)')
    ap.add_argument('--qtype', default='numerical', choices=[v[0] for v in QTYPES.values()])
    ap.add_argument('--duration', type=float, default=600.0)
    ap.add_argument('--questions-file', default=os.path.join(
        os.path.dirname(__file__), '..', '..', 'questions', 'questions.json'))
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), 'out'))
    args = ap.parse_args()

    if args.question:
        qtype, question = args.qtype, args.question
        tag = f'custom_{int(time.time())}'
    else:
        if not args.scene:
            raise SystemExit('--scene 或 --question 必须给一个')
        qtype, question = load_question(args.questions_file, args.scene, args.q)
        tag = f'{args.scene}_q{args.q}'

    rclpy.init()
    node = Evaluator(question, qtype)
    node.get_logger().info(f'[{qtype}] {question}')
    try:
        while rclpy.ok() and not node.done(args.duration):
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.get_logger().info('中断,写盘退出')
    node.result.update(scene=args.scene or 'custom', q=args.q, qtype=qtype,
                       duration=time.time() - node.t0)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'{tag}.json')
    json.dump(node.result, open(out_path, 'w'), indent=1)
    print(f'→ {out_path}')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
