#!/usr/bin/env python3
"""Headless challenge-question publisher and response recorder."""



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
    raise SystemExit(f'scene {scene} is not present in {questions_file}')


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
        if t - self._last_traj_t >= 0.5:
            p = msg.pose.pose.position
            self.result['trajectory'].append(
                [round(t, 2), round(p.x, 3), round(p.y, 3)])
            self._last_traj_t = t

    def done(self, duration):
        t = time.time()
        if t - self.t0 >= duration:
            return True
        if self.qtype != 'instruction_following' and self.t_answer:
            return t - self.t_answer > 10.0
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default=None)
    ap.add_argument('--q', type=int, choices=range(1, 6), default=1)
    ap.add_argument('--question', default=None, help='question text, used with --qtype')
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
            raise SystemExit('provide either --scene or --question')
        qtype, question = load_question(args.questions_file, args.scene, args.q)
        tag = f'{args.scene}_q{args.q}'

    rclpy.init()
    node = Evaluator(question, qtype)
    node.get_logger().info(f'[{qtype}] {question}')
    try:
        while rclpy.ok() and not node.done(args.duration):
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.get_logger().info('interrupted; writing the result before exit')
    node.result.update(scene=args.scene or 'custom', q=args.q, qtype=qtype,
                       duration=time.time() - node.t0)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'{tag}.json')
    json.dump(node.result, open(out_path, 'w'), indent=1)
    print(f'-> {out_path}')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
