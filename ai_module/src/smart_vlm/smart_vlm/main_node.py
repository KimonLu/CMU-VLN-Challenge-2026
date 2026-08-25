"""ROS 2 state machine for perception, exploration, and challenge responses."""

import threading
import time
import traceback

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

import yaml

from .projection import PanoProjector
from .perception import Detector
from .semantic_map import SemanticMap
from .exploration import (GridMap, Explorer, bootstrap_waypoints, decimate,
                          decimate_final_approach, line_waypoints)
from .llm_client import LLMClient
from .answering import QuestionParser, Answerer
from .pose_buffer import PoseBuffer, TimedValueBuffer, stamp_to_sec, keyframe_due


def pc2_to_xyz_intensity(msg):

    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0)
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    off = {f.name: f.offset for f in msg.fields}
    xyz = np.stack([data[:, off[k]:off[k] + 4].copy().view(np.float32).ravel()
                    for k in ('x', 'y', 'z')], axis=1)
    inten = (data[:, off['intensity']:off['intensity'] + 4].copy()
             .view(np.float32).ravel()) if 'intensity' in off else np.zeros(n)
    ok = np.isfinite(xyz).all(axis=1)
    return xyz[ok], inten[ok]


def img_to_bgr(msg):
    import cv2
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    enc = msg.encoding.lower()
    if enc in ('rgb8',):
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr  # bgr8


class SmartVLM(Node):
    def __init__(self):
        super().__init__('smart_vlm')
        self.declare_parameter('config_path', '')
        with open(self.get_parameter('config_path').value) as f:
            self.cfg = yaml.safe_load(f)

        cg = ReentrantCallbackGroup()
        self.create_subscription(String, '/challenge_question',
                                 self.on_question, 5, callback_group=cg)
        self.create_subscription(Odometry, '/state_estimation',
                                 self.on_odom, 10, callback_group=cg)
        self.create_subscription(Image, '/camera/image',
                                 self.on_image, 2, callback_group=cg)
        self.create_subscription(PointCloud2, '/registered_scan',
                                 self.on_scan, 2, callback_group=cg)
        self.create_subscription(PointCloud2, '/terrain_map_ext',
                                 self.on_terrain, 2, callback_group=cg)

        self.pub_wp = self.create_publisher(Pose2D, '/way_point_with_heading', 5)
        self.pub_marker = self.create_publisher(Marker, '/selected_object_marker', 5)
        self.pub_num = self.create_publisher(Int32, '/numerical_response', 5)


        self.pose = None


        self.pose_buf = PoseBuffer(maxlen=4096)
        self.scan_history = TimedValueBuffer(maxlen=128, max_gap_s=0.35)
        self.question = None
        self.scan_buf = np.zeros((0, 3))
        self.last_kf_xy = None
        self.last_kf_yaw = None
        self.kf_queue = []
        self.kf_lock = threading.Lock()


        self.proj = PanoProjector(self.cfg['projection'])
        self.gm = GridMap(self.cfg['mapping']['grid_res_m'])
        self.smap = SemanticMap({**self.cfg['mapping'],
                                 'min_lidar_pts': self.cfg['perception']['min_lidar_pts']},
                                self.proj, self.get_logger())
        self.explorer = Explorer(self.gm, self.cfg['exploration'], self.get_logger())
        self.llm = LLMClient(self.cfg['llm'], self.get_logger())
        self.parser = QuestionParser(self.llm, self.get_logger())
        self.answerer = Answerer(self.smap, self.gm, self.llm, self.get_logger())
        self.detector = None

        self.t_question = None
        self.answered = False

        threading.Thread(target=self.mission_thread, daemon=True).start()
        threading.Thread(target=self.perception_thread, daemon=True).start()


    def on_question(self, msg):
        if self.question is None:
            self.question = msg.data
            self.t_question = time.time()
            self.get_logger().info(f'QUESTION: {msg.data}')

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pose = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        self.pose = pose
        self.pose_buf.push(stamp_to_sec(msg.header.stamp), pose)

    def on_scan(self, msg):


        if self.pose is None:
            return
        xyz, _ = pc2_to_xyz_intensity(msg)
        r = np.linalg.norm(xyz[:, :2] - np.array(self.pose[:2]), axis=1)
        xyz = xyz[r < self.cfg['mapping'].get('max_point_range_m', 40.0)]
        self.scan_buf = xyz
        self.scan_history.push(stamp_to_sec(msg.header.stamp), xyz)
        zr = self.cfg['mapping']['obstacle_z_range']
        m = (xyz[:, 2] > zr[0]) & (xyz[:, 2] < zr[1])
        self.gm.add_obstacles(xyz[m][:, :2])

    def on_terrain(self, msg):
        if self.pose is None:
            return
        xyz, inten = pc2_to_xyz_intensity(msg)
        r = np.linalg.norm(xyz[:, :2] - np.array(self.pose[:2]), axis=1)
        m = r < self.cfg['mapping'].get('max_point_range_m', 40.0)
        self.gm.update_terrain(xyz[m], inten[m],
                               self.cfg['mapping']['terrain_obstacle_intensity'])

    def on_image(self, msg):
        if self.detector is None:
            return
        image_t = stamp_to_sec(msg.header.stamp)
        pose = self.pose_buf.query(image_t)
        if pose is None:
            return
        scan = self.scan_history.query(image_t)
        if scan is None:

            scan = np.zeros((0, 3))
        due, xy, yaw = keyframe_due(pose, self.last_kf_xy, self.last_kf_yaw,
                                    self.cfg['perception']['keyframe_trans_m'],
                                    self.cfg['perception']['keyframe_rot_deg'])
        if not due:
            return
        self.last_kf_xy, self.last_kf_yaw = xy, yaw
        with self.kf_lock:
            self.kf_queue = [(img_to_bgr(msg), scan.copy(), pose)]


    def perception_thread(self):
        while rclpy.ok():
            with self.kf_lock:
                item = self.kf_queue.pop() if self.kf_queue else None
            if item is None:
                time.sleep(0.05)
                continue
            try:
                pano, scan, pose = item
                dets = []
                for view, yaw in self.proj.make_views(pano):
                    for d in self.detector.detect(view):
                        box = self.proj.view_box_to_pano_box(d['box'], yaw)
                        dets.append({**d, 'box': box})
                self.smap.integrate(pano, dets, scan, pose)
            except Exception:
                self.get_logger().error(traceback.format_exc())


    def mission_thread(self):
        try:
            # BOOT
            self.detector = Detector(self.cfg['perception'], self.get_logger())
            self.llm.health_check()
            while self.question is None and rclpy.ok():
                time.sleep(0.1)
            T = self.cfg['timing']
            t0 = self.t_question
            deadline = lambda key: t0 + T[key]
            watchdog = threading.Timer(T['hard_deadline'],
                                       self.force_answer)
            watchdog.daemon = True
            watchdog.start()


            parsed = self._with_deadline(
                lambda: self.parser.parse(self.question), T['parse_deadline'])
            if parsed is None:
                parsed = self.parser._regex_fallback(self.question)
            qtype = parsed.get('type', 'instruction_following')
            self.get_logger().info(f'PARSED: {parsed}')
            task_vocab = list(parsed.get('detection_vocab', []))
            task_vocab += list(parsed.get('target_nouns', []))
            for con in parsed.get('constraints', []):
                task_vocab += [con.get('target', '')]
                task_vocab += list(con.get('anchors') or [])
            self.detector.set_task_vocab(task_vocab)
            self.parsed = parsed

            # EXPLORE
            key = {'numerical': 'explore_deadline_numerical',
                   'object_reference': 'explore_deadline_objref',
                   'instruction_following': 'explore_deadline_instr'}[qtype]
            self.explore_until(deadline(key), parsed, qtype)

            self.get_logger().info(
                'MAP after explore:\n'
                + self.smap.scene_text(parsed.get('target_nouns')))


            if qtype == 'numerical':
                n = self._with_deadline(
                    lambda: self.answerer.answer_numerical(self.question, parsed),
                    T['answer_deadline'])
                if n is None:
                    subj = parsed.get('count_subject') or ''
                    n = max(len(self.smap.by_label(subj.split()[-1]))
                            if subj else 2, 1)
                self.publish_num(n)
            elif qtype == 'object_reference':
                obj = self._with_deadline(
                    lambda: self.answerer.answer_object_reference(
                        self.question, parsed), T['answer_deadline'])
                if obj is None:
                    objs = self.smap.confirmed() or self.smap.objects
                    obj = objs[0] if objs else None
                if obj:
                    self.publish_marker(obj)
                    self.goto((obj.center[0], obj.center[1]))
            else:
                plan = self._with_deadline(
                    lambda: self.answerer.plan_instruction(
                        self.question, parsed, self.pose[:2],
                        self.cfg['exploration']['waypoint_step_m'],
                        self.cfg['exploration'].get(
                            'final_waypoint_step_m', 1.0),
                        self.cfg['exploration'].get(
                            'final_approach_radius_m', 3.0),
                        return_meta=True,
                        approach_cfg=self.cfg['exploration'],
                        return_alternatives=True),
                    T['answer_deadline'])
                wps = plan[0] if plan else []
                penalty_zones = plan[1] if plan else []
                final_start_idx = plan[2] if plan and len(plan) > 2 else None
                final_alternatives = plan[3] if plan and len(plan) > 3 else []
                if not wps:
                    cons = [c for c in parsed.get('constraints', [])
                            if c.get('action') in ('goto', 'stop_at', 'pass_near')]
                    if cons:
                        o = self.answerer.resolve(cons[-1].get('target', ''), parsed,
                                                  constraint=cons[-1],
                                                  fallback_any=False)
                        if o:
                            goal = self.gm.nearest_free(
                                o.center[:2], max_r=2.0)
                            path = self.gm.astar(self.pose[:2], goal)
                            if path:
                                wps, final_start_idx = \
                                    decimate_final_approach(
                                        path,
                                        self.cfg['exploration'][
                                            'waypoint_step_m'],
                                        self.cfg['exploration'].get(
                                            'final_waypoint_step_m', 1.0),
                                        self.cfg['exploration'].get(
                                            'final_approach_radius_m', 3.0),
                                        return_start=True)
                            else:
                                wps = line_waypoints(
                                    self.pose[:2], goal,
                                    self.cfg['exploration'].get(
                                        'final_waypoint_step_m', 1.0))
                                final_start_idx = 0
                self.get_logger().info(f'PLAN: {len(wps)} waypoints')
                self.follow_waypoints(
                    wps, t0 + T['hard_deadline'] - 5,
                    penalty_zones=penalty_zones,
                    final_start_idx=final_start_idx,
                    final_alternatives=final_alternatives)
            self.answered = True
            watchdog.cancel()
            self.get_logger().info('DONE')
        except Exception:
            self.get_logger().error(traceback.format_exc())
            self.force_answer()

    @staticmethod
    def _with_deadline(fn, timeout):

        box = {}

        def run():
            try:
                box['out'] = fn()
            except Exception:
                box['err'] = traceback.format_exc()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout)
        if 'err' in box:
            raise RuntimeError(box['err'])
        return box.get('out')


    def explore_until(self, t_end, parsed, qtype):
        cfg = self.cfg['exploration']


        if self.pose:
            initial = bootstrap_waypoints(self.pose[:2], cfg)
            if not initial:
                self.get_logger().info(
                    'Bootstrap translation sweep skipped (360 panorama)')
            for waypoint in initial:
                self.goto(waypoint, timeout=8, reach=0.4)
        while time.time() < t_end and rclpy.ok():
            if qtype != 'numerical' and self._early_stop_ok(parsed, qtype):
                self.get_logger().info('Targets found, stop exploring early')
                return
            goal = self.explorer.next_goal(np.array(self.pose[:2]))
            if goal is None:
                self.get_logger().info('No frontier left')
                return

            path = self.gm.astar(np.array(self.pose[:2]), goal)
            wps = decimate(path, cfg['waypoint_step_m']) if path else [tuple(goal)]
            reached = True
            for wp in wps:
                if time.time() >= t_end:
                    return
                if not self.goto(wp, timeout=min(25, max(1, t_end - time.time())),
                                 stuck_check=True):
                    reached = False
                    break
            if not reached:
                self.explorer.give_up_current()
            else:
                self.explorer.complete_current()

    def targets_found(self, parsed):
        nouns = parsed.get('target_nouns', [])
        if not nouns:
            return False
        return all(self.smap.by_label(n.split()[-1]) for n in nouns)

    def _early_stop_ok(self, parsed, qtype):


        if qtype == 'instruction_following':
            finals = [c for c in parsed.get('constraints', [])
                      if c.get('action') in ('goto', 'stop_at', 'pass_near')]
            if finals:
                last = finals[-1]
                if last.get('relation') not in (None, '', 'none'):
                    return False
                return self.answerer.can_ground(
                    last.get('target', ''), parsed, last)
        if qtype == 'object_reference':


            required = list(parsed.get('target_nouns') or [])
            for con in parsed.get('constraints', []):
                required += list(con.get('anchors') or [])
            if not required or not all(self.answerer._label_objs(p)
                                       for p in required if p):
                return False
            target = parsed.get('target_nouns', [''])[0]
            relevant = next((c for c in parsed.get('constraints', [])
                             if target and target.lower() in
                             str(c.get('target', '')).lower()), None)
            return self.answerer.can_ground(target, parsed, relevant)
        return self.targets_found(parsed)


    def goto(self, xy, timeout=25, stuck_check=False, reach=None):
        msg = Pose2D()
        msg.x, msg.y, msg.theta = float(xy[0]), float(xy[1]), 0.0
        self.pub_wp.publish(msg)
        t0 = time.time()
        cfg = self.cfg['exploration']
        if reach is None:
            reach = cfg['reach_dist_m']
        start_p = np.asarray(self.pose[:2], dtype=float)
        start_dist = float(np.linalg.norm(start_p - np.asarray(xy)))
        last_p, last_t = start_p.copy(), t0
        progress_ref_dist = start_dist
        progress_t = t0
        self.get_logger().info(
            f'WAYPOINT start goal=({xy[0]:.2f},{xy[1]:.2f}) '
            f'dist={start_dist:.2f}m timeout={timeout:.1f}s reach={reach:.2f}m')
        while time.time() - t0 < timeout and rclpy.ok():
            p = np.array(self.pose[:2])
            if np.linalg.norm(p - np.array(xy)) < reach:
                self._last_goto_result = {
                    'status': 'reached', 'elapsed': time.time() - t0,
                    'moved': float(np.linalg.norm(p - start_p)),
                    'remaining': float(np.linalg.norm(p - np.asarray(xy)))}
                self.get_logger().info(
                    f"WAYPOINT reached elapsed={self._last_goto_result['elapsed']:.1f}s "
                    f"moved={self._last_goto_result['moved']:.2f}m "
                    f"remaining={self._last_goto_result['remaining']:.2f}m")
                return True
            if (stuck_check and cfg.get('goal_progress_enabled', False) and
                    time.time() - progress_t >
                    cfg.get('progress_timeout_s', cfg['stuck_timeout_s'])):
                dist = float(np.linalg.norm(p - np.asarray(xy)))
                improvement = progress_ref_dist - dist
                if improvement < cfg.get('progress_min_delta_m', 0.15):
                    self.get_logger().warn(
                        f'Waypoint no goal progress at ({p[0]:.2f},{p[1]:.2f}) '
                        f'toward ({xy[0]:.2f},{xy[1]:.2f}); '
                        f'improvement={improvement:.2f}m')
                    self._last_goto_result = {
                        'status': 'no_goal_progress',
                        'elapsed': time.time() - t0,
                        'moved': float(np.linalg.norm(p - start_p)),
                        'remaining': dist, 'improvement': improvement}
                    return False


                progress_ref_dist = min(progress_ref_dist, dist)
                progress_t = time.time()
            elif (stuck_check and not cfg.get('goal_progress_enabled', False)
                  and time.time() - last_t > cfg['stuck_timeout_s']):
                if np.linalg.norm(p - last_p) < cfg['stuck_min_move_m']:
                    self.get_logger().warn(
                        f'Waypoint stuck at ({p[0]:.2f},{p[1]:.2f}) '
                        f'toward ({xy[0]:.2f},{xy[1]:.2f})')
                    self._last_goto_result = {
                        'status': 'stuck', 'elapsed': time.time() - t0,
                        'moved': float(np.linalg.norm(p - start_p)),
                        'remaining': float(np.linalg.norm(p - np.asarray(xy)))}
                    return False
                last_p, last_t = p, time.time()
            time.sleep(0.1)
        p = np.array(self.pose[:2])
        self.get_logger().warn(
            f'Waypoint timeout at ({p[0]:.2f},{p[1]:.2f}) '
            f'toward ({xy[0]:.2f},{xy[1]:.2f})')
        self._last_goto_result = {
            'status': 'timeout', 'elapsed': time.time() - t0,
            'moved': float(np.linalg.norm(p - start_p)),
            'remaining': float(np.linalg.norm(p - np.asarray(xy)))}
        return False

    def _follow_final_path(self, wps, deadline, reach):

        per_wp = self.cfg['timing']['waypoint_timeout_s']
        for idx, wp in enumerate(wps):
            remaining = deadline - time.time()
            if remaining <= 0:
                self.get_logger().warn('Final approach budget exhausted')
                return False
            self.get_logger().info(
                f'FINAL waypoint {idx + 1}/{len(wps)} '
                f'goal=({wp[0]:.2f},{wp[1]:.2f}) '
                f'budget_left={remaining:.1f}s')
            self._last_failed_wp = None
            self._last_failed_from = np.asarray(self.pose[:2], dtype=float)
            if not self.goto(wp, timeout=min(per_wp, remaining),
                             stuck_check=True, reach=reach):
                self._last_failed_wp = np.asarray(wp, dtype=float)
                result = getattr(self, '_last_goto_result', {})
                self.get_logger().warn(
                    f"FINAL waypoint failed class={result.get('status', 'unknown')} "
                    f"moved={result.get('moved', float('nan')):.2f}m "
                    f"remaining={result.get('remaining', float('nan')):.2f}m")
                return False
        return True

    def _failure_penalty(self):


        wp = getattr(self, '_last_failed_wp', None)
        if wp is None:
            return None
        cur = np.asarray(self.pose[:2], dtype=float)
        vec = np.asarray(wp, dtype=float) - cur
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return None
        end = cur + vec / dist * min(0.8, dist)
        cfg = self.cfg['exploration']
        return (tuple(cur), tuple(end),
                float(cfg.get('failure_penalty_width_m', 0.5)),
                float(cfg.get('failure_penalty_cost', 25.0)))

    def _remember_failure(self, zones):
        if not self.cfg['exploration'].get('failure_memory_enabled', False):
            return
        zone = self._failure_penalty()
        if zone is not None:
            zones.append(zone)
            self.get_logger().warn(
                f'FINAL failure memory add zone={len(zones)} '
                f'from=({zone[0][0]:.2f},{zone[0][1]:.2f}) '
                f'to=({zone[1][0]:.2f},{zone[1][1]:.2f})')

    def follow_waypoints(self, wps, t_abort, penalty_zones=None,
                         final_start_idx=None, final_alternatives=None):

        if not wps:
            return False
        cfg = self.cfg['exploration']
        timing = self.cfg['timing']
        final_goal = np.asarray(wps[-1], dtype=float)
        radius = cfg.get('final_approach_radius_m', 3.0)
        final_step = cfg.get('final_waypoint_step_m', 1.0)
        final_reach = cfg.get('final_reach_dist_m', 0.5)
        base_replans = int(
            cfg.get('failure_memory_replan_attempts', 2)
            if cfg.get('failure_memory_enabled', False)
            else cfg.get('final_replan_attempts', 1))
        alternatives = []
        for point in (final_alternatives or []):
            p = np.asarray(point, dtype=float)
            if np.linalg.norm(p - final_goal) < 0.5:
                continue
            if all(np.linalg.norm(p - old) >= 0.5 for old in alternatives):
                alternatives.append(p)
        alternate_replans = min(
            len(alternatives), int(cfg.get('approach_replan_attempts', 2)))
        max_replans = max(base_replans, alternate_replans)
        dynamic_zones = list(penalty_zones or [])
        final_deadline = None


        final_start = (len(wps) - 1 if final_start_idx is None else
                       max(0, min(int(final_start_idx), len(wps) - 1)))

        for wp_idx, wp in enumerate(wps):
            if time.time() > t_abort:
                self.get_logger().warn('Time up, abort waypoint execution')
                break
            in_final = wp_idx >= final_start
            if not in_final:
                self.goto(
                    wp,
                    timeout=min(timing['waypoint_timeout_s'],
                                max(0.1, t_abort - time.time())))
                continue

            if final_deadline is None:
                final_deadline = min(
                    t_abort,
                    time.time() + timing.get('final_approach_timeout_s', 70))
                self.get_logger().info(
                    f'FINAL APPROACH start goal=({final_goal[0]:.2f},'
                    f'{final_goal[1]:.2f}) radius={radius:.1f}m '
                    f'budget={max(0.0, final_deadline-time.time()):.1f}s')

            if self._follow_final_path([wp], final_deadline, final_reach):
                continue

            self._remember_failure(dynamic_zones)


            for attempt in range(1, max_replans + 1):
                if time.time() >= final_deadline:
                    break
                cur = np.asarray(self.pose[:2], dtype=float)
                retry_goal = (alternatives.pop(0)
                              if alternatives else final_goal)
                path = self.gm.astar(
                    cur, retry_goal, penalty_zones=dynamic_zones)
                is_alternate = not np.allclose(retry_goal, final_goal)
                if is_alternate and not path:
                    self.get_logger().warn(
                        f'FINAL REPLAN {attempt}/{max_replans}: '
                        'alternate standoff became unreachable, skip')
                    continue
                mode = (('alternate standoff A*' if is_alternate else 'A*')
                        if path else 'direct fallback')
                retry_wps = ((decimate(path, final_step)
                              or [tuple(retry_goal)])
                             if path else [tuple(retry_goal)])
                self.get_logger().warn(
                    f'FINAL REPLAN {attempt}/{max_replans}: '
                    f'{len(retry_wps)} waypoints '
                    f'({mode}) penalty_zones={len(dynamic_zones)}')
                if self._follow_final_path(
                        retry_wps, final_deadline, final_reach):
                    dist = np.linalg.norm(
                        np.asarray(self.pose[:2]) - retry_goal)
                    self.get_logger().info(
                        f'FINAL APPROACH done after replan mode={mode} '
                        f'dist={dist:.2f}m')
                    return True
                self._remember_failure(dynamic_zones)
            dist = np.linalg.norm(np.asarray(self.pose[:2]) - final_goal)
            self.get_logger().warn(
                f'FINAL APPROACH failed dist={dist:.2f}m')
            return False

        dist = np.linalg.norm(np.asarray(self.pose[:2]) - final_goal)
        reached = dist < final_reach
        self.get_logger().info(
            f'FINAL APPROACH done reached={reached} dist={dist:.2f}m')
        return reached


    def publish_num(self, n):
        m = Int32()
        m.data = int(n)
        self.pub_num.publish(m)
        self.get_logger().info(f'ANSWER numerical: {n}')

    def publish_marker(self, obj):
        mk = Marker()
        mk.header.frame_id = 'map'
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = obj.label
        mk.id = int(obj.oid)
        mk.action = Marker.ADD
        mk.type = Marker.CUBE
        mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = \
            (float(v) for v in obj.center)
        mk.pose.orientation.w = 1.0
        mk.scale.x, mk.scale.y, mk.scale.z = (float(v) for v in obj.size)
        mk.color.a, mk.color.b = 0.5, 1.0
        self.pub_marker.publish(mk)
        self.get_logger().info(f'ANSWER marker: {obj.brief()}')


    def force_answer(self):
        if self.answered:
            return
        self.answered = True
        self.get_logger().warn('HARD DEADLINE - emitting best-effort answer')
        try:
            parsed = getattr(self, 'parsed', None) or \
                self.parser._regex_fallback(self.question or 'go')
            qtype = parsed.get('type')
            if qtype == 'numerical':
                subj = parsed.get('count_subject') or ''
                n = len(self.smap.by_label(subj.split()[-1])) if subj else 2
                self.publish_num(max(n, 1))
            elif qtype == 'object_reference':
                objs = self.smap.confirmed() or self.smap.objects
                if objs:
                    self.publish_marker(objs[0])
            else:
                cons = [c for c in parsed.get('constraints', [])
                        if c.get('action') in ('goto', 'stop_at')]
                if cons:
                    o = self.answerer.resolve(cons[-1].get('target', ''), parsed)
                    if o:
                        self.goto((o.center[0], o.center[1]), timeout=20)
        except Exception:
            self.get_logger().error(traceback.format_exc())


def main():
    rclpy.init()
    node = SmartVLM()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
