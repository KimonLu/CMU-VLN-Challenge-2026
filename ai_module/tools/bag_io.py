"""Shared rosbag2 and live-topic data acquisition helpers."""

import numpy as np


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
    if msg.encoding.lower() == 'rgb8':
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def odom_to_pose(msg):
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    return (p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def stamp_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def read_bag(path, topics):

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, raw, t = reader.read_next()
        if topic in topics:
            yield topic, deserialize_message(raw, get_message(types[topic])), t * 1e-9


def collect_synced_samples(source_iter, n_samples=10, min_gap_s=1.0,
                           odom_max_dt=0.15):


    samples = []
    odoms = []                                    # [(t, pose)]
    scan = None
    last_take = -1e9
    for topic, msg, _ in source_iter:
        if topic == '/state_estimation':
            odoms.append((stamp_sec(msg), odom_to_pose(msg)))
            if len(odoms) > 2000:
                del odoms[:1000]
        elif topic == '/registered_scan':
            scan, _ = pc2_to_xyz_intensity(msg)
        elif topic == '/camera/image':
            t_img = stamp_sec(msg)
            if scan is None or not odoms or t_img - last_take < min_gap_s:
                continue
            ts = np.array([o[0] for o in odoms])
            k = int(np.argmin(np.abs(ts - t_img)))
            if abs(ts[k] - t_img) > odom_max_dt:
                continue
            samples.append((img_to_bgr(msg), scan.copy(), odoms[k][1]))
            last_take = t_img
            if len(samples) >= n_samples:
                break
    return samples


def live_source(topics_types, duration_s):

    import queue
    import time
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node('tool_collector')
    q = queue.Queue()
    for topic, mtype in topics_types.items():
        node.create_subscription(
            mtype, topic,
            lambda msg, t=topic: q.put((t, msg, time.time())), 10)
    t0 = time.time()
    try:
        while time.time() - t0 < duration_s:
            rclpy.spin_once(node, timeout_sec=0.1)
            while not q.empty():
                yield q.get()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
