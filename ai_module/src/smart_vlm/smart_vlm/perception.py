"""CPU/GPU-adaptive open-vocabulary object detection."""

import threading
import numpy as np


class Detector:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.log = logger
        self.lock = threading.Lock()
        self.vocab = list(cfg['base_vocab'])
        try:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            self.device = 'cpu'
        model_name = cfg['model_gpu'] if self.device == 'cuda' else cfg['model_cpu']
        import os
        if os.path.sep in model_name and not os.path.exists(model_name):
            model_name = os.path.basename(model_name)
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(model_name)


        self.model.to(self.device)
        self._sync_clip_device(self.model)
        self.model.set_classes(self.vocab)
        self.log.info(f'Detector ready: {model_name} on {self.device}')

    @staticmethod
    def _sync_clip_device(yolo_model):
        """Keep Ultralytics' cached CLIP tokenizer device in sync.

        ``YOLOWorld.predict(device='cuda')`` moves the registered CLIP module
        to CUDA, but Ultralytics 8.4.90 leaves the adapter's plain ``device``
        attribute at ``cpu``.  A later ``set_classes`` then creates CPU token
        indices for CUDA weights and raises a device-mismatch RuntimeError.
        """
        world_model = getattr(yolo_model, 'model', None)
        clip_model = getattr(world_model, 'clip_model', None)
        clip_core = getattr(clip_model, 'model', None)
        if clip_model is None or clip_core is None:
            return
        try:
            clip_model.device = next(clip_core.parameters()).device
        except (AttributeError, StopIteration):
            return

    def extend_vocab(self, words):

        with self.lock:
            new = [w for w in words if w and w not in self.vocab]
            if new:
                self.vocab += new
                self._sync_clip_device(self.model)
                self.model.set_classes(self.vocab)
                self.log.info(f'Vocab extended: {new}')

    def set_task_vocab(self, words):


        with self.lock:
            vocab = []
            for word in words:
                value = str(word or '').strip()
                if value and value.lower() not in {v.lower() for v in vocab}:
                    vocab.append(value)
            if not vocab:
                vocab = list(self.cfg['base_vocab'])
            self.vocab = vocab
            self._sync_clip_device(self.model)
            self.model.set_classes(self.vocab)
            self.log.info(f'Task vocab ({len(vocab)}): {vocab}')

    def detect(self, img):

        with self.lock:
            res = self.model.predict(img, conf=self.cfg['conf_thresh'],
                                     iou=self.cfg.get('iou_thresh', 0.5),
                                     imgsz=self.cfg.get('image_size', 640),
                                     max_det=self.cfg.get('max_detections', 50),
                                     device=self.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            out.append({'label': self.vocab[int(b.cls)],
                        'conf': float(b.conf),
                        'box': tuple(float(v) for v in b.xyxy[0])})
        return out


_COLOR_TABLE = [
    ('black',  lambda h, s, v: v < 50),
    ('white',  lambda h, s, v: v > 200 and s < 40),
    ('gray',   lambda h, s, v: s < 40),
    ('red',    lambda h, s, v: h < 10 or h > 170),
    ('orange', lambda h, s, v: 10 <= h < 22),
    ('yellow', lambda h, s, v: 22 <= h < 35),
    ('green',  lambda h, s, v: 35 <= h < 78),
    ('teal',   lambda h, s, v: 78 <= h < 95),
    ('blue',   lambda h, s, v: 95 <= h < 130),
    ('purple', lambda h, s, v: 130 <= h < 150),
    ('pink',   lambda h, s, v: 150 <= h <= 170),
]


def dominant_color(bgr_crop):
    import cv2
    if bgr_crop.size == 0:
        return 'unknown'
    h_, w_ = bgr_crop.shape[:2]
    center = bgr_crop[h_ // 4: 3 * h_ // 4, w_ // 4: 3 * w_ // 4]
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    med = np.median(hsv, axis=0)
    for name, fn in _COLOR_TABLE:
        if fn(*med):
            return name
    return 'brown'
