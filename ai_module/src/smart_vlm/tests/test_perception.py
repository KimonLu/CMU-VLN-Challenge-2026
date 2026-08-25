from types import SimpleNamespace

from smart_vlm.perception import Detector


class _Core:
    def parameters(self):
        yield SimpleNamespace(device='cuda:0')


def test_sync_clip_device_uses_actual_parameter_device():
    clip = SimpleNamespace(model=_Core(), device='cpu')
    yolo = SimpleNamespace(model=SimpleNamespace(clip_model=clip))

    Detector._sync_clip_device(yolo)

    assert clip.device == 'cuda:0'


def test_sync_clip_device_accepts_model_without_cached_clip():
    Detector._sync_clip_device(SimpleNamespace(model=SimpleNamespace()))


def test_set_task_vocab_replaces_irrelevant_base_classes():
    calls = []
    d = Detector.__new__(Detector)
    d.lock = __import__('threading').Lock()
    d.cfg = {'base_vocab': ['sofa', 'chair', 'table']}
    d.log = SimpleNamespace(info=lambda *args: None)
    d.model = SimpleNamespace(set_classes=lambda words: calls.append(list(words)),
                              model=SimpleNamespace())
    d.vocab = list(d.cfg['base_vocab'])
    d.set_task_vocab(['bowl', 'table', 'BOWL', ''])
    assert d.vocab == ['bowl', 'table']
    assert calls[-1] == ['bowl', 'table']
