"""pytest 公共配置:无 ROS 环境下测试 smart_vlm 各模块(main_node 除外)。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class FakeLogger:
    def info(self, *a, **k):
        pass

    warn = info
    error = info


@pytest.fixture
def logger():
    return FakeLogger()


PROJ_CFG = {'img_w': 1920, 'img_h': 640, 'vfov_deg': 120,
            'azimuth_flip': False, 'elevation_flip': False}


@pytest.fixture
def proj_cfg():
    return dict(PROJ_CFG)
