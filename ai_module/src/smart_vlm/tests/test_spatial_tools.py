import numpy as np

from smart_vlm import spatial_tools as st
from smart_vlm.semantic_map import MapObject


def mk(oid, center, size=(0.5, 0.5, 0.5), label='obj'):
    return MapObject(oid, label, 'gray', np.asarray(center, float),
                     np.asarray(size, float))


def test_closest_farthest():
    a, b, anchor = mk(0, (1, 0, 0)), mk(1, (5, 0, 0)), mk(2, (0, 0, 0))
    assert st.closest([a, b], anchor).oid == 0
    assert st.farthest([a, b], anchor).oid == 1


def test_between():
    a, b = mk(0, (0, 0, 0)), mk(1, (4, 0, 0))
    assert st.between(mk(2, (2, 0.3, 0)), a, b)
    assert not st.between(mk(3, (2, 2.0, 0)), a, b)      # 偏离走廊
    assert not st.between(mk(4, (0.2, 0, 0)), a, b)      # 太靠端点
    assert not st.between(mk(5, (2, 0, 0)), a, a)        # 重合锚点


def test_on_top_and_above():
    table = mk(0, (0, 0, 0.4), size=(1.0, 1.0, 0.8))     # 顶面 0.8
    cup = mk(1, (0.1, 0.1, 0.9), size=(0.1, 0.1, 0.2))   # 底面 0.8
    high = mk(2, (0, 0, 2.0), size=(0.2, 0.2, 0.2))
    off = mk(3, (2.0, 0, 0.9), size=(0.1, 0.1, 0.2))
    assert st.on_top(cup, table)
    assert not st.on_top(high, table)                    # 悬空不接触
    assert not st.on_top(off, table)                     # 水平错位
    assert st.above(high, table)
    assert st.below(table, high)


def test_apply_relation():
    anchor = mk(9, (0, 0, 0))
    a, b = mk(0, (1, 0, 0)), mk(1, (5, 0, 0))
    assert [o.oid for o in st.apply_relation('closest', [a, b], [anchor])] == [0]
    assert [o.oid for o in st.apply_relation('farthest', [a, b], [anchor])] == [1]
    assert st.apply_relation('none', [a, b], []) == [a, b]
    assert st.apply_relation('near', [a, b], [anchor]) == [a]
    assert st.apply_relation('whatever', [a, b], [anchor]) == [a, b]
    assert st.apply_relation('closest', [], [anchor]) == []


def test_relation_table_mentions_ids():
    a, anchor = mk(0, (1, 0, 0)), mk(9, (0, 0, 0))
    txt = st.relation_table([a], [anchor])
    assert 'obj0' in txt and 'obj9' in txt and 'dist' in txt
