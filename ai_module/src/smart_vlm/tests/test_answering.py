import numpy as np
import pytest

from smart_vlm.answering import Answerer, QuestionParser
from smart_vlm.exploration import FREE, GridMap
from smart_vlm.projection import PanoProjector
from smart_vlm.semantic_map import MapObject, SemanticMap
from conftest import FakeLogger, PROJ_CFG


class FakeLLM:
    def __init__(self, out=None):
        self.out = out
        self.calls = []

    def ask(self, prompt, images=None, expect_json=True):
        self.calls.append(prompt)
        self.images = getattr(self, 'images', [])
        self.images.append(images)
        return self.out(prompt) if callable(self.out) else self.out


def mk(oid, label, center, size=(0.5, 0.5, 0.5), color='gray', n_obs=2):
    return MapObject(oid, label, color, np.asarray(center, float),
                     np.asarray(size, float), n_obs=n_obs)


def make_answerer(objects, llm=None):
    smap = SemanticMap({'merge_dist_m': 0.5, 'confirm_obs': 2,
                        'min_lidar_pts': 8},
                       PanoProjector(PROJ_CFG), FakeLogger())
    smap.objects = list(objects)
    gm = GridMap(res=0.2)
    gm.grid[:, :] = FREE
    llm = llm or FakeLLM(None)
    return Answerer(smap, gm, llm, FakeLogger()), llm


# ---------- QuestionParser ----------

def test_normalize_fills_defaults():
    out = QuestionParser._normalize(
        {'type': 'numerical', 'constraints': [{'anchors': None}]})
    assert out['target_nouns'] == [] and out['detection_vocab'] == []
    c = out['constraints'][0]
    assert c['anchors'] == [] and c['action'] == 'goto' and c['relation'] == 'none'
    assert out['count_subject'] is None and out['color_filters'] == {}


def test_normalize_rejects_bad_type():
    assert QuestionParser._normalize({'type': 'bogus'}) is None
    assert QuestionParser._normalize(None) is None
    assert QuestionParser._normalize(['not', 'dict']) is None


def test_parse_falls_back_on_garbage_llm(logger):
    p = QuestionParser(FakeLLM({'type': 'bogus'}), logger)
    out = p.parse('Go to the sofa and stop')
    assert out['type'] == 'instruction_following'    # 正则回退
    assert 'sofa' in out['target_nouns']


def test_regex_fallback_types(logger):
    p = QuestionParser(FakeLLM(None), logger)
    assert p.parse('How many chairs are there')['type'] == 'numerical'
    assert p.parse('Find the pillow on the sofa')['type'] == 'object_reference'
    assert p.parse('Take the path between columns')['type'] == \
        'instruction_following'


def test_regex_fallback_matches_plural_nouns(logger):
    """冒烟实测:兜底正则漏掉复数 → "chairs" 解析出空 count_subject。"""
    p = QuestionParser(FakeLLM(None), logger)
    out = p.parse('How many chairs are in the room?')
    assert out['count_subject'] == 'chair'
    assert out['target_nouns'] == ['chair']
    out = p.parse('Walk past the benches and stop')
    assert 'bench' in out['target_nouns']


def test_parse_type_forced_by_question_prefix(logger):
    """SJTU 实测:deepseek 把 "Find the X on Y" 错判成 instruction_following;
    比赛题面前缀固定 → 规则强制覆盖 type,槽位仍用 LLM 的。"""
    llm_out = {'type': 'instruction_following', 'target_nouns': ['vase'],
               'detection_vocab': ['vase'], 'count_subject': None,
               'color_filters': {},
               'constraints': [{'action': 'goto', 'target': 'vase',
                                'anchors': ['coffee table'], 'relation': 'on'}]}
    p = QuestionParser(FakeLLM(llm_out), logger)
    out = p.parse('Find the vase on the coffee table.')
    assert out['type'] == 'object_reference'
    assert out['constraints'][0]['anchors'] == ['coffee table']   # 槽位保留


def test_parse_type_forced_numerical(logger):
    llm_out = {'type': 'object_reference', 'target_nouns': ['chair'],
               'detection_vocab': [], 'count_subject': 'chair',
               'color_filters': {}, 'constraints': []}
    p = QuestionParser(FakeLLM(llm_out), logger)
    assert p.parse('How many chairs are there?')['type'] == 'numerical'


def test_parse_repairs_nested_support_reference(logger):
    llm_out = {'type': 'object_reference', 'target_nouns': ['lamp', 'nightstand'],
               'detection_vocab': ['lamp', 'nightstand'], 'constraints': [],
               'count_subject': None, 'color_filters': {}}
    out = QuestionParser(FakeLLM(llm_out), logger).parse(
        'Find the lamp on the nightstand that has the photo on it.')
    assert out['target_nouns'] == ['lamp']
    assert out['constraints'][0] == {
        'action': 'goto', 'target': 'lamp', 'anchors': ['nightstand'],
        'relation': 'on'}
    assert out['constraints'][1]['anchors'] == ['photo']
    assert 'photo' in out['detection_vocab']


# ---------- resolve ----------

def test_resolve_color_filter():
    ans, _ = make_answerer([mk(0, 'chair', (1, 0, 0), color='red'),
                            mk(1, 'chair', (2, 0, 0), color='blue')])
    parsed = {'color_filters': {'chair': 'red'}, 'constraints': []}
    assert ans.resolve('chair', parsed).oid == 0


def test_resolve_longest_suffix():
    ans, _ = make_answerer([mk(0, 'coffee table', (1, 0, 0)),
                            mk(1, 'sofa', (2, 0, 0))])
    parsed = {'color_filters': {}, 'constraints': []}
    assert ans.resolve('the small coffee table', parsed).oid == 0


def test_arbitrate_picks_llm_choice():
    llm = FakeLLM({'object_id': 1, 'confidence': 0.9})
    ans, _ = make_answerer([mk(0, 'chair', (1, 0, 0)),
                            mk(1, 'chair', (5, 0, 0))], llm)
    parsed = {'color_filters': {}, 'constraints': [], 'target_nouns': ['chair']}
    assert ans.resolve('chair', parsed).oid == 1


def test_resolve_uses_nested_qualified_anchor():
    """lamp on [nightstand with photo]：先用 photo 选 nightstand，再选其上的 lamp。"""
    objs = [
        mk(0, 'nightstand', (0, 0, 0.35), size=(1, 1, 0.7)),
        mk(1, 'nightstand', (4, 0, 0.35), size=(1, 1, 0.7)),
        mk(2, 'photo', (4, 0, 0.85), size=(0.2, 0.2, 0.2)),
        mk(3, 'lamp', (0, 0, 1.0), size=(0.3, 0.3, 0.6)),
        mk(4, 'lamp', (4, 0, 1.0), size=(0.3, 0.3, 0.6)),
    ]
    ans, _ = make_answerer(objs)
    cons = [
        {'action': 'goto', 'target': 'lamp', 'anchors': ['nightstand'],
         'relation': 'on'},
        {'action': 'stop_at', 'target': 'nightstand', 'anchors': ['photo'],
         'relation': 'with'},
    ]
    parsed = {'color_filters': {}, 'constraints': cons, 'target_nouns': ['lamp']}
    assert ans.resolve('lamp', parsed, constraint=cons[0]).oid == 4


def test_resolve_complex_closest_uses_support_then_final_anchor():
    """bowl on table closest screen：承载过滤后按最终锚点全局 argmin。"""
    objs = [
        mk(0, 'table', (0, 0, 0.4), size=(2, 2, 0.8)),
        mk(1, 'bowl', (-0.5, 0, 0.9), size=(0.2, 0.2, 0.2)),
        mk(2, 'bowl', (0.5, 0, 0.9), size=(0.2, 0.2, 0.2)),
        mk(3, 'folding screen', (3, 0, 1.0)),
    ]
    ans, _ = make_answerer(objs)
    con = {'action': 'goto', 'target': 'bowl',
           'anchors': ['table', 'folding screen'], 'relation': 'closest'}
    parsed = {'color_filters': {}, 'constraints': [con],
              'target_nouns': ['bowl']}
    assert ans.resolve('bowl', parsed, constraint=con).oid == 2


# ---------- numerical ----------

def test_answer_numerical_with_on_relation():
    objs = [mk(0, 'sofa', (0, 0, 0.4), size=(2, 1, 0.8)),
            mk(1, 'pillow', (0.3, 0.2, 0.9), size=(0.3, 0.3, 0.2)),
            mk(2, 'pillow', (-0.5, -0.2, 0.9), size=(0.3, 0.3, 0.2)),
            mk(3, 'pillow', (3.0, 3.0, 0.1), size=(0.3, 0.3, 0.2))]
    ans, llm = make_answerer(objs)
    parsed = {'type': 'numerical', 'count_subject': 'pillow',
              'target_nouns': ['pillow', 'sofa'], 'color_filters': {},
              'constraints': [{'action': 'goto', 'target': '',
                               'anchors': ['sofa'], 'relation': 'on'}]}
    assert ans.answer_numerical('How many pillows are on the sofa', parsed) == 2


def test_direct_visual_count_bypasses_detector_recall_limit():
    """最佳全景 VLM 直接列举；即使 3D 物体库只检出 1 个也可回答 2。"""
    llm = FakeLLM({'items': [{'where': 'left'}, {'where': 'right'}]})
    ans, _ = make_answerer([mk(0, 'pillow', (0, 0, 0))], llm)
    ans.smap._remember_view(np.full((640, 1920, 3), 120, np.uint8), [])
    parsed = {'type': 'numerical', 'count_subject': 'pillow',
              'target_nouns': ['pillow'], 'color_filters': {}, 'constraints': []}
    assert ans.answer_numerical('How many pillows?', parsed) == 2
    assert len(llm.images) == 1 and len(llm.images[0]) == 1
    assert llm.images[0][0].shape[:2] == (1280, 1280)
    assert 'overlap' in llm.calls[0]


def test_direct_visual_count_accepts_count_only_schema():
    llm = FakeLLM({'count': 6})
    ans, _ = make_answerer([], llm)
    ans.smap._remember_view(np.full((640, 1920, 3), 120, np.uint8), [])
    parsed = {'type': 'numerical', 'count_subject': 'chair',
              'target_nouns': ['chair'], 'color_filters': {}, 'constraints': []}
    assert ans.answer_numerical('How many chairs?', parsed) == 6


def test_count_check_sends_single_montage_image():
    """冒烟实测:智谱 glm-4v-flash 单请求只收一张图(错误码 1210),
    复核必须把 N 张裁剪图拼成 1 张发送。"""
    objs = [mk(i, 'chair', (i, 0, 0)) for i in range(3)]
    for o in objs:
        o.best_crop = np.full((60, 50, 3), 128, dtype=np.uint8)
    llm = FakeLLM({'valid_ids': [0, 2]})
    ans, _ = make_answerer(objs, llm)
    parsed = {'type': 'numerical', 'count_subject': 'chair',
              'target_nouns': ['chair'], 'color_filters': {}, 'constraints': []}
    n = ans.answer_numerical('How many chairs are in the room?', parsed)
    assert len(llm.images) == 1 and len(llm.images[0]) == 1   # 仅 1 张图
    img = llm.images[0][0]
    assert img.shape[1] > 3 * 50                              # 确为横向拼图
    assert n == 2                                             # id=1 被否决
    assert '[0, 1, 2]' in llm.calls[0]                        # prompt ids 与发图一致


def test_count_check_uncropped_candidates_not_penalized():
    """无裁剪图的候选没被 VLM 看过,不得因复核被扣数。"""
    objs = [mk(i, 'chair', (i, 0, 0)) for i in range(3)]
    objs[0].best_crop = np.full((60, 50, 3), 128, dtype=np.uint8)
    llm = FakeLLM({'valid_ids': [0]})
    ans, _ = make_answerer(objs, llm)
    parsed = {'type': 'numerical', 'count_subject': 'chair',
              'target_nouns': ['chair'], 'color_filters': {}, 'constraints': []}
    # 3 个候选,只有 id=0 有图且被确认 → 未看过的 1、2 保留,答案 3
    assert ans.answer_numerical('How many chairs?', parsed) == 3
    assert '[0]' in llm.calls[0]


def test_count_check_skips_tiny_crops():
    """SJTU 实测:<48px 的裁剪放大后连人都认不出,VLM 保守否决造成误杀 →
    过小的 crop 不送复核、不因复核扣数。"""
    objs = [mk(i, 'chair', (i, 0, 0)) for i in range(3)]
    objs[0].best_crop = np.full((60, 60, 3), 128, np.uint8)
    objs[1].best_crop = np.full((20, 20, 3), 128, np.uint8)    # 过小
    objs[2].best_crop = np.full((60, 60, 3), 128, np.uint8)
    llm = FakeLLM({'valid_ids': [0]})
    ans, _ = make_answerer(objs, llm)
    parsed = {'type': 'numerical', 'count_subject': 'chair',
              'target_nouns': ['chair'], 'color_filters': {}, 'constraints': []}
    n = ans.answer_numerical('How many chairs?', parsed)
    assert '[0, 2]' in llm.calls[0]               # id=1 未被送审
    assert n == 2                                 # id=2 被否决;id=1 不受影响


def test_montage_mixed_sizes():
    from smart_vlm.answering import montage
    crops = [np.zeros((80, 50, 3), np.uint8), np.zeros((30, 120, 3), np.uint8)]
    img = montage(crops, [3, 7], tile_h=64)
    assert img.ndim == 3 and img.shape[0] == 64 + 28          # 统一高度+标注条


# ---------- instruction following ----------

BASE_PARSED = {'color_filters': {}, 'target_nouns': []}


def test_plan_goto_reaches_target():
    ans, _ = make_answerer([mk(0, 'plant', (3, 0, 0.3))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'stop_at', 'target': 'plant',
                               'anchors': [], 'relation': 'none'}]}
    wps, zones = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert wps and zones == []
    assert np.linalg.norm(np.asarray(wps[-1]) - [3, 0]) < 1.35


def test_plan_pass_between_two_anchors():
    ans, _ = make_answerer([mk(0, 'column', (2, -1, 1)),
                            mk(1, 'column', (2, 1, 1))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'pass_between', 'target': '',
                               'anchors': ['column', 'column'],
                               'relation': 'between'}]}
    wps, _ = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert any(np.linalg.norm(np.asarray(w) - [2, 0]) < 0.1 for w in wps)
    assert len(wps) == 3                          # 入口/中点/出口


def test_plan_pass_between_then_reaches_target():
    """SJTU 实测:deepseek 把最终目的地塞进 pass_between.target 且不给单独
    goto → 穿过间隙后必须继续前往 target,否则永远到不了目的地。"""
    ans, _ = make_answerer([mk(0, 'column', (2, -1, 1)),
                            mk(1, 'column', (2, 1, 1)),
                            mk(2, 'plant', (6, 0, 0.3))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'pass_between', 'target': 'plant',
                               'anchors': ['column', 'column'],
                               'relation': 'between'}]}
    wps, _ = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert any(np.linalg.norm(np.asarray(w) - [2, 0]) < 0.1 for w in wps)
    assert np.linalg.norm(np.asarray(wps[-1]) - [6, 0]) < 1.5


def test_plan_pass_between_single_anchor_degrades():
    """P0 ⑥:anchors 只有 1 个时降级为 pass_near,不越界崩溃。"""
    ans, _ = make_answerer([mk(0, 'column', (2, 0, 1))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'pass_between', 'target': '',
                               'anchors': ['column'], 'relation': 'between'}]}
    wps, _ = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert wps
    assert np.linalg.norm(np.asarray(wps[-1]) - [2, 0]) < 1.5


def test_plan_pass_between_no_anchor_skipped():
    ans, _ = make_answerer([mk(0, 'sofa', (2, 0, 0.4))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'pass_between', 'target': '',
                               'anchors': [], 'relation': 'between'}]}
    wps, zones = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert wps == [] and zones == []


def test_plan_avoid_target_only():
    """avoid 的对象在 target 而非 anchors 时也要生效。"""
    ans, _ = make_answerer([mk(0, 'rug', (1, 1, 0.05), size=(1.5, 1.0, 0.1))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'avoid', 'target': 'rug',
                               'anchors': [], 'relation': 'none'}]}
    wps, zones = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert len(zones) == 1


def test_plan_avoid_none_anchors_no_crash():
    ans, _ = make_answerer([mk(0, 'sofa', (2, 0, 0.4))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'avoid', 'target': '',
                               'anchors': None, 'relation': 'none'}]}
    wps, zones = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert zones == []                            # 无可解析对象 → 无禁区,不崩


def test_plan_waypoint_step_param():
    ans, _ = make_answerer([mk(0, 'plant', (10, 0, 0.3))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'goto', 'target': 'plant',
                               'anchors': [], 'relation': 'none'}]}
    wps_fine, _ = ans.plan_instruction('q', parsed, (0.0, 0.0), step=1.0)
    wps_coarse, _ = ans.plan_instruction('q', parsed, (0.0, 0.0), step=4.0)
    assert len(wps_fine) > len(wps_coarse)        # P1 ⑨a:step 生效


def test_plan_densifies_only_final_constraint():
    """q5 修复只改最后接近段，不改前序约束的航点节奏。"""
    objs = [mk(0, 'lamp', (4, 0, 1)), mk(1, 'plant', (10, 0, 0.3))]
    parsed = {**BASE_PARSED,
              'constraints': [
                  {'action': 'pass_near', 'target': 'lamp',
                   'anchors': [], 'relation': 'none'},
                  {'action': 'stop_at', 'target': 'plant',
                   'anchors': [], 'relation': 'none'}]}
    coarse, _ = make_answerer(objs)[0].plan_instruction(
        'q', parsed, (0.0, 0.0), step=2.5)
    dense, _ = make_answerer(objs)[0].plan_instruction(
        'q', parsed, (0.0, 0.0), step=2.5,
        final_step=1.0, final_radius=3.0)
    # 前一个目标和最后一段的粗前缀不变，只在终点前增加航点。
    assert np.allclose(dense[:3], coarse[:3])
    assert np.allclose(dense[-1], coarse[-1])
    assert len(dense) > len(coarse)
    assert all(not np.allclose(a, b) for a, b in zip(dense, dense[1:]))


def test_plan_meta_keeps_pass_between_outside_final_retry_segment():
    """柜体离 between 航点很近时，末段重规划也不得跳过入口/中点/出口。"""
    ans, _ = make_answerer([
        mk(0, 'lamp', (1, 0, 1)),
        mk(1, 'sofa', (2, -1, 0.5)),
        mk(2, 'table', (2, 1, 0.5)),
        mk(3, 'cabinet', (3.5, 0, 0.5))])
    parsed = {**BASE_PARSED,
              'constraints': [
                  {'action': 'pass_near', 'target': 'lamp',
                   'anchors': [], 'relation': 'none'},
                  {'action': 'pass_between', 'target': '',
                   'anchors': ['sofa', 'table'], 'relation': 'between'},
                  {'action': 'stop_at', 'target': 'cabinet',
                   'anchors': [], 'relation': 'none'}]}
    wps, _, final_start = ans.plan_instruction(
        'q', parsed, (0.0, 0.0), step=2.5,
        final_step=1.0, final_radius=3.0, return_meta=True)
    mid_idx = next(i for i, w in enumerate(wps)
                   if np.linalg.norm(np.asarray(w) - [2.0, 0.0]) < 0.1)
    assert final_start > mid_idx + 1                 # 必须先执行 between 出口
    assert final_start < len(wps)


def test_plan_full_meta_returns_diverse_final_standoffs():
    ans, _ = make_answerer([mk(0, 'cabinet', (5, 0, 0.5))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'stop_at', 'target': 'cabinet',
                               'anchors': [], 'relation': 'none'}]}
    wps, _, final_start, alternatives = ans.plan_instruction(
        'q', parsed, (0.0, 0.0), final_step=1.0, final_radius=3.0,
        return_meta=True, return_alternatives=True)
    assert wps and final_start is not None and alternatives
    final = np.asarray(wps[-1])
    assert all(np.linalg.norm(np.asarray(p) - final) >= 0.75
               for p in alternatives)
    assert all(np.linalg.norm(np.asarray(a) - np.asarray(b)) >= 0.75
               for i, a in enumerate(alternatives)
               for b in alternatives[i + 1:])


def test_plan_single_point_astar_path_does_not_crash():
    ans, _ = make_answerer([mk(0, 'plant', (0, 0, 0.3))])
    ans.gm.astar = lambda *args, **kwargs: [(0.1, 0.1)]
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'stop_at', 'target': 'plant',
                               'anchors': [], 'relation': 'none'}]}
    wps, _, final_start = ans.plan_instruction(
        'q', parsed, (0.0, 0.0), final_step=1.0,
        final_radius=3.0, return_meta=True)
    assert wps == [(0.1, 0.1)] and final_start == 0


def test_plan_final_astar_miss_uses_dense_local_waypoints():
    """q5 密集家具区：A* 暂时无路也不得一步直冲终点。"""
    ans, _ = make_answerer([mk(0, 'cabinet', (5, 0, 0.5))])
    ans.gm.astar = lambda *args, **kwargs: None
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'stop_at', 'target': 'cabinet',
                               'anchors': [], 'relation': 'none'}]}
    wps, _, final_start = ans.plan_instruction(
        'q', parsed, (0.0, 0.0), final_step=1.0,
        final_radius=3.0, return_meta=True)
    assert final_start == 0 and len(wps) >= 5
    assert np.linalg.norm(np.asarray(wps[-1]) - [5, 0]) < 1.5
    assert all(np.linalg.norm(np.asarray(b) - np.asarray(a)) <= 1.01
               for a, b in zip([(0.0, 0.0)] + wps, wps))


def test_answer_object_reference_returns_obj():
    ans, _ = make_answerer([mk(0, 'vase', (1, 1, 0.5))])
    parsed = {'type': 'object_reference', 'target_nouns': ['vase'],
              'color_filters': {}, 'constraints': []}
    assert ans.answer_object_reference('Find the vase', parsed).oid == 0


# ---------- 2026-07-09 优化:relation-aware resolve + 探索早停 grounding + 安全回退 ----------
# 背景:q5 停在原点附近(离真 cabinet 6.7m)。根因=①探索见到任一同名物体就早停
# ②目标类别缺失时 resolve 抓全库随便一个当终点。下列用例锁定修复后的行为。

def _cab_pic():
    """cabinet0 上方有 picture(关联);cabinet1 在远处孤立(无关联 picture)。"""
    cab0 = mk(0, 'cabinet', (0, 0, 0.5), size=(1.0, 0.5, 1.0))   # 顶=1.0
    pic = mk(1, 'picture', (0, 0, 1.7), size=(0.6, 0.05, 0.4))   # 底=1.5,在其上方
    cab1 = mk(2, 'cabinet', (5, 5, 0.5), size=(1.0, 0.5, 1.0))
    return [cab0, pic, cab1]


def test_resolve_relation_picks_associated_object():
    """"the cabinet with a picture above it":用约束关系确定性收窄,不靠 LLM。"""
    ans, llm = make_answerer(_cab_pic())         # FakeLLM(None):若误走 LLM 会拿到 None
    con = {'action': 'stop_at', 'target': 'cabinet',
           'anchors': ['picture'], 'relation': 'above'}
    parsed = {**BASE_PARSED, 'constraints': [con]}
    obj = ans.resolve('cabinet', parsed, constraint=con)
    assert obj is not None and obj.oid == 0      # 选带 picture 的那个
    assert llm.calls == []                       # 全程未调用 LLM


def test_resolve_wall_picture_offset_still_associates():
    """墙上画与靠墙柜有 ~0.4m 水平偏移(q5 真实几何):above() 用尺寸判太紧会漏,
    _stacked 水平容差应正确关联。"""
    cab = mk(0, 'cabinet', (-2.2, -6.4, 0.4), size=(0.5, 1.3, 0.7))
    pic = mk(1, 'picture', (-2.6, -6.6, 1.4), size=(0.1, 0.8, 0.7))
    decoy = mk(2, 'cabinet', (2.0, 0.0, 0.5), size=(0.5, 0.5, 1.0))
    ans, llm = make_answerer([cab, pic, decoy])
    con = {'action': 'stop_at', 'target': 'cabinet',
           'anchors': ['picture'], 'relation': 'above'}
    obj = ans.resolve('cabinet', {**BASE_PARSED, 'constraints': [con]},
                      constraint=con)
    assert obj is not None and obj.oid == 0
    assert llm.calls == []


def test_resolve_relation_anchor_beyond_first_three():
    """真锚点排在多个干扰 picture 之后:anchor 不应被截断,否则漏配真柜(q5 根因)。"""
    objs = [mk(10 + i, 'picture', (-2.6, -1.0 - i, 1.5), size=(0.1, 0.4, 0.6))
            for i in range(5)]                 # 5 个高处干扰 picture(不在任何柜上方)
    cab = mk(0, 'cabinet', (3.0, 3.0, 0.4), size=(0.5, 1.0, 0.7))
    truepic = mk(1, 'picture', (3.1, 3.0, 1.4), size=(0.1, 0.8, 0.7))
    ans, llm = make_answerer(objs + [cab, truepic])
    con = {'action': 'stop_at', 'target': 'cabinet',
           'anchors': ['picture'], 'relation': 'above'}
    obj = ans.resolve('cabinet', {**BASE_PARSED, 'constraints': [con]},
                      constraint=con)
    assert obj is not None and obj.oid == 0
    assert llm.calls == []


def test_resolve_relation_confidence_tiebreak():
    """多个柜都满足"上方有画":按观测置信度 n_obs 择优,不掷给 LLM。"""
    cabA = mk(0, 'cabinet', (0, 0, 0.4), size=(0.5, 1.0, 0.7), n_obs=15)
    picA = mk(1, 'picture', (0.2, 0, 1.4), size=(0.1, 0.6, 0.7))
    cabB = mk(2, 'cabinet', (4, 0, 0.4), size=(0.5, 1.0, 0.7), n_obs=3)
    picB = mk(3, 'picture', (4.1, 0, 1.4), size=(0.1, 0.6, 0.7))
    ans, llm = make_answerer([cabA, picA, cabB, picB])
    con = {'action': 'stop_at', 'target': 'cabinet',
           'anchors': ['picture'], 'relation': 'above'}
    obj = ans.resolve('cabinet', {**BASE_PARSED, 'constraints': [con]},
                      constraint=con)
    assert obj is not None and obj.oid == 0      # 见 15 次者胜
    assert llm.calls == []


def test_resolve_final_target_absent_returns_none():
    """最终目标类别缺失 + fallback_any=False → 返回 None,不乱抓近处物。"""
    ans, _ = make_answerer([mk(0, 'sofa', (1, 0, 0.4)), mk(1, 'lamp', (2, 0, 1))])
    parsed = {**BASE_PARSED, 'constraints': []}
    assert ans.resolve('cabinet', parsed, fallback_any=False) is None


def test_resolve_absent_default_still_best_effort():
    """默认 fallback_any=True 保持旧行为(object_reference 依赖):仍返回某物。"""
    ans, _ = make_answerer([mk(0, 'sofa', (1, 0, 0.4))])
    parsed = {**BASE_PARSED, 'constraints': []}
    assert ans.resolve('cabinet', parsed) is not None


def test_can_ground_requires_relation_satisfied():
    """关系必须满足且候选观测稳定，才能触发探索早停。"""
    con = {'action': 'stop_at', 'target': 'cabinet',
           'anchors': ['picture'], 'relation': 'above'}
    parsed = {**BASE_PARSED, 'constraints': [con]}
    ans, llm = make_answerer([mk(0, 'cabinet', (5, 5, 0.5), size=(1, 0.5, 1))])
    assert ans.can_ground('cabinet', parsed, con) is False
    assert llm.calls == []
    low_conf = _cab_pic()
    ans2, _ = make_answerer(low_conf)
    assert ans2.can_ground('cabinet', parsed, con) is False  # q5 误检初期
    low_conf[0].n_obs = 6
    ans3, _ = make_answerer(low_conf)
    assert ans3.can_ground('cabinet', parsed, con) is True


def test_can_ground_no_constraint_label_presence():
    """无约束/关系时,类别存在即视为可 grounding。"""
    ans, _ = make_answerer([mk(0, 'lamp', (2, 0, 1))])
    parsed = {**BASE_PARSED, 'constraints': []}
    assert ans.can_ground('lamp', parsed) is True
    assert ans.can_ground('cabinet', parsed) is False


def test_plan_final_absent_no_bogus_far_waypoint():
    """最终 stop_at 目标缺失时,终点停在已到达的中间物体附近,不被随机物拉走。"""
    ans, _ = make_answerer([mk(0, 'lamp', (2, 0, 1)), mk(1, 'sofa', (1, 3, 0.4))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'pass_near', 'target': 'lamp',
                               'anchors': [], 'relation': 'none'},
                              {'action': 'stop_at', 'target': 'cabinet',
                               'anchors': [], 'relation': 'none'}]}
    wps, _ = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert wps                                    # 到达了 lamp
    assert np.linalg.norm(np.asarray(wps[-1]) - [2, 0]) < 1.5


def test_plan_stop_at_relation_disambiguates_endpoint():
    """两个 cabinet 时,终点指向"上方有 picture"的那个,而非孤立的。"""
    ans, llm = make_answerer(_cab_pic() + [mk(3, 'lamp', (1, 0, 1))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'stop_at', 'target': 'cabinet',
                               'anchors': ['picture'], 'relation': 'above'}]}
    wps, _ = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert np.linalg.norm(np.asarray(wps[-1]) - [0, 0]) < 1.5   # 选 cabinet0
    assert llm.calls == []
