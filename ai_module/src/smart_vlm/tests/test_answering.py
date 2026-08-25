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
    assert out['type'] == 'instruction_following'
    assert 'sofa' in out['target_nouns']


def test_regex_fallback_types(logger):
    p = QuestionParser(FakeLLM(None), logger)
    assert p.parse('How many chairs are there')['type'] == 'numerical'
    assert p.parse('Find the pillow on the sofa')['type'] == 'object_reference'
    assert p.parse('Take the path between columns')['type'] == \
        'instruction_following'


def test_regex_fallback_matches_plural_nouns(logger):

    p = QuestionParser(FakeLLM(None), logger)
    out = p.parse('How many chairs are in the room?')
    assert out['count_subject'] == 'chair'
    assert out['target_nouns'] == ['chair']
    out = p.parse('Walk past the benches and stop')
    assert 'bench' in out['target_nouns']


def test_parse_type_forced_by_question_prefix(logger):


    llm_out = {'type': 'instruction_following', 'target_nouns': ['vase'],
               'detection_vocab': ['vase'], 'count_subject': None,
               'color_filters': {},
               'constraints': [{'action': 'goto', 'target': 'vase',
                                'anchors': ['coffee table'], 'relation': 'on'}]}
    p = QuestionParser(FakeLLM(llm_out), logger)
    out = p.parse('Find the vase on the coffee table.')
    assert out['type'] == 'object_reference'
    assert out['constraints'][0]['anchors'] == ['coffee table']


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


    objs = [mk(i, 'chair', (i, 0, 0)) for i in range(3)]
    for o in objs:
        o.best_crop = np.full((60, 50, 3), 128, dtype=np.uint8)
    llm = FakeLLM({'valid_ids': [0, 2]})
    ans, _ = make_answerer(objs, llm)
    parsed = {'type': 'numerical', 'count_subject': 'chair',
              'target_nouns': ['chair'], 'color_filters': {}, 'constraints': []}
    n = ans.answer_numerical('How many chairs are in the room?', parsed)
    assert len(llm.images) == 1 and len(llm.images[0]) == 1
    img = llm.images[0][0]
    assert img.shape[1] > 3 * 50
    assert n == 2
    assert '[0, 1, 2]' in llm.calls[0]


def test_count_check_uncropped_candidates_not_penalized():

    objs = [mk(i, 'chair', (i, 0, 0)) for i in range(3)]
    objs[0].best_crop = np.full((60, 50, 3), 128, dtype=np.uint8)
    llm = FakeLLM({'valid_ids': [0]})
    ans, _ = make_answerer(objs, llm)
    parsed = {'type': 'numerical', 'count_subject': 'chair',
              'target_nouns': ['chair'], 'color_filters': {}, 'constraints': []}

    assert ans.answer_numerical('How many chairs?', parsed) == 3
    assert '[0]' in llm.calls[0]


def test_count_check_skips_tiny_crops():


    objs = [mk(i, 'chair', (i, 0, 0)) for i in range(3)]
    objs[0].best_crop = np.full((60, 60, 3), 128, np.uint8)
    objs[1].best_crop = np.full((20, 20, 3), 128, np.uint8)
    objs[2].best_crop = np.full((60, 60, 3), 128, np.uint8)
    llm = FakeLLM({'valid_ids': [0]})
    ans, _ = make_answerer(objs, llm)
    parsed = {'type': 'numerical', 'count_subject': 'chair',
              'target_nouns': ['chair'], 'color_filters': {}, 'constraints': []}
    n = ans.answer_numerical('How many chairs?', parsed)
    assert '[0, 2]' in llm.calls[0]
    assert n == 2


def test_montage_mixed_sizes():
    from smart_vlm.answering import montage
    crops = [np.zeros((80, 50, 3), np.uint8), np.zeros((30, 120, 3), np.uint8)]
    img = montage(crops, [3, 7], tile_h=64)
    assert img.ndim == 3 and img.shape[0] == 64 + 28


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
    assert len(wps) == 3


def test_plan_pass_between_then_reaches_target():


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
    assert zones == []


def test_plan_waypoint_step_param():
    ans, _ = make_answerer([mk(0, 'plant', (10, 0, 0.3))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'goto', 'target': 'plant',
                               'anchors': [], 'relation': 'none'}]}
    wps_fine, _ = ans.plan_instruction('q', parsed, (0.0, 0.0), step=1.0)
    wps_coarse, _ = ans.plan_instruction('q', parsed, (0.0, 0.0), step=4.0)
    assert len(wps_fine) > len(wps_coarse)


def test_plan_densifies_only_final_constraint():

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

    assert np.allclose(dense[:3], coarse[:3])
    assert np.allclose(dense[-1], coarse[-1])
    assert len(dense) > len(coarse)
    assert all(not np.allclose(a, b) for a, b in zip(dense, dense[1:]))


def test_plan_meta_keeps_pass_between_outside_final_retry_segment():

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
    assert final_start > mid_idx + 1
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


def _cab_pic():

    cab0 = mk(0, 'cabinet', (0, 0, 0.5), size=(1.0, 0.5, 1.0))
    pic = mk(1, 'picture', (0, 0, 1.7), size=(0.6, 0.05, 0.4))
    cab1 = mk(2, 'cabinet', (5, 5, 0.5), size=(1.0, 0.5, 1.0))
    return [cab0, pic, cab1]


def test_resolve_relation_picks_associated_object():

    ans, llm = make_answerer(_cab_pic())
    con = {'action': 'stop_at', 'target': 'cabinet',
           'anchors': ['picture'], 'relation': 'above'}
    parsed = {**BASE_PARSED, 'constraints': [con]}
    obj = ans.resolve('cabinet', parsed, constraint=con)
    assert obj is not None and obj.oid == 0
    assert llm.calls == []


def test_resolve_wall_picture_offset_still_associates():


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

    objs = [mk(10 + i, 'picture', (-2.6, -1.0 - i, 1.5), size=(0.1, 0.4, 0.6))
            for i in range(5)]
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

    cabA = mk(0, 'cabinet', (0, 0, 0.4), size=(0.5, 1.0, 0.7), n_obs=15)
    picA = mk(1, 'picture', (0.2, 0, 1.4), size=(0.1, 0.6, 0.7))
    cabB = mk(2, 'cabinet', (4, 0, 0.4), size=(0.5, 1.0, 0.7), n_obs=3)
    picB = mk(3, 'picture', (4.1, 0, 1.4), size=(0.1, 0.6, 0.7))
    ans, llm = make_answerer([cabA, picA, cabB, picB])
    con = {'action': 'stop_at', 'target': 'cabinet',
           'anchors': ['picture'], 'relation': 'above'}
    obj = ans.resolve('cabinet', {**BASE_PARSED, 'constraints': [con]},
                      constraint=con)
    assert obj is not None and obj.oid == 0
    assert llm.calls == []


def test_resolve_final_target_absent_returns_none():

    ans, _ = make_answerer([mk(0, 'sofa', (1, 0, 0.4)), mk(1, 'lamp', (2, 0, 1))])
    parsed = {**BASE_PARSED, 'constraints': []}
    assert ans.resolve('cabinet', parsed, fallback_any=False) is None


def test_resolve_absent_default_still_best_effort():

    ans, _ = make_answerer([mk(0, 'sofa', (1, 0, 0.4))])
    parsed = {**BASE_PARSED, 'constraints': []}
    assert ans.resolve('cabinet', parsed) is not None


def test_can_ground_requires_relation_satisfied():

    con = {'action': 'stop_at', 'target': 'cabinet',
           'anchors': ['picture'], 'relation': 'above'}
    parsed = {**BASE_PARSED, 'constraints': [con]}
    ans, llm = make_answerer([mk(0, 'cabinet', (5, 5, 0.5), size=(1, 0.5, 1))])
    assert ans.can_ground('cabinet', parsed, con) is False
    assert llm.calls == []
    low_conf = _cab_pic()
    ans2, _ = make_answerer(low_conf)
    assert ans2.can_ground('cabinet', parsed, con) is False
    low_conf[0].n_obs = 6
    ans3, _ = make_answerer(low_conf)
    assert ans3.can_ground('cabinet', parsed, con) is True


def test_can_ground_no_constraint_label_presence():

    ans, _ = make_answerer([mk(0, 'lamp', (2, 0, 1))])
    parsed = {**BASE_PARSED, 'constraints': []}
    assert ans.can_ground('lamp', parsed) is True
    assert ans.can_ground('cabinet', parsed) is False


def test_plan_final_absent_no_bogus_far_waypoint():

    ans, _ = make_answerer([mk(0, 'lamp', (2, 0, 1)), mk(1, 'sofa', (1, 3, 0.4))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'pass_near', 'target': 'lamp',
                               'anchors': [], 'relation': 'none'},
                              {'action': 'stop_at', 'target': 'cabinet',
                               'anchors': [], 'relation': 'none'}]}
    wps, _ = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert wps
    assert np.linalg.norm(np.asarray(wps[-1]) - [2, 0]) < 1.5


def test_plan_stop_at_relation_disambiguates_endpoint():

    ans, llm = make_answerer(_cab_pic() + [mk(3, 'lamp', (1, 0, 1))])
    parsed = {**BASE_PARSED,
              'constraints': [{'action': 'stop_at', 'target': 'cabinet',
                               'anchors': ['picture'], 'relation': 'above'}]}
    wps, _ = ans.plan_instruction('q', parsed, (0.0, 0.0))
    assert np.linalg.norm(np.asarray(wps[-1]) - [0, 0]) < 1.5
    assert llm.calls == []
