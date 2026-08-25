import json
from types import SimpleNamespace

import numpy as np
import pytest

import smart_vlm.llm_client as lc
from smart_vlm.llm_client import LLMClient, extract_json
from conftest import FakeLogger


# ---------- extract_json regression cases ----------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {'a': 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {'a': 1}
    assert extract_json('```\n[1, 2]\n```') == [1, 2]


def test_extract_json_with_prose():
    assert extract_json('Sure! Here is it: {"a": 1} hope it helps') == {'a': 1}


def test_extract_json_value_starting_with_json_letters():

    out = extract_json('```json\n{"note": "just so"}\n```')
    assert out == {'note': 'just so'}


def test_extract_json_prefers_fenced_block():


    txt = 'Ids {3, 12} look plausible.\n```json\n{"valid_ids": [12]}\n```'
    assert extract_json(txt) == {'valid_ids': [12]}


def test_extract_json_invalid():
    assert extract_json('no json here') is None
    assert extract_json('') is None
    assert extract_json(None) is None


# ---------- FakeOpenAI ----------

class FakeClient:


    def __init__(self, responder, log):
        self._responder = responder
        self._log = log
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self._log.append(kw)
        text = self._responder(kw)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text))])

    def with_options(self, **kw):
        return self


def make_llm(monkeypatch, responders, budget=9, retries=1, extra=None):


    calls = {name: [] for name in responders}
    import openai

    def fake_openai(base_url=None, api_key=None, timeout=None):
        name = base_url.rsplit('/', 1)[-1]
        return FakeClient(responders[name], calls[name])

    monkeypatch.setattr(openai, 'OpenAI', fake_openai)
    cfg = {'providers': [{'name': n, 'base_url': f'http://x/{n}',
                          'api_key': 'testkey', 'text_model': 'tm',
                          'vision_model': 'vm', **(extra or {})}
                         for n in responders],
           'timeout_s': 1, 'max_retries': retries,
           'call_budget_per_question': budget}
    return LLMClient(cfg, FakeLogger()), calls


def test_ask_basic_json(monkeypatch):
    llm, calls = make_llm(monkeypatch, {'p1': lambda kw: '{"x": 1}'})
    assert llm.ask('q') == {'x': 1}
    assert len(calls['p1']) == 1


def test_failover_to_second_provider(monkeypatch):
    def boom(kw):
        raise RuntimeError('down')
    llm, calls = make_llm(monkeypatch, {'p1': boom, 'p2': lambda kw: '{"x": 2}'})
    monkeypatch.setattr(lc.time, 'sleep', lambda s: None)
    assert llm.ask('q') == {'x': 2}
    assert llm.providers[0]['alive'] is False
    assert llm.providers[1]['alive'] is True


def test_call_budget(monkeypatch):
    llm, calls = make_llm(monkeypatch, {'p1': lambda kw: '{"x": 1}'}, budget=2)
    assert llm.ask('q1') == {'x': 1}
    assert llm.ask('q2') == {'x': 1}
    assert llm.ask('q3') is None
    assert len(calls['p1']) == 2


def test_cache_key_includes_image_content(monkeypatch):

    llm, calls = make_llm(monkeypatch, {'p1': lambda kw: '{"x": 1}'})
    img_a = np.zeros((8, 8, 3), dtype=np.uint8)
    img_b = np.full((8, 8, 3), 255, dtype=np.uint8)
    llm.ask('same', images=[img_a])
    llm.ask('same', images=[img_b])
    assert len(calls['p1']) == 2
    llm.ask('same', images=[img_a])
    assert len(calls['p1']) == 2


def test_text_only_content_is_plain_string(monkeypatch):


    llm, calls = make_llm(monkeypatch, {'p1': lambda kw: '{"x": 1}'})
    llm.ask('hello')
    assert calls['p1'][0]['messages'][0]['content'] == 'hello'


def test_image_content_is_multimodal_list(monkeypatch):
    llm, calls = make_llm(monkeypatch, {'p1': lambda kw: '{"x": 1}'})
    llm.ask('hello', images=[np.zeros((8, 8, 3), dtype=np.uint8)])
    content = calls['p1'][0]['messages'][0]['content']
    assert isinstance(content, list)
    assert content[0] == {'type': 'text', 'text': 'hello'}
    assert content[1]['type'] == 'image_url'


def test_json_repair_retry(monkeypatch):
    state = {'n': 0}

    def responder(kw):
        state['n'] += 1
        if state['n'] == 1:
            return '{"a": 1,,}'
        return '{"a": 1}'
    llm, calls = make_llm(monkeypatch, {'p1': responder})
    assert llm.ask('q') == {'a': 1}
    assert state['n'] == 2
    assert 'Fix to valid JSON' in calls['p1'][1]['messages'][0]['content']


def test_min_interval_between_provider_calls(monkeypatch):


    llm, calls = make_llm(monkeypatch, {'p1': lambda kw: '{"x": 1}'},
                          extra={'min_interval_s': 6.5})
    clock = {'t': 100.0}
    sleeps = []
    monkeypatch.setattr(lc.time, 'monotonic', lambda: clock['t'])

    def fake_sleep(s):
        sleeps.append(s)
        clock['t'] += s
    monkeypatch.setattr(lc.time, 'sleep', fake_sleep)
    llm.ask('q1')
    assert sleeps == []
    llm.ask('q2')
    assert len(calls['p1']) == 2
    assert sleeps and abs(sum(sleeps) - 6.5) < 0.1


def test_replace_me_keys_skipped(monkeypatch):
    import openai
    monkeypatch.setattr(openai, 'OpenAI',
                        lambda **kw: (_ for _ in ()).throw(AssertionError))
    cfg = {'providers': [{'name': 'p1', 'base_url': 'http://x',
                          'api_key': 'REPLACE_ME_ZHIPU_KEY',
                          'text_model': 'tm', 'vision_model': 'vm'}],
           'timeout_s': 1, 'max_retries': 1, 'call_budget_per_question': 9}
    llm = LLMClient(cfg, FakeLogger())
    assert llm.providers == []
    assert llm.ask('q') is None
    assert llm.health_check() is False


def test_health_check_concurrent_marks_dead(monkeypatch):
    def boom(kw):
        raise RuntimeError('down')
    llm, _ = make_llm(monkeypatch, {'p1': boom, 'p2': lambda kw: 'ok'})
    assert llm.health_check(ping_timeout=2.0) is True
    assert llm.providers[0]['alive'] is False
    assert llm.providers[1]['alive'] is True
