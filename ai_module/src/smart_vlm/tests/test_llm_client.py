import json
from types import SimpleNamespace

import numpy as np
import pytest

import smart_vlm.llm_client as lc
from smart_vlm.llm_client import LLMClient, extract_json
from conftest import FakeLogger


# ---------- extract_json(P0 ④) ----------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {'a': 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {'a': 1}
    assert extract_json('```\n[1, 2]\n```') == [1, 2]


def test_extract_json_with_prose():
    assert extract_json('Sure! Here is it: {"a": 1} hope it helps') == {'a': 1}


def test_extract_json_value_starting_with_json_letters():
    """旧 lstrip('json') 会逐字符误删 j/s/o/n 开头的正文。"""
    out = extract_json('```json\n{"note": "just so"}\n```')
    assert out == {'note': 'just so'}


def test_extract_json_prefers_fenced_block():
    """qwen3.6-27b 实测:先输出含花括号的解释文字再给 ```json 围栏 →
    贪婪正则会把解释里的 { 一起吞掉,须优先解析围栏块。"""
    txt = 'Ids {3, 12} look plausible.\n```json\n{"valid_ids": [12]}\n```'
    assert extract_json(txt) == {'valid_ids': [12]}


def test_extract_json_invalid():
    assert extract_json('no json here') is None
    assert extract_json('') is None
    assert extract_json(None) is None


# ---------- FakeOpenAI ----------

class FakeClient:
    """responder(kwargs) -> str 或 raise。记录所有调用。"""

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
    """responders: {provider_name: responder};按 cfg 顺序命名 p1, p2...
    extra: 附加到每个 provider 配置的键(如 min_interval_s)。"""
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
    monkeypatch.setattr(lc.time, 'sleep', lambda s: None)   # 不真等退避
    assert llm.ask('q') == {'x': 2}
    assert llm.providers[0]['alive'] is False
    assert llm.providers[1]['alive'] is True


def test_call_budget(monkeypatch):
    llm, calls = make_llm(monkeypatch, {'p1': lambda kw: '{"x": 1}'}, budget=2)
    assert llm.ask('q1') == {'x': 1}
    assert llm.ask('q2') == {'x': 1}
    assert llm.ask('q3') is None                 # 预算耗尽
    assert len(calls['p1']) == 2


def test_cache_key_includes_image_content(monkeypatch):
    """P0 ⑤:同 prompt 不同图片不得命中同一缓存。"""
    llm, calls = make_llm(monkeypatch, {'p1': lambda kw: '{"x": 1}'})
    img_a = np.zeros((8, 8, 3), dtype=np.uint8)
    img_b = np.full((8, 8, 3), 255, dtype=np.uint8)
    llm.ask('same', images=[img_a])
    llm.ask('same', images=[img_b])
    assert len(calls['p1']) == 2                 # 旧实现只会调 1 次
    llm.ask('same', images=[img_a])
    assert len(calls['p1']) == 2                 # 相同图片 → 命中缓存


def test_text_only_content_is_plain_string(monkeypatch):
    """智谱等供应商对列表形式 content 的纯文本消息不解析(当成空消息回复),
    冒烟实测发现 → 纯文本必须发字符串。"""
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
            return '{"a": 1,,}'                  # 坏 JSON
        return '{"a": 1}'                        # 修复请求的回复
    llm, calls = make_llm(monkeypatch, {'p1': responder})
    assert llm.ask('q') == {'a': 1}
    assert state['n'] == 2
    assert 'Fix to valid JSON' in calls['p1'][1]['messages'][0]['content']


def test_min_interval_between_provider_calls(monkeypatch):
    """SJTU 实测:超限=请求挂起 ~58s 而非 429 → timeout 15s 会连环假超时把
    供应商标 DOWN。供应商可配 min_interval_s 主动限速,首次调用不等待。"""
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
    assert sleeps == []                          # 首次不等待
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
