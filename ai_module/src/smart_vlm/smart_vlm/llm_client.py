"""Failover LLM/VLM client with retries, caching, and call budgets."""

import base64
import hashlib
import json
import re
import threading
import time

import cv2


def extract_json(text):


    if not text:
        return None
    fence = re.search(r'```(?:json)?(.*?)```', text, re.S)
    for seg in ([fence.group(1)] if fence else []) + [text]:
        m = re.search(r'\{.*\}|\[.*\]', seg, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


class LLMClient:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.log = logger
        self.cache = {}
        self.calls = 0
        self.providers = []
        from openai import OpenAI
        for p in cfg['providers']:
            if not p.get('api_key') or p['api_key'].startswith('REPLACE_ME'):
                logger.warn(f"provider {p['name']}: API key is not configured; skipping")
                continue
            try:
                cli = OpenAI(base_url=p['base_url'], api_key=p['api_key'],
                             timeout=cfg['timeout_s'])
                self.providers.append({**p, 'client': cli, 'alive': True})
            except Exception as e:
                logger.warn(f"provider {p['name']} init failed: {e}")


    def health_check(self, ping_timeout=5.0):
        def ping(p):
            try:
                p['client'].with_options(timeout=ping_timeout) \
                    .chat.completions.create(
                        model=p['text_model'], max_tokens=2,
                        messages=[{'role': 'user', 'content': 'hi'}])
                p['alive'] = True
                self.log.info(f"API OK: {p['name']}")
            except Exception as e:
                p['alive'] = False
                self.log.warn(f"API DOWN: {p['name']}: {e}")
        threads = [threading.Thread(target=ping, args=(p,), daemon=True)
                   for p in self.providers]
        for t in threads:
            t.start()
        for t in threads:
            t.join(ping_timeout + 1.0)
        return any(p['alive'] for p in self.providers)


    def ask(self, prompt, images=None, expect_json=True):

        if self.calls >= self.cfg['call_budget_per_question']:
            self.log.warn('LLM call budget exhausted')
            return None
        b64s = [self._encode(img) for img in (images or [])]
        key = hashlib.md5((prompt + '|' + '|'.join(b64s)).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]


        if b64s:
            content = [{'type': 'text', 'text': prompt}]
            content += [{'type': 'image_url',
                         'image_url': {'url': f'data:image/jpeg;base64,{b}'}}
                        for b in b64s]
        else:
            content = prompt
        for p in self.providers:
            if not p['alive']:
                continue
            model = p['vision_model'] if images else p['text_model']
            delay = 1.0
            for attempt in range(self.cfg['max_retries']):
                try:
                    self._throttle(p)
                    self.calls += 1
                    rsp = p['client'].chat.completions.create(
                        model=model, temperature=0,
                        messages=[{'role': 'user', 'content': content}])
                    text = rsp.choices[0].message.content
                    out = self._parse_json(text, p, model) if expect_json else text
                    if out is not None:
                        self.cache[key] = out
                        return out
                    self.log.warn(
                        f"{p['name']}/{model}: could not extract JSON; switching provider;"
                        f" output excerpt: {str(text)[:120]!r}")
                    break
                except Exception as e:
                    self.log.warn(f"{p['name']} attempt {attempt}: {e}")
                    time.sleep(delay)
                    delay *= 2
            p['alive'] = False
        self.log.warn('ask: all providers failed; returning None')
        return None

    @staticmethod
    def _throttle(p):


        mi = p.get('min_interval_s')
        if not mi:
            return
        last = p.get('_last_call')
        if last is not None:
            wait = mi - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        p['_last_call'] = time.monotonic()

    def _parse_json(self, text, p, model):
        out = extract_json(text)
        if out is not None:
            return out
        try:
            self._throttle(p)
            rsp = p['client'].chat.completions.create(
                model=model, temperature=0,
                messages=[{'role': 'user', 'content':
                           f'Fix to valid JSON, output only JSON:\n{text}'}])
            return extract_json(rsp.choices[0].message.content)
        except Exception:
            return None

    def _encode(self, img):
        small = self._shrink(img, self.cfg.get('vision_max_side', 512))
        return base64.b64encode(
            cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY,
                                         self.cfg.get('jpeg_quality', 80)])[1]
        ).decode()

    @staticmethod
    def _shrink(img, max_side=512):
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
