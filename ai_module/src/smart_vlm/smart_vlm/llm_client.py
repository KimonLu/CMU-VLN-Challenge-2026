"""多供应商 LLM/VLM 客户端:failover + 重试 + 缓存 + 调用预算(报告 §10)。"""
import base64
import hashlib
import json
import re
import threading
import time

import cv2


def extract_json(text):
    """从模型输出中提取首个 JSON 对象/数组。失败返回 None。

    优先解析 ``` 围栏块:VLM 常先输出含花括号的解释文字再给围栏 JSON,
    全文贪婪匹配会把解释里的 { 一起吞掉。"""
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
                logger.warn(f"provider {p['name']}: api_key 未配置,跳过")
                continue
            try:
                cli = OpenAI(base_url=p['base_url'], api_key=p['api_key'],
                             timeout=cfg['timeout_s'])
                self.providers.append({**p, 'client': cli, 'alive': True})
            except Exception as e:
                logger.warn(f"provider {p['name']} init failed: {e}")

    # ---------- 启动自检(报告 §10.2):并发 ping,不拖慢 BOOT ----------
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

    # ---------- 核心调用 ----------
    def ask(self, prompt, images=None, expect_json=True):
        """images: BGR ndarray 列表(自动转 base64)。全链失败返回 None。"""
        if self.calls >= self.cfg['call_budget_per_question']:
            self.log.warn('LLM call budget exhausted')
            return None
        b64s = [self._encode(img) for img in (images or [])]
        key = hashlib.md5((prompt + '|' + '|'.join(b64s)).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        # 纯文本必须用字符串 content:部分供应商(如智谱文本模型)对
        # 列表形式的 content 不解析,会当成空消息回复
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
                        f"{p['name']}/{model}: 无法从输出提取 JSON,换供应商;"
                        f" 输出片段: {str(text)[:120]!r}")
                    break  # JSON 修复也失败 → 换供应商
                except Exception as e:
                    self.log.warn(f"{p['name']} attempt {attempt}: {e}")
                    time.sleep(delay)
                    delay *= 2
            p['alive'] = False   # 本题内不再尝试该供应商(每题重启进程)
        self.log.warn('ask: 所有供应商均失败,返回 None')
        return None

    @staticmethod
    def _throttle(p):
        """供应商最小调用间隔:部分服务(SJTU)超限=请求挂起 ~58s 而非 429,
        会被 timeout 判成连环超时假 DOWN → 主动限速绕开。首次调用不等待。"""
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
        try:  # 一次修复重试(报告 §9.5)
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
