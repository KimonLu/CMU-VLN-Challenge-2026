"""问题解析 + 三题型 pipeline(报告 §9)。"""
import re
import cv2
import numpy as np
from . import spatial_tools as st
from .exploration import decimate, decimate_final_approach, line_waypoints

# 关系目标只有 2–4 次观测时容易由局部误检“提前满足”，
# q5 2026-07-10 实测因此在 46s 选中错柜。普通无关系目标不受影响。
EARLY_STOP_REL_MIN_OBS = 5

PARSE_PROMPT = '''You are the language front-end of an indoor robot. Parse the question into JSON:
{"type": "numerical|object_reference|instruction_following",
 "target_nouns": ["..."],
 "detection_vocab": ["..."],
 "constraints": [{"action": "goto|stop_at|avoid|pass_between|pass_near",
                  "target": "<noun phrase>", "anchors": ["<noun phrase>"],
                  "relation": "closest|farthest|between|near|on|above|below|with|none"}],
 "count_subject": "<noun or null>",
 "color_filters": {"<noun>": "<color or null>"}}
Rules:
- target_nouns are the objects the robot must count, mark, pass, or reach; keep
  compound names such as "potted plant", "coffee table", and "computer monitor".
- detection_vocab MUST contain every physical noun/landmark mentioned in the
  question (including small qualifiers such as photo, magazine, phone, folder,
  figurine, whiteboard, and exit sign), plus useful visual synonyms.
- Preserve spatial relations instead of splitting every noun into relation=none.
- For instruction_following, emit one constraint per requested motion, in execution
  order; "avoid" entries describe forbidden areas and are not motion targets.
- For "X on Y closest to Z", target=X, anchors=[Y,Z], relation=closest.
- For "X on Y that has Z on it", emit X-on-Y and Y-with-Z constraints.
Question: "{Q}"
Only output JSON.'''

ARBITRATE_PROMPT = '''{SCENE}

Question: "{Q}"
Candidate objects: {CANDS}
Geometric facts:
{FACTS}
Pick the single object id that best answers the question.
Output JSON: {"object_id": <int>, "confidence": <0-1>}'''

COUNT_CHECK_PROMPT = '''The image is a montage: crops of detected "{LABEL}" candidates side by side, each labeled with its red id number (ids: {IDS}).
Question: "{Q}"
Which ids truly match the question (correct object type and attributes)?
If a crop is too small or blurry to judge confidently, keep its id.
Output JSON: {"valid_ids": [..]}'''

DIRECT_COUNT_PROMPT = '''The image is a 2x2 montage of four perspective views made
from one 360-degree robot panorama. The four views overlap, so the same physical
object can appear in adjacent tiles. The left/right edge of the original panorama
also wraps around.

Question: "{Q}"

List every visible object that satisfies the complete question exactly once. Do not
infer hidden objects and do not double-count an object repeated in overlapping tiles.
Return JSON only: {"items": [{"where": "short unique location description"}]}'''

# 复核送审的 crop 最短边下限(px):SJTU 实测,更小的图放大后连人都认不出,
# VLM 只能保守否决造成误杀 → 看不清的不送审、不扣数
MIN_CROP_PX = 48


def _norm_phrase(value):
    value = re.sub(r'^(?:the|a|an)\s+', '', str(value or '').lower().strip())
    return value[:-1] if value.endswith('s') and not value.endswith('ss') else value


def montage(crops, ids, tile_h=224, pad=4):
    """多张裁剪图横拼成一张带 id 标注的图。

    部分 VLM(如智谱 glm-4v-flash)单请求只接受一张图,复核必须拼图后发送。
    """
    tiles = []
    for crop, oid in zip(crops, ids):
        h, w = crop.shape[:2]
        s = tile_h / h
        t = cv2.resize(crop, (max(1, int(w * s)), tile_h))
        t = cv2.copyMakeBorder(t, 28, 0, pad, pad,
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, str(oid), (pad + 2, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(t)
    return cv2.hconcat(tiles)


class QuestionParser:
    def __init__(self, llm, logger):
        self.llm = llm
        self.log = logger

    VALID_TYPES = ('numerical', 'object_reference', 'instruction_following')

    def parse(self, question):
        out = self.llm.ask(PARSE_PROMPT.replace('{Q}', question))
        norm = self._normalize(out)
        if norm is not None:
            # SJTU 实测:LLM 对 "Find the X on Y" 这类带介词短语的题会错判
            # 成 instruction_following;比赛题面前缀固定 → 规则强制覆盖 type,
            # 槽位(constraints/color_filters 等)仍用 LLM 的
            norm['type'] = self._rule_type(question) or norm['type']
            self._repair_object_reference(question, norm)
            return norm
        self.log.warn('parse: LLM 解析失败,落入正则兜底')
        return self._regex_fallback(question)

    @staticmethod
    def _rule_type(question):
        """题面前缀能确定题型时返回之,否则 None(保留 LLM 判断)。"""
        ql = question.lower().strip()
        if ql.startswith(('how many', 'count')):
            return 'numerical'
        if ql.startswith('find'):
            return 'object_reference'
        return None

    @staticmethod
    def _repair_object_reference(question, parsed):
        """补强比赛中稳定出现的嵌套承载句式，避免 LLM 把所有名词拆成 none。"""
        if parsed.get('type') != 'object_reference':
            return
        m = re.match(
            r'(?i)^find\s+the\s+(.+?)\s+on\s+the\s+(.+?)\s+'
            r'(?:that\s+has|with)\s+the\s+(.+?)\s+on\s+it[.]?$',
            question.strip())
        if not m:
            return
        target, support, qualifier = (x.strip().lower() for x in m.groups())
        parsed['target_nouns'] = [target]
        parsed['constraints'] = [
            {'action': 'goto', 'target': target, 'anchors': [support],
             'relation': 'on'},
            {'action': 'stop_at', 'target': support, 'anchors': [qualifier],
             'relation': 'with'},
        ]
        parsed['detection_vocab'] = list(dict.fromkeys(
            list(parsed.get('detection_vocab') or [])
            + [target, support, qualifier]))

    @classmethod
    def _normalize(cls, out):
        """LLM 输出 schema 归一化:补默认键、剔除非法值。不合格返回 None。"""
        if not isinstance(out, dict) or out.get('type') not in cls.VALID_TYPES:
            return None
        out['target_nouns'] = [str(n) for n in (out.get('target_nouns') or []) if n]
        out['detection_vocab'] = [str(n) for n in (out.get('detection_vocab') or []) if n]
        cons = []
        for c in (out.get('constraints') or []):
            if not isinstance(c, dict):
                continue
            c['anchors'] = [str(a) for a in (c.get('anchors') or []) if a]
            c.setdefault('action', 'goto')
            c.setdefault('target', '')
            c.setdefault('relation', 'none')
            cons.append(c)
        out['constraints'] = cons
        out['count_subject'] = out.get('count_subject') or None
        cf = out.get('color_filters')
        out['color_filters'] = cf if isinstance(cf, dict) else {}
        return out

    def _regex_fallback(self, q):
        """API 全挂时的纯规则解析(报告 §10.2 pure_local_mode)。"""
        ql = q.lower()
        qtype = self._rule_type(q) or 'instruction_following'
        nouns = re.findall(
            r'\b(sofa|chair|table|bed|pillow|window|door|picture|painting|tv|lamp|'
            r'plant|vase|trash can|fridge|refrigerator|nightstand|bench|stool|book|'
            r'bowl|cup|kettle|clock|curtain|mirror|speaker|ottoman|suitcase|tray|'
            r'screen|column|figurine|sink|flower)(?:e?s)?\b', ql)
        nouns = list(dict.fromkeys(nouns))
        return {'type': qtype, 'target_nouns': nouns, 'detection_vocab': nouns,
                'constraints': [{'action': 'goto', 'target': n, 'anchors': [],
                                 'relation': 'none'} for n in nouns],
                'count_subject': nouns[0] if nouns else None, 'color_filters': {}}


class Answerer:
    def __init__(self, smap, gridmap, llm, logger):
        self.smap = smap
        self.gm = gridmap
        self.llm = llm
        self.log = logger

    # ---------- 通用:名词短语 -> 物体解析 ----------
    def _label_objs(self, phrase, exclude_ids=()):
        """最长后缀匹配类别名 → 候选 MapObject 列表(无 LLM)。"""
        words = (phrase or '').lower().split()
        for n in range(len(words), 0, -1):
            cands = self.smap.by_label(' '.join(words[-n:]))
            if cands:
                return [c for c in cands if c.oid not in exclude_ids]
        return []

    @staticmethod
    def _stacked(c, a, xy=0.8, z_gap=0.4):
        """垂直堆叠关系:水平接近 + 明显高差。容忍"墙上画 vs 靠墙柜"的
        ~0.4m 水平偏移——st.above() 用物体自身尺寸判水平对齐,对薄画/薄柜
        过紧会漏配(q5:pic(-2.6,-6.6) 在 cab(-2.2,-6.4) 上方,dx=0.4 被 above 判否)。
        对称双向:relation 编码 anchor↔target 任一向都命中。"""
        return (st.hdist(c, a) < xy
                and abs(c.center[2] - a.center[2]) > z_gap)

    def _assoc(self, c, a):
        """c 与 a 的上下承载关系:垂直堆叠或紧贴承载。不含纯 xy 邻近——
        "任一 picture 在 1.5m 内"过宽会放行同高近处错物(q5 多 cabinet 消歧教训)。"""
        return self._stacked(c, a) or st.on_top(c, a) or st.on_top(a, c)

    def _relation_filter(self, cands, constraint, parsed, strict=False):
        """用约束自身的空间关系确定性收窄候选(不调 LLM)。
        strict=True(探索早停用):anchor 未找到或关系不满足 → 返回空(视为未 grounding);
        strict=False(grounding 用):收窄为空则退回原候选,避免过滤过度。"""
        rel = (constraint or {}).get('relation')
        if rel in ('none', None, ''):
            return cands
        anchor_phrases = list(constraint.get('anchors') or [])
        anchor_groups = [self._label_objs(a) for a in anchor_phrases]
        # 关系链中的锚点也可能被下一条 constraint 限定：
        # monitor closest to [cabinet with phone] / lamp on [nightstand with photo]。
        for idx, (phrase, group) in enumerate(zip(anchor_phrases, anchor_groups)):
            for nested in parsed.get('constraints', []):
                if nested is constraint:
                    continue
                nt = str(nested.get('target', ''))
                if (_norm_phrase(nt) != _norm_phrase(phrase)
                        or nested.get('relation') in (None, '', 'none')):
                    continue
                refs = [o for a in (nested.get('anchors') or [])
                        for o in self._label_objs(a)]
                if not refs:
                    continue
                nr = nested.get('relation')
                if nr == 'closest':
                    narrowed = [min(group, key=lambda c: min(
                        st.hdist(c, r) for r in refs))] if group else []
                elif nr == 'farthest':
                    narrowed = [max(group, key=lambda c: min(
                        st.hdist(c, r) for r in refs))] if group else []
                elif nr == 'near':
                    narrowed = [c for c in group if any(st.near(c, r)
                                                        for r in refs)]
                else:
                    narrowed = [c for c in group if any(self._assoc(c, r)
                                                        for r in refs)]
                if narrowed:
                    anchor_groups[idx] = narrowed
                break
        anchors = [o for group in anchor_groups for o in group]
        if not anchors:
            return [] if strict else cands
        if rel == 'closest':
            # 常见嵌套关系：“X on A closest to B”。先保留位于/贴近 A 的 X，
            # 再按到 B 的距离作全局 argmin；Arabic/Chinese/Home/Office 四类题统一。
            if len(anchor_groups) >= 2 and anchor_groups[0] and anchor_groups[-1]:
                supported = [c for c in cands if any(
                    self._assoc(c, a) or st.near(c, a, slack=0.8)
                    for a in anchor_groups[0])]
                pool = supported or cands
                return [min(pool, key=lambda c: min(
                    st.hdist(c, a) for a in anchor_groups[-1]))] if pool else []
            return [min(cands, key=lambda c: min(st.hdist(c, a)
                                                 for a in anchors))] if cands else []
        if rel == 'farthest':
            return [st.farthest(cands, anchors[0])] if cands else []
        if rel == 'between' and len(anchors) >= 2:
            sub = [c for c in cands if st.between(c, anchors[0], anchors[1])]
        elif rel == 'near':
            sub = [c for c in cands if any(st.near(c, a) for a in anchors)]
        else:                       # above/below/on/with:垂直堆叠/承载判定
            sub = [c for c in cands if any(self._assoc(c, a) for a in anchors)]
        return sub if (sub or strict) else cands

    def can_ground(self, phrase, parsed, constraint=None):
        """目标是否已能被确定性 grounding(无 LLM)。供探索早停判据用:
        避免"见到任一同名物体就早停"——带关系的目标要等关系也满足(q5 停错教训)。"""
        cands = self._label_objs(phrase)
        if not cands:
            return False
        color = (parsed.get('color_filters') or {}).get(phrase)
        if color:
            cands = [c for c in cands if c.color == color] or cands
        if constraint:
            cands = self._relation_filter(cands, constraint, parsed, strict=True)
            rel = constraint.get('relation')
            if rel not in (None, '', 'none'):
                return any(c.n_obs >= EARLY_STOP_REL_MIN_OBS for c in cands)
        return len(cands) > 0

    def resolve(self, phrase, parsed, exclude_ids=(), constraint=None,
                fallback_any=True):
        """名词短语 → 最佳 MapObject(颜色过滤 + 约束空间关系 + LLM 兜底)。
        constraint:该短语所属约束,用其 relation/anchors 确定性消歧。
        fallback_any=False:类别完全缺失时返回 None(不抓全库随便一个),
        供指令题目标/锚点解析用,避免把机器人送到错误终点。"""
        cands = self._label_objs(phrase, exclude_ids)
        color = (parsed.get('color_filters') or {}).get(phrase)
        if color:
            cands = [c for c in cands if c.color == color] or cands
        if constraint and len(cands) > 1:
            cands = self._relation_filter(cands, constraint, parsed) or cands
            # 空间关系已确定性收窄后仍多候选:按观测置信度(n_obs)择优,
            # 避免把关键终点掷给 LLM 抖动(q5:多个几何等价柜,取观测最稳的那个)
            rel = (constraint or {}).get('relation')
            if rel not in (None, 'none', '') and len(cands) > 1:
                cands = [max(cands, key=lambda o: o.n_obs)]
        if not cands:
            if not fallback_any:
                return None
            cands = self.smap.confirmed()[:20]       # 兜底:全库给 LLM 选
        if len(cands) == 1:
            return cands[0]
        return self._arbitrate(phrase, cands, parsed)

    def _arbitrate(self, question_or_phrase, cands, parsed):
        anchors = []
        for c in parsed.get('constraints', []):
            for a in c.get('anchors', []):
                objs = self.smap.by_label(a.split()[-1])
                anchors += objs[:2]
        out = self.llm.ask(ARBITRATE_PROMPT
                           .replace('{SCENE}', self.smap.scene_text(
                               parsed.get('target_nouns')))
                           .replace('{Q}', question_or_phrase)
                           .replace('{CANDS}', str([c.oid for c in cands]))
                           .replace('{FACTS}', st.relation_table(cands, anchors)
                                    if anchors else 'n/a'))
        if out and 'object_id' in out:
            for c in cands:
                if c.oid == out['object_id']:
                    return c
        return cands[0] if cands else None

    # ---------- 数量问答(报告 §9.2)----------
    def _best_view_montage(self, label):
        pano = self.smap.best_view(label)
        if pano is None:
            return None
        views = [view for view, _ in self.smap.proj.make_views(pano)]
        if len(views) != 4:
            return pano
        labels = ('front', 'left', 'back', 'right')
        tiles = []
        for view, name in zip(views, labels):
            tile = view.copy()
            cv2.putText(tile, name, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        return cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])

    def _direct_visual_count(self, question, subj):
        image = self._best_view_montage(subj)
        if image is None:
            return None
        out = self.llm.ask(DIRECT_COUNT_PROMPT.replace('{Q}', question),
                           images=[image])
        if not isinstance(out, dict):
            return None
        if isinstance(out.get('items'), list):
            count = len(out['items'])
        else:
            # GLM/DeepSeek 偶尔遵循语义但把 schema 简化成 {"count": N}。
            try:
                count = int(out['count'])
            except (KeyError, TypeError, ValueError):
                return None
        if count < 0 or count > 50:
            return None
        self.log.info(f'visual count {subj!r}: {count}; response={out}')
        return count

    def answer_numerical(self, question, parsed):
        subj = parsed.get('count_subject') or \
            (parsed['target_nouns'][0] if parsed['target_nouns'] else '')
        # 单帧直接计数绕开「开放词汇检测漏一个，物体库就永远少一个」的召回上限。
        # API/VLM 不可用时 fail-quiet，继续使用下面原有的 3D 几何计数。
        visual_n = self._direct_visual_count(question, subj)
        if visual_n is not None:
            return int(visual_n)
        cands = self.smap.by_label(subj.split()[-1]) if subj else []
        color = (parsed.get('color_filters') or {}).get(subj)
        if color:
            cands = [c for c in cands if c.color == color] or cands
        for con in parsed.get('constraints', []):
            anchors = []
            for a in con.get('anchors', []):
                o = self.resolve(a, parsed)
                if o:
                    anchors.append(o)
            cands = st.apply_relation(con.get('relation'), cands, anchors)
        n = len(cands)
        # VLM 复核:拼成单图发送(智谱等单请求只收一张图);
        # 只对"发给 VLM 看过且被否决"的候选扣数,没有裁剪图或裁剪图
        # 过小看不清的候选不受影响
        shown = [(c.oid, c.best_crop) for c in cands
                 if c.best_crop is not None
                 and min(c.best_crop.shape[:2]) >= MIN_CROP_PX][:6]
        if shown:
            ids = [oid for oid, _ in shown]
            img = montage([crop for _, crop in shown], ids)
            out = self.llm.ask(COUNT_CHECK_PROMPT
                               .replace('{LABEL}', subj)
                               .replace('{IDS}', str(ids))
                               .replace('{Q}', question), images=[img])
            if out and isinstance(out.get('valid_ids'), list):
                valid = set()
                for v in out['valid_ids']:
                    try:
                        valid.add(int(v))
                    except (TypeError, ValueError):
                        pass
                n = len(cands) - len(set(ids) - valid)
        return int(n)

    # ---------- 目标指代(报告 §9.3)----------
    def answer_object_reference(self, question, parsed):
        target = parsed['target_nouns'][0] if parsed['target_nouns'] else question
        obj = self.resolve(target, parsed)
        if obj is None:
            objs = self.smap.confirmed() or self.smap.objects
            obj = objs[0] if objs else None
        return obj      # main_node 负责发 Marker

    # ---------- 指令跟随(报告 §9.4)----------
    def plan_instruction(self, question, parsed, robot_xy, step=2.5,
                         final_step=None, final_radius=0.0,
                         return_meta=False, approach_cfg=None,
                         return_alternatives=False):
        """默认返回 (waypoints [(x,y)...], penalty_zones)；
        ``return_meta=True`` 时额外返回 final_start_idx；同时启用
        ``return_alternatives`` 时再返回最终物体周围的多样化备选停靠点。

        ``step`` 是常规航点间距；若给定 ``final_step``，只对最后一个
        导航约束的末 ``final_radius`` 米路径加密，避免 q5 式密集家具区
        在最后一个粗航点前卡住。
        """
        penalty_zones, waypoints = [], []
        approach_cfg = approach_cfg or {}
        final_start_idx = None
        final_alternatives = []
        used = []
        cur = np.asarray(robot_xy, dtype=float)
        constraints = parsed.get('constraints', [])
        moving = [i for i, c in enumerate(constraints)
                  if c.get('action', 'goto') in
                  ('goto', 'stop_at', 'pass_near', 'pass_between')]
        final_moving_idx = moving[-1] if moving else None
        for con_idx, con in enumerate(constraints):
            act = con.get('action', 'goto')
            anc = [a for a in (con.get('anchors') or []) if a]
            if act == 'avoid':
                # LLM 可能把禁行对象放 target 而非 anchors,两处都看
                phrases = anc or ([con['target']] if con.get('target') else [])
                anchors = [self.resolve(a, parsed, fallback_any=False)
                           for a in phrases]
                anchors = [a for a in anchors if a]
                if len(anchors) >= 2:
                    penalty_zones.append((tuple(anchors[0].center[:2]),
                                          tuple(anchors[1].center[:2]), 1.2, 50.0))
                elif len(anchors) == 1:
                    c = anchors[0].center[:2]
                    penalty_zones.append((tuple(c), tuple(c),
                                          max(anchors[0].size[:2]) + 0.8, 50.0))
                continue
            if act == 'pass_between':
                if len(anc) >= 2:
                    a = self.resolve(anc[0], parsed, fallback_any=False)
                    b = self.resolve(anc[-1], parsed,
                                     exclude_ids=[a.oid] if a else (),
                                     fallback_any=False)
                    if a and b:
                        p1, p2 = a.center[:2], b.center[:2]
                        mid = (p1 + p2) / 2
                        perp = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
                        perp = perp / (np.linalg.norm(perp) + 1e-6) * 1.2
                        ent, ext = mid + perp, mid - perp
                        if np.linalg.norm(ent - cur) > np.linalg.norm(ext - cur):
                            ent, ext = ext, ent
                        waypoints += [tuple(ent), tuple(mid), tuple(ext)]
                        cur = ext
                    # SJTU 实测:LLM 常把最终目的地塞进 pass_between.target
                    # 而不另给 goto → 穿过间隙后继续前往 target
                    tgt = (con.get('target') or '').strip()
                    if not (a and b) or not tgt or tgt in anc:
                        continue
                    act = 'goto'
                elif not anc:
                    continue
                else:
                    act = 'pass_near'                  # 单锚点降级
                    con = {**con, 'target': con.get('target') or anc[0]}
            # goto / stop_at / pass_near
            tgt_obj = self.resolve(con.get('target', ''), parsed,
                                   exclude_ids=[o.oid for o in used],
                                   constraint=con, fallback_any=False)
            if tgt_obj is None:
                # 目标类别缺失:不抓随机物当航点(避免错误终点),跳过该约束
                self.log.info(f"resolve {con.get('target')!r} → MISS (skip)")
                continue
            used.append(tgt_obj)
            c0 = tgt_obj.center
            self.log.info(f"resolve {con.get('target')!r} → obj{tgt_obj.oid} "
                          f"{tgt_obj.label} @({c0[0]:.1f},{c0[1]:.1f})")
            if act == 'pass_near':
                min_r, preferred_r, max_r = 0.6, 0.8, 1.2
            else:
                min_r = approach_cfg.get('approach_min_radius_m', 0.8)
                preferred_r = approach_cfg.get(
                    'approach_preferred_radius_m', 1.2)
                max_r = approach_cfg.get('approach_max_radius_m', 2.0)
            ranked = self.gm.reachable_standoffs(
                cur, tgt_obj.center[:2], min_r=min_r, max_r=max_r,
                preferred_r=preferred_r,
                min_clearance=approach_cfg.get(
                    'approach_min_clearance_m', 0.35),
                penalty_zones=penalty_zones)
            if ranked:
                goal, path = ranked[0]['point'], ranked[0]['path']
                self.log.info(
                    f"approach {con.get('target')!r}: path-ranked "
                    f"r={ranked[0]['radius']:.2f}m "
                    f"clearance={ranked[0]['clearance']:.2f}m "
                    f"options={len(ranked)}")
                if con_idx == final_moving_idx:
                    # 只保留空间上真正不同的候选，避免失败后换到同一侧
                    # 相邻栅格，造成“看似重规划、实际原路重试”。
                    chosen = np.asarray(path[-1], dtype=float)
                    for item in ranked[1:]:
                        point = np.asarray(item['path'][-1], dtype=float)
                        if np.linalg.norm(point - chosen) < 0.75:
                            continue
                        if any(np.linalg.norm(point - np.asarray(old)) < 0.75
                               for old in final_alternatives):
                            continue
                        final_alternatives.append(tuple(point))
                        if len(final_alternatives) >= 4:
                            break
            else:
                # 地形尚未覆盖目标周围时保留旧行为；不因新增选择器让已有
                # 可解任务变成空计划。
                goal = self.gm.nearest_free(tgt_obj.center[:2], max_r=max_r)
                path = self.gm.astar(cur, goal, penalty_zones=penalty_zones)
                self.log.info(
                    f"approach {con.get('target')!r}: legacy fallback")
            if path:
                if con_idx == final_moving_idx and final_step is not None:
                    segment, local_final_start = decimate_final_approach(
                        path, step=step, final_step=final_step,
                        final_radius=final_radius, return_start=True)
                    if not segment:
                        segment, local_final_start = [tuple(goal)], 0
                    final_start_idx = len(waypoints) + local_final_start
                else:
                    segment = decimate(path, step) or [tuple(goal)]
            else:
                if con_idx == final_moving_idx and final_step is not None:
                    # 全局图暂时不连通不等于局部不可走：给局部规划器一串
                    # 细航点，而不是从数米外一步直冲家具区终点。
                    segment = line_waypoints(cur, goal, final_step)
                else:
                    segment = [tuple(goal)]
                if con_idx == final_moving_idx:
                    final_start_idx = len(waypoints)
            waypoints += segment
            cur = np.asarray(waypoints[-1])
        if final_start_idx is None and waypoints:
            # 最后约束只有 pass_between 固定点或目标缺失时，
            # 仅把已规划路径的最后一点当作可重规划的终点。
            final_start_idx = len(waypoints) - 1
        if return_meta:
            if return_alternatives:
                return (waypoints, penalty_zones, final_start_idx,
                        final_alternatives)
            return waypoints, penalty_zones, final_start_idx
        return waypoints, penalty_zones
