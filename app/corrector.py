"""
SpellCorrector — the SciSpell correction engine (V5).
Source notebooks: 04 (core), 08 (trust map), 09 (channels), 10 (modern LM).
V5 scoring: error model + blended context, where context = β·modern-news
bigram + (1−β)·domain bigram, and the unigram prior = α-mixture of domain
frequency and modern frequency (wordfreq). OOV rule: a context word may back
off to <unk>; a candidate word never does — candidates cannot inherit the
aggregate probability mass of all rare words.
All resources are artefacts; missing artefacts degrade gracefully to V3/V1.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

from edit_distance import tokenize, weighted_edit_distance
from language_model import BigramLM

try:
    from wordfreq import zipf_frequency
    _HAS_WORDFREQ = True
except ImportError:
    _HAS_WORDFREQ = False

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"

_SOUNDEX = {**{c: "1" for c in "bfpv"}, **{c: "2" for c in "cgjkqsxz"},
            **{c: "3" for c in "dt"}, "l": "4", **{c: "5" for c in "mn"}, "r": "6"}

def soundex(w):
    w = [c for c in w.lower() if c.isalpha()]
    if not w:
        return ""
    first = w[0]
    codes, prev = [], _SOUNDEX.get(first, "")
    for c in w[1:]:
        d = _SOUNDEX.get(c, "")
        if c in "hw":
            continue
        if d and d != prev:
            codes.append(d)
        prev = d
    return (first.upper() + "".join(codes) + "000")[:4]

VOWELS = set("aeiou")

def skeleton(w):
    s = "".join(c for c in w if c not in VOWELS)
    return s if s else w

class ModernLM:
    """Bigram LM over cleaned modern news text (see notebook 10)."""
    def __init__(self, path, prune=1):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.k, self.V, self.N = d["k"], d["V"], d["N"]
        self.uni = d["unigrams"]
        self.bi = {tuple(key.split(" ")): c for key, c in d["bigrams"].items()
                   if c >= prune}
        self.vocab = set(self.uni)

    def p_bigram(self, a, b):
        """Context may back off to <unk>; the TARGET word may not."""
        a = a if a in self.vocab else "<unk>"
        c = self.bi.get((a, b), 0) if b in self.vocab else 0
        return (c + self.k) / (self.uni.get(a, 0) + self.k * self.V)

    def p_unigram(self, w):
        c = self.uni.get(w, 0) if w in self.vocab else 0
        return (c + self.k) / (self.N + self.k * self.V)

class SpellCorrector:
    SND_LEN_WINDOW = 2
    SND_CAP = 200

    def __init__(self, data_dir, lam=3.0, m_sus=1.0, alpha=0.5, beta=0.8, prune=1):
        data_dir = Path(data_dir)
        self.lam = lam
        self.m_sus = m_sus
        self.dictionary = set(
            (data_dir / "dictionary.txt").read_text(encoding="utf-8").split("\n"))
        self.word_freq = {w: int(c) for w, c in json.loads(
            (data_dir / "word_freq.json").read_text(encoding="utf-8")).items()}
        self.lm = BigramLM(data_dir / "language_model.json")
        conf = json.loads((data_dir / "confusion_sets.json").read_text(encoding="utf-8"))
        self.margin_sym  = conf["margins"]["symmetric"]
        self.margin_asym = conf["margins"]["asymmetric"]
        self.confusable = {w: (set(a), self.margin_sym)
                           for w, a in conf["symmetric"].items()}
        self.confusable.update({w: ({a}, self.margin_asym)
                                for w, a in conf["asymmetric"].items()})
        trust_path = data_dir / "word_trust.json"
        self.suspicious = {}
        if trust_path.exists():
            trust = json.loads(trust_path.read_text(encoding="utf-8"))
            self.suspicious = {w: [a for a, _ in alts]
                               for w, alts in trust["suspicious"].items()}
        modern_path = data_dir / "modern_lm.json"
        self.mlm = ModernLM(modern_path, prune=prune) if modern_path.exists() else None
        self.alpha = alpha if (_HAS_WORDFREQ and self.mlm) else 0.0
        self.beta  = beta if self.mlm else 0.0
        self._zipf_cache = {}
        self.sound_index, self.skel_index = defaultdict(list), defaultdict(list)
        for w in self.dictionary:
            self.sound_index[soundex(w)].append(w)
            self.skel_index[skeleton(w)].append(w)

    # ── candidate generation ──
    def _edits1(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        return ({L + R[1:] for L, R in splits if R} |
                {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1} |
                {L + c + R[1:] for L, R in splits if R for c in ALPHABET} |
                {L + c + R for L, R in splits for c in ALPHABET})

    def _known(self, strings):
        return {s for s in strings if s in self.dictionary}

    def _channel_candidates(self, word):
        skel = set(self.skel_index.get(skeleton(word), []))
        bucket = self.sound_index.get(soundex(word), [])
        snd = sorted((w for w in bucket
                      if abs(len(w) - len(word)) <= self.SND_LEN_WINDOW),
                     key=lambda w: (abs(len(w) - len(word)), w))[:self.SND_CAP]
        return (skel | set(snd)) - {word}

    def candidates(self, word):
        if word in self.dictionary:
            return {word}, 0
        c1 = self._known(self._edits1(word))
        if c1:
            return c1, 1
        c2 = self._known({e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)})
        return (c2, 2) if c2 else (set(), -1)

    # ── scoring ──
    def _p_modern_word(self, w):
        if w not in self._zipf_cache:
            z = zipf_frequency(w, "en") if _HAS_WORDFREQ else 0.0
            self._zipf_cache[w] = (10 ** z) / 1e9 if z > 0 else 1e-11
        return self._zipf_cache[w]

    def _p_uni_mix(self, w):
        return (1 - self.alpha) * self.lm.p_unigram(w) + self.alpha * self._p_modern_word(w)

    def _p_ctx(self, a, b):
        p_dom = self.lm.p_bigram(a, b)
        if not self.mlm:
            return p_dom
        return self.beta * self.mlm.p_bigram(a, b) + (1 - self.beta) * p_dom

    def _ctx_logp(self, w, left, right):
        """Pure domain bigram — the calibrated confusable margin test."""
        s = 0.0
        if left  is not None: s += math.log(self.lm.p_bigram(left, w))
        if right is not None: s += math.log(self.lm.p_bigram(w, right))
        if left is None and right is None: s += math.log(self.lm.p_unigram(w))
        return s

    def _ctx_logp_blend(self, w, left, right):
        s = 0.0
        if left is not None:
            s += math.log(0.7 * self._p_ctx(left, w) + 0.3 * self._p_uni_mix(w))
        if right is not None:
            s += math.log(0.7 * self._p_ctx(w, right) + 0.3 * self._p_uni_mix(w))
        if left is None and right is None:
            s += math.log(self._p_uni_mix(w))
        return s

    def suggest(self, word, left=None, right=None, top=5):
        if word in self.dictionary:
            return [(word, 0.0)], 0
        def rank(cands):
            return sorted(((w, -self.lam * weighted_edit_distance(word, w)
                            + self._ctx_logp_blend(w, left, right)) for w in cands),
                          key=lambda x: (-x[1], x[0]))
        c1 = self._known(self._edits1(word))
        ranked = rank(c1)
        tier = 1 if c1 else -1
        c2 = set()
        if len(ranked) < top:
            c2 = self._known({e2 for e1 in self._edits1(word)
                              for e2 in self._edits1(e1)}) - c1
            ranked += rank(c2)
            if not c1:
                tier = 2 if c2 else -1
        if len(ranked) < top:
            ch = self._channel_candidates(word) - c1 - c2
            ranked += rank(ch)
            if tier == -1 and ch:
                tier = 3
        return ranked[:top], tier

    def _check_confusable(self, word, left, right):
        alts, margin = self.confusable[word]
        ranking = sorted(((w, self._ctx_logp(w, left, right))
                          for w in {word} | alts), key=lambda x: -x[1])
        s_word = dict(ranking)[word]
        best_alt, s_best = next((w, s) for w, s in ranking if w != word)
        if s_best - s_word > margin:
            return True, best_alt, ranking
        return False, None, ranking

    def _check_suspicious(self, word, left, right):
        cands = [word] + self.suspicious[word]
        ranking = sorted(((w, self._ctx_logp_blend(w, left, right)) for w in cands),
                         key=lambda x: -x[1])
        s_word = dict(ranking)[word]
        best_alt, s_best = next((w, s) for w, s in ranking if w != word)
        return s_best - s_word, best_alt, [(w, s) for w, s in ranking if w != word]

    def analyze(self, text, top=5):
        tokens = tokenize(text)
        results = []
        for i, tok in enumerate(tokens):
            left  = tokens[i - 1] if i > 0 else None
            right = tokens[i + 1] if i < len(tokens) - 1 else None
            if tok not in self.dictionary:
                ranked, tier = self.suggest(tok, left, right, top)
                results.append({"token": tok, "status": "non_word",
                                "suggestions": ranked, "tier": tier})
            elif tok in self.confusable:
                flagged, best, ranking = self._check_confusable(tok, left, right)
                results.append({"token": tok,
                                "status": "real_word_error" if flagged else "ok",
                                "suggestions": [(w, s) for w, s in ranking
                                                 if w != tok] if flagged else [],
                                "tier": 0})
            else:
                results.append({"token": tok, "status": "ok",
                                "suggestions": [], "tier": 0})
        if self.suspicious:
            broken = [r["status"] == "non_word" or r["token"] in self.suspicious
                      for r in results]
            for i, r in enumerate(results):
                if r["status"] != "ok" or r["token"] not in self.suspicious:
                    continue
                if not ((i > 0 and broken[i-1]) or
                        (i < len(results)-1 and broken[i+1])):
                    continue
                left  = results[i-1]["token"] if i > 0 else None
                right = results[i+1]["token"] if i < len(results)-1 else None
                gap, best, ranking = self._check_suspicious(r["token"], left, right)
                if gap > self.m_sus:
                    r.update(status="suspicious_word", suggestions=ranking[:top],
                             gap=round(gap, 1))
        return results
