"""
Bigram language model for SciSpell.
Developed and documented in notebooks/03_Language_Model.ipynb — that notebook is
the source of truth; edit there and re-export.
"""
import json
import math
from pathlib import Path

class BigramLM:
    """Add-k smoothed bigram model loaded from language_model.json."""

    def __init__(self, model_path):
        payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.k    = payload["k"]
        self.bos  = payload["bos"]
        self.eos  = payload["eos"]
        self.unk  = payload["unk"]
        self.uni  = payload["unigrams"]
        self.bi   = payload["bigrams"]
        self.V    = len(self.uni) + 1                      # +1 for <UNK>
        self.N    = sum(self.uni.values())

    def _norm(self, w):
        return w if w in self.uni else self.unk

    def p_unigram(self, w):
        """Add-k smoothed unigram probability P(w)."""
        return (self.uni.get(self._norm(w), 0) + self.k) / (self.N + self.k * self.V)

    def p_bigram(self, prev, w):
        """Add-k smoothed bigram probability P(w | prev)."""
        prev, w = self._norm(prev), self._norm(w)
        return ((self.bi.get(prev + " " + w, 0) + self.k)
                / (self.uni.get(prev, 0) + self.k * self.V))

    def logp_sentence(self, tokens):
        """Natural-log probability of a token list (markers added here)."""
        seq = [self.bos] + list(tokens) + [self.eos]
        return sum(math.log(self.p_bigram(v, w)) for v, w in zip(seq[:-1], seq[1:]))

    def perplexity(self, sentences):
        """Corpus perplexity over an iterable of token lists."""
        logp, n = 0.0, 0
        for toks in sentences:
            logp += self.logp_sentence(toks)
            n += len(toks) + 1
        return math.exp(-logp / n)
