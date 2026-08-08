"""
Edit distance module for SciSpell.
Developed and documented in notebooks/02_Edit_Distance.ipynb — that notebook is
the source of truth; edit there and re-export.
"""
import re
import numpy as np

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize(text):
    """Lowercase the text and return alphabetic tokens (internal apostrophes kept)."""
    return TOKEN_RE.findall(text.lower())

KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]

def _build_adjacency():
    pos = {ch: (r, c) for r, row in enumerate(KEYBOARD_ROWS)
                      for c, ch in enumerate(row)}
    adj = {ch: set() for ch in pos}
    for a, (ra, ca) in pos.items():
        for b, (rb, cb) in pos.items():
            if a != b and abs(ra - rb) <= 1 and abs(ca - cb) <= 1:
                adj[a].add(b)
    return adj

KEY_ADJ = _build_adjacency()

COST_SUB_ADJACENT = 0.6
COST_SUB_FAR      = 1.0
COST_INS_DEL      = 0.9
COST_TRANSPOSE    = 0.7

def levenshtein(source, target):
    """Classic edit distance: insertion, deletion, substitution, all cost 1."""
    m, n = len(source), len(target)
    D = np.zeros((m + 1, n + 1), dtype=int)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                          D[i - 1, j - 1] + cost)
    return int(D[m, n])

def damerau_levenshtein(source, target):
    """Edit distance with adjacent transposition as a fourth unit-cost operation."""
    m, n = len(source), len(target)
    D = np.zeros((m + 1, n + 1), dtype=int)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                          D[i - 1, j - 1] + cost)
            if (i > 1 and j > 1 and source[i - 1] == target[j - 2]
                    and source[i - 2] == target[j - 1]):
                D[i, j] = min(D[i, j], D[i - 2, j - 2] + 1)
    return int(D[m, n])

def weighted_edit_distance(source, target):
    """QWERTY-weighted Damerau distance; lower = more plausible typing slip."""
    m, n = len(source), len(target)
    D = np.zeros((m + 1, n + 1))
    D[:, 0] = np.arange(m + 1) * COST_INS_DEL
    D[0, :] = np.arange(n + 1) * COST_INS_DEL
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s_ch, t_ch = source[i - 1], target[j - 1]
            if s_ch == t_ch:
                sub = 0.0
            elif t_ch in KEY_ADJ.get(s_ch, set()):
                sub = COST_SUB_ADJACENT
            else:
                sub = COST_SUB_FAR
            D[i, j] = min(D[i - 1, j] + COST_INS_DEL,
                          D[i, j - 1] + COST_INS_DEL,
                          D[i - 1, j - 1] + sub)
            if (i > 1 and j > 1 and s_ch == target[j - 2]
                    and source[i - 2] == t_ch):
                D[i, j] = min(D[i, j], D[i - 2, j - 2] + COST_TRANSPOSE)
    return float(D[m, n])
