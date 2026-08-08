"""
SciSpell — the computational philology laboratory.
Visual layer: archival paper, ink navy, figure-palette accents,
Fraunces / Instrument Sans / JetBrains Mono. Engine: V5 + v3/DistilBERT.
Run from project root:  <python> -m streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(APP_DIR))

from corrector import SpellCorrector
from edit_distance import damerau_levenshtein, TOKEN_RE

# ── resources ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Warming the correction engine…")
def load_corrector():
    return SpellCorrector(DATA_DIR)

@st.cache_resource
def load_corpus_words(_sc):
    return sorted(_sc.word_freq.items())

sc = load_corrector()

st.set_page_config(page_title="SciSpell", page_icon="✳", layout="wide",
                   initial_sidebar_state="collapsed")

# ── the visual system ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,900&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --paper:#FBFAF7; --panel:#FFFFFF; --ink:#1C2B3A; --graphite:#5A6472;
  --line:#E4DFD3; --red:#C2453D; --amber:#B98A2E; --green:#3D7A52;
  --red-bg:#F8E7E5; --amber-bg:#F5EDDA; --green-bg:#E7F0EA;
}
html, body, [class*="css"]{ font-family:'Instrument Sans',sans-serif; color:var(--ink); }
.stApp{ background:var(--paper); }
#MainMenu, footer, header[data-testid="stHeader"]{ visibility:hidden; height:0; }
.block-container{ padding-top:2.2rem; max-width:1180px; }

/* hero */
.hero-kicker{ font-family:'JetBrains Mono',monospace; font-size:.72rem;
  letter-spacing:.22em; color:var(--graphite); text-transform:uppercase; }
.hero-title{ font-family:'Fraunces',serif; font-weight:900; font-size:clamp(3rem,7vw,5.2rem);
  line-height:.95; letter-spacing:-.02em; margin:.1em 0 .05em;
  animation:rise .7s cubic-bezier(.2,.7,.2,1) both; }
.hero-title .dot{ color:var(--red); }
.hero-sub{ font-size:1.02rem; color:var(--graphite); max-width:46rem;
  animation:rise .7s .08s cubic-bezier(.2,.7,.2,1) both; }
.specimen-row{ display:flex; gap:1.6rem; flex-wrap:wrap; margin:1.1rem 0 .9rem;
  animation:rise .7s .16s cubic-bezier(.2,.7,.2,1) both; }
.spec{ font-family:'JetBrains Mono',monospace; font-size:.74rem; color:var(--graphite); }
.spec b{ color:var(--ink); font-weight:600; }
@keyframes rise{ from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:none} }
@media (prefers-reduced-motion: reduce){ *{animation:none !important; transition:none !important} }

/* pills (all radio groups) — hide the dot, keep text alive in every state */
div[role="radiogroup"]{ gap:.45rem; }
div[role="radiogroup"] label{
  border:1px solid var(--line); border-radius:999px; padding:.4rem 1.15rem;
  background:var(--panel); font-weight:500; transition:all .15s; cursor:pointer; }
div[role="radiogroup"] label > div:first-child{ display:none; }   /* the dot */
div[role="radiogroup"] label:hover{ border-color:var(--ink); }
div[role="radiogroup"] label:has(input:checked){
  background:var(--ink); border-color:var(--ink); }
div[role="radiogroup"] label p{ font-size:.9rem; color:var(--ink); }
div[role="radiogroup"] label:has(input:checked) p{ color:#FFFFFF !important; }

/* panels & inputs */
.stTextArea textarea, .stTextInput input{ background:var(--panel);
  border:1px solid var(--line); border-radius:10px; font-size:1.02rem;
  line-height:1.7; color:var(--ink); }
.stTextArea textarea:focus, .stTextInput input:focus{
  border-color:var(--ink); box-shadow:0 0 0 3px #1C2B3A22; }
.stButton>button{ background:var(--ink); color:#fff; border:none; border-radius:999px;
  padding:.55rem 1.5rem; font-weight:600; letter-spacing:.01em; }
.stButton>button:hover{ background:var(--red); color:#fff; }
.stButton>button:focus{ box-shadow:0 0 0 3px #C2453D44; }
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:12px;
  background:var(--panel); }

/* the specimen text — inline diff chips */
.reading{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:1.5rem 1.8rem 1.2rem; line-height:2.15; font-size:1.08rem; }
.tag{ display:inline-flex; align-items:baseline; gap:.32em; border-radius:6px;
  padding:.05em .5em .08em; margin-inline:.12em; white-space:nowrap; }
.tag .orig{ text-decoration:line-through; text-decoration-thickness:2px; opacity:.5; }
.tag .arrow{ opacity:.45; font-size:.82em; }
.tag .fix{ font-family:'JetBrains Mono',monospace; font-weight:600; font-size:.85em; }
.tag.nonword{ background:var(--red-bg); }
.tag.nonword .orig{ text-decoration-color:var(--red); }
.tag.nonword .fix{ color:var(--red); }
.tag.realword, .tag.suspicious{ background:var(--amber-bg); }
.tag.realword .orig, .tag.suspicious .orig{ text-decoration-color:var(--amber); }
.tag.realword .fix, .tag.suspicious .fix{ color:var(--amber); }
.legend{ font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--graphite);
  margin-top:.55rem; }
.legend .sw{ display:inline-block; width:.7em; height:.7em; border-radius:2px;
  vertical-align:-1px; margin-right:.35em; }

/* suggestion & why tables */
.sugg{ width:100%; border-collapse:collapse; background:var(--panel);
  border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.sugg th{ font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--graphite); text-align:left;
  padding:.6rem .9rem; border-bottom:1px solid var(--line); font-weight:600; }
.sugg td{ padding:.5rem .9rem; border-bottom:1px solid #F0ECE2; font-size:.95rem; }
.sugg tr:last-child td{ border-bottom:none; }
.sugg td.num{ font-family:'JetBrains Mono',monospace; text-align:right; color:var(--graphite); }
.sugg tr:first-child td:first-child{ font-weight:600; color:var(--green); }

/* verdicts + custom meter (wears the verdict's colour) */
.verdict{ border-radius:12px; padding:1rem 1.3rem; font-size:1.05rem; margin:.4rem 0; }
.verdict b{ font-family:'Fraunces',serif; font-weight:900; font-size:1.25rem; }
.verdict.pos{ background:var(--green-bg); color:var(--green); }
.verdict.neg{ background:var(--red-bg); color:var(--red); }
.verdict.unsure{ background:var(--amber-bg); color:var(--amber); }
.meter{ height:8px; border-radius:999px; background:#EDE9DE; overflow:hidden;
  margin:.5rem 0 .2rem; }
.meter > div{ height:100%; border-radius:999px; transition:width .4s ease; }
.meter-label{ font-family:'JetBrains Mono',monospace; font-size:.74rem;
  color:var(--graphite); margin-bottom:.8rem; }

.sect{ font-family:'Fraunces',serif; font-weight:600; font-size:1.5rem; margin:1.4rem 0 .5rem; }
hr{ border:none; border-top:1px solid var(--line); margin:1.6rem 0; }
</style>
""", unsafe_allow_html=True)

# ── hero ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-kicker">A computational philology laboratory</div>
<div class="hero-title">SciSpell<span class="dot">.</span></div>
<div class="hero-sub">Noisy-channel spelling correction taught on Darwin, Einstein
and Faraday — re-armed with a modern language model — beside a sentiment engine
measured on three domains.</div>
<div class="specimen-row">
  <div class="spec">CORPUS <b>229,325</b> tokens</div>
  <div class="spec">DICTIONARY <b>376,407</b> words</div>
  <div class="spec">SPELLING ACC@5 <b>96.3%</b></div>
  <div class="spec">SENTIMENT OOD <b>92%</b></div>
</div>
""", unsafe_allow_html=True)

page = st.radio(" ", ["Spell checker", "Sentiment"], horizontal=True,
                label_visibility="collapsed")
st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════ SPELL PAGE ════════════════════════════════
if page == "Spell checker":
    col_editor, col_dict = st.columns([1.9, 1], gap="large")

    with col_editor:
        text = st.text_area("Text to examine", height=150, max_chars=500,
                            placeholder="Paste up to 500 characters — typos, "
                                        "textspeak, dialect all welcome…")
        st.markdown(f"<div class='spec'>{len(text)}/500 CHARACTERS</div>",
                    unsafe_allow_html=True)
        check = st.button("Examine text")

        if check and text.strip():
            st.session_state["results"] = sc.analyze(text, top=8)
            st.session_state["checked_text"] = text

        results = st.session_state.get("results")
        if results and st.session_state.get("checked_text") == text:
            flagged_map, cls_map = {}, {"non_word": "nonword",
                                        "real_word_error": "realword",
                                        "suspicious_word": "suspicious"}
            for r in results:
                if r["status"] != "ok":
                    fix = r["suggestions"][0][0] if r["suggestions"] else "?"
                    flagged_map[r["token"]] = (cls_map[r["status"]], fix)
            out, last = [], 0
            for m in TOKEN_RE.finditer(text.lower()):
                out.append(text[last:m.start()])
                piece = text[m.start():m.end()]
                if m.group() in flagged_map:
                    css, fix = flagged_map[m.group()]
                    out.append(f'<span class="tag {css}">'
                               f'<span class="orig">{piece}</span>'
                               f'<span class="arrow">→</span>'
                               f'<span class="fix">{fix}</span></span>')
                else:
                    out.append(piece)
                last = m.end()
            out.append(text[last:])
            n_nw = sum(r["status"] == "non_word" for r in results)
            n_ctx = sum(r["status"] in ("real_word_error", "suspicious_word")
                        for r in results)
            st.markdown(f'<div class="reading">{"".join(out)}</div>'
                        f'<div class="legend">'
                        f'<span class="sw" style="background:var(--red)"></span>'
                        f'{n_nw} NON-WORD&nbsp;&nbsp;&nbsp;'
                        f'<span class="sw" style="background:var(--amber)"></span>'
                        f'{n_ctx} CONTEXT&nbsp;&nbsp;&nbsp;'
                        f'{len(results)} TOKENS EXAMINED</div>',
                        unsafe_allow_html=True)

            flagged = [r for r in results if r["status"] != "ok"]
            if flagged:
                st.markdown('<div class="sect">Suggestions</div>',
                            unsafe_allow_html=True)
                labels = [r["token"] for r in flagged]
                pick = st.radio("Flagged words", labels, horizontal=True,
                                label_visibility="collapsed")
                r = next(x for x in flagged if x["token"] == pick)
                rows = "".join(
                    f"<tr><td>{w}</td>"
                    f"<td class='num'>{damerau_levenshtein(r['token'], w)}</td>"
                    f"<td class='num'>{s:.2f}</td></tr>"
                    for w, s in r["suggestions"])
                st.markdown(f"<table class='sugg'><tr><th>suggestion</th>"
                            f"<th>min edit distance</th><th>score (log)</th></tr>"
                            f"{rows}</table>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict pos"><b>Clean.</b> '
                            'No spelling issues found.</div>',
                            unsafe_allow_html=True)

    with col_dict:
        st.markdown('<div class="sect">Corpus dictionary</div>',
                    unsafe_allow_html=True)
        query = st.text_input("Search the corpus", placeholder="e.g. evol",
                              label_visibility="collapsed")
        words = load_corpus_words(sc)
        shown = ([(w, f) for w, f in words if w.startswith(query.lower())]
                 if query else words)
        st.markdown(f"<div class='spec'>{len(shown):,} OF {len(words):,} "
                    f"CORPUS WORDS · SORTED</div>", unsafe_allow_html=True)
        st.dataframe({"word": [w for w, _ in shown[:2000]],
                      "frequency": [f for _, f in shown[:2000]]},
                     height=420, width="stretch")

# ══════════════════════════════ SENTIMENT PAGE ════════════════════════════
else:
    import joblib

    @st.cache_resource(show_spinner="Loading fast model…")
    def load_fast():
        return joblib.load(DATA_DIR / "sentiment_model_v3.joblib")

    @st.cache_resource(show_spinner="Loading deep model… (first time is slow)")
    def load_deep():
        import torch
        from transformers import (DistilBertTokenizerFast,
                                  DistilBertForSequenceClassification)
        d = DATA_DIR / "distilbert_sentiment"
        tok = DistilBertTokenizerFast.from_pretrained(d)
        net = DistilBertForSequenceClassification.from_pretrained(d)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        return tok, net.to(device).eval(), device, torch

    deep_available = (DATA_DIR / "distilbert_sentiment").exists()
    mode = st.radio("Model", ["Fast — classical ensemble", "Deep — DistilBERT"]
                    if deep_available else ["Fast — classical ensemble"],
                    horizontal=True)
    st.markdown("<div class='spec'>"
                + ("MOVIE 90.8% · AMAZON 83.2% · YELP 84.2% · EXPLAINABLE"
                   if mode.startswith("Fast") else
                   "MOVIE 91.3% · AMAZON 92.2% · YELP 92.0% · READS WORD ORDER")
                + "</div>", unsafe_allow_html=True)

    review = st.text_area("Text to read", height=150,
                          placeholder="Paste a review — movies, products, "
                                      "restaurants…")
    if st.button("Read sentiment") and review.strip():
        if mode.startswith("Fast"):
            p_pos = float(load_fast().predict_proba([review])[0][1])
        else:
            tok, net, device, torch = load_deep()
            with torch.no_grad():
                enc = tok([review], truncation=True, padding=True,
                          max_length=256, return_tensors="pt").to(device)
                p_pos = float(torch.softmax(net(**enc).logits, -1)[0, 1])

        if 0.42 <= p_pos <= 0.58:
            verdict_color = "var(--amber)"
            st.markdown(f'<div class="verdict unsure"><b>Uncertain.</b> '
                        f'P(positive) = {p_pos:.1%} — this text is mixed, subtle, '
                        f'or unlike the training data. Treat any verdict as a '
                        f'guess.</div>', unsafe_allow_html=True)
        elif p_pos > 0.58:
            verdict_color = "var(--green)"
            st.markdown(f'<div class="verdict pos"><b>Positive.</b> '
                        f'Confidence {p_pos:.1%}</div>', unsafe_allow_html=True)
        else:
            verdict_color = "var(--red)"
            st.markdown(f'<div class="verdict neg"><b>Negative.</b> '
                        f'Confidence {1-p_pos:.1%}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="meter"><div style="width:{p_pos:.1%};'
            f'background:{verdict_color}"></div></div>'
            f'<div class="meter-label">P(POSITIVE) = {p_pos:.1%}</div>',
            unsafe_allow_html=True)

        if mode.startswith("Fast"):
            st.markdown('<div class="sect">Why</div>', unsafe_allow_html=True)
            rows = "".join(f"<tr><td>{f}</td><td class='num'>{c:+.3f}</td></tr>"
                           for f, c in load_fast().explain(review, top=8))
            st.markdown(f"<table class='sugg'><tr><th>feature</th>"
                        f"<th>contribution</th></tr>{rows}</table>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='spec'>THE DEEP MODEL READS ORDER AND SCOPE — "
                        "NO EXACT FEATURE DECOMPOSITION. SWITCH TO FAST FOR ONE."
                        "</div>", unsafe_allow_html=True)

    with st.expander("Model card"):
        st.markdown("""
- **Fast** — soft-vote ensemble (TF-IDF LogReg + NB log-count-ratio), trained on
  24,902 deduplicated IMDB reviews + 1,000 adapted product/restaurant sentences.
- **Deep** — DistilBERT fine-tuned one epoch on the same mix; a second epoch is
  untested by protocol.
- **Known floor** — sarcasm defeats lexicon, linear, and transformer alike
  (37/37 triangulated). The Uncertain band exists because 76% of measured errors
  were low-confidence.
- **Discipline** — every number above is a single, final touch of the official
  test split per shipped model.
""")


# ══════════════════════════════ THE LABORATORY ════════════════════════════
st.markdown("""
<style>
.lab-wrap{ margin-top:3.2rem; }
.lab-kicker{ font-family:'JetBrains Mono',monospace; font-size:.72rem;
  letter-spacing:.22em; color:var(--graphite); text-transform:uppercase; }
.lab-title{ font-family:'Fraunces',serif; font-weight:900; font-size:2rem;
  margin:.15em 0 .8em; }
.lab-lead{ background:var(--ink); color:#F5F2EA; border-radius:16px;
  padding:1.8rem 2rem 1.6rem; margin-bottom:1rem;
  transition:transform .18s ease, box-shadow .18s ease; }
.lab-lead:hover{ transform:translateY(-4px); box-shadow:0 14px 34px #1C2B3A33; }
.lab-lead .lab-mono{ width:64px; height:64px; border-radius:14px; display:flex;
  align-items:center; justify-content:center; font-family:'Fraunces',serif;
  font-weight:900; font-size:1.6rem; background:#F5F2EA; color:var(--ink);
  margin-bottom:1rem; }
.lab-lead .lab-name{ font-family:'Fraunces',serif; font-weight:900;
  font-size:1.7rem; line-height:1.15; }
.lab-lead .lab-role{ font-family:'JetBrains Mono',monospace; font-size:.7rem;
  letter-spacing:.18em; text-transform:uppercase; color:#D9A03F;
  margin:.5em 0 .6em; }
.lab-lead .lab-desc{ font-size:1rem; color:#C9CBBF; line-height:1.65;
  max-width:58rem; }
.lab-lead .lab-desc b{ color:#F5F2EA; }
.lab-scope{ display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:1.1rem; }
.lab-scope .spec{ color:#9AA3AE; }
.lab-scope .spec b{ color:#F5F2EA; }
.lab-grid{ display:flex; gap:1rem; flex-wrap:wrap; }
.lab-card{ flex:1 1 240px; background:var(--panel); border:1px solid var(--line);
  border-radius:14px; padding:1.3rem 1.4rem 1.1rem;
  transition:transform .18s ease, box-shadow .18s ease; }
.lab-card:hover{ transform:translateY(-3px); box-shadow:0 8px 22px #1C2B3A12; }
.lab-card .lab-mono{ width:48px; height:48px; border-radius:11px; display:flex;
  align-items:center; justify-content:center; font-family:'Fraunces',serif;
  font-weight:900; font-size:1.15rem; margin-bottom:.8rem; }
.lab-card.c2 .lab-mono{ background:var(--amber-bg); color:var(--amber); }
.lab-card.c3 .lab-mono{ background:var(--green-bg); color:var(--green); }
.lab-card .lab-name{ font-family:'Fraunces',serif; font-weight:600;
  font-size:1.08rem; line-height:1.2; }
.lab-card .lab-role{ font-family:'JetBrains Mono',monospace; font-size:.66rem;
  letter-spacing:.16em; text-transform:uppercase; margin:.45em 0 .45em; }
.lab-card.c2 .lab-role{ color:var(--amber); }
.lab-card.c3 .lab-role{ color:var(--green); }
.lab-card .lab-desc{ font-size:.9rem; color:var(--graphite); line-height:1.55; }
.lab-foot{ font-family:'JetBrains Mono',monospace; font-size:.72rem;
  color:var(--graphite); margin-top:1.6rem; }
@media (prefers-reduced-motion: reduce){
  .lab-lead, .lab-lead:hover, .lab-card, .lab-card:hover{
    transform:none; box-shadow:none; } }
</style>

<div class="lab-wrap">
  <div class="lab-kicker">The laboratory</div>
  <div class="lab-title">Who built this<span style="color:var(--red)">.</span></div>

  <div class="lab-lead">
    <div class="lab-mono">ZR</div>
    <div class="lab-name">Zaker Hussain Rasooli</div>
    <div class="lab-role">Principal — research &amp; engineering</div>
    <div class="lab-desc">Designed and built SciSpell end to end:
    <b>the corpus and two-layer dictionary</b>, <b>the edit-distance family</b>,
    <b>both language models</b>, <b>the noisy-channel engine</b> with its trust
    map and candidate channels, <b>every evaluation and calibration</b>,
    <b>the sentiment ensemble and DistilBERT fine-tune</b>, and
    <b>this application</b> — from the first downloaded byte of Darwin to the
    pixels on this page.</div>
    <div class="lab-scope">
      <div class="spec"><b>7</b> notebooks</div>
      <div class="spec"><b>6</b> shipped models</div>
      <div class="spec"><b>46+</b> verification checks</div>
      <div class="spec"><b>2</b> engines, one laboratory</div>
    </div>
  </div>

  <div class="lab-grid">
    <div class="lab-card c2">
      <div class="lab-mono">HR</div>
      <div class="lab-name">Hasan MD Rakibul</div>
      <div class="lab-role">Collaborator</div>
      <div class="lab-desc">Testing, demo preparation, and review across the
      correction engine's development.</div>
    </div>
    <div class="lab-card c3">
      <div class="lab-mono">AA</div>
      <div class="lab-name">Abraheem AbdulSalam</div>
      <div class="lab-role">Collaborator</div>
      <div class="lab-desc">Testing, documentation, and review across the
      sentiment experiments.</div>
    </div>
  </div>

  <div class="lab-foot">BUILT AS NLP COURSEWORK AT ASIA PACIFIC UNIVERSITY ·
  THEN REBUILT, MEASURED, AND EXTENDED AS SCISPELL V2 · 2026</div>
</div>
""", unsafe_allow_html=True)