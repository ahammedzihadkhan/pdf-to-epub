"""Offline text corrector learned from page pairs: (offline EasyOCR text, Gemini text).

  python -m bangla_ai_ocr.text_corrector collect --per-book 20 --sleep 0.5   # slow: ~1 min/page on CPU, resumable
  python -m bangla_ai_ocr.text_corrector train                               # seconds

Learns two things and keeps them only if they lower CER on held-out pages:
  1. word substitutions EasyOCR keeps getting wrong (aligned against Gemini text)
  2. a vocabulary of correct words: an unknown word one edit away from a common word is replaced
"""
import argparse
import html
import json
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(__file__).resolve().parent / "models"
PAIRS = MODELS / "ocr_pairs.jsonl"
MODEL_PATH = MODELS / "text_corrector.json"
REPORT = MODELS / "text_corrector_report.json"
WORD = re.compile(r"\S+")
PUNCT = "\"'“”‘’()[]{}.,;:!?।॥—–-…*"


def clean(text):
    """Page markdown -> plain comparable text."""
    text = html.unescape(text or "")
    text = re.sub(r"\[PHOTO_\d+\]|\*\*|^#+\s*", "", text, flags=re.M)
    return unicodedata.normalize("NFC", re.sub(r"[ \t]+", " ", text)).strip()


def split_word(tok):
    core = tok.strip(PUNCT)
    if not core:
        return tok, "", ""
    i = tok.index(core)
    return tok[:i], core, tok[i + len(core):]


def cores(text):
    return [c for c in (split_word(t)[1] for t in WORD.findall(text)) if c]


def cer(pred, gold):
    gold, pred = re.sub(r"\s+", " ", gold), re.sub(r"\s+", " ", pred)
    return Levenshtein.distance(pred, gold) / max(1, len(gold))


def deletes(w):
    return {w[:i] + w[i + 1:] for i in range(len(w))}


class TextCorrector:
    def __init__(self, cfg=None, path=MODEL_PATH):
        cfg = cfg if cfg is not None else json.loads(Path(path).read_text(encoding="utf-8"))
        self.subs, self.vocab = cfg.get("subs", {}), cfg.get("vocab", {})
        self.idx = defaultdict(list)  # delete-variant -> vocab words, for edit-distance-1 lookup
        for v in self.vocab:
            for d in deletes(v):
                self.idx[d].append(v)

    def word(self, w):
        if w in self.subs:
            return self.subs[w]
        if not self.vocab or w in self.vocab or len(w) < 3:
            return w
        cands = set(self.idx.get(w, ()))
        for d in deletes(w):
            cands.update(self.idx.get(d, ()))
            if d in self.vocab:
                cands.add(d)
        cands = sorted((v for v in cands if Levenshtein.distance(v, w) == 1), key=self.vocab.get, reverse=True)
        if not cands or (len(cands) > 1 and self.vocab[cands[0]] < 2 * self.vocab[cands[1]]):
            return w  # no clear winner: leave it
        return cands[0]

    def fix(self, text):
        def rep(m):
            pre, core, post = split_word(m.group(0))
            return pre + self.word(unicodedata.normalize("NFC", core)) + post if core else m.group(0)
        return WORD.sub(rep, text)


def learn_subs(rows, min_count, min_prob):
    seen = defaultdict(Counter)
    for r in rows:
        a, b = cores(r["easyocr"]), cores(r["gemini"])
        for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
                for x, y in zip(a[i1:i2], b[j1:j2]):
                    seen[x][y] += 1
    subs = {}
    for x, c in seen.items():
        y, n = c.most_common(1)[0]
        # skip words EasyOCR usually gets right, and pairs too different to be a misread
        if y != x and n >= min_count and n / sum(c.values()) >= min_prob and Levenshtein.normalized_distance(x, y) <= 0.5:
            subs[x] = y
    return subs


def read_pairs():
    return [json.loads(l) for l in PAIRS.open(encoding="utf-8")] if PAIRS.exists() else []


def gemini_pages():
    for cache in sorted(ROOT.glob("*_ocr_cache")):
        if cache.name.endswith("_local_ocr_cache"):
            continue
        book = cache.name[:-len("_ocr_cache")]
        for p in sorted(cache.glob("page_*.json")):
            if p.stem == "page_0000":
                continue  # cover
            try:
                yield book, p.stem, clean(json.loads(p.read_text(encoding="utf-8")).get("content_markdown"))
            except (ValueError, OSError):
                continue


def collect(per_book=20, sleep=0.5, threads=2, seed=0):
    import cv2
    import torch
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    sys.path.insert(0, str(ROOT))
    import bangla_pdf_to_epub as app
    from bangla_ai_ocr.layout_model import LayoutModel, imread
    from bangla_ai_ocr.hybrid_engine import HybridBanglaOCR

    MODELS.mkdir(exist_ok=True)
    done = {(r["book"], r["page"]) for r in read_pairs()}
    by_book = defaultdict(list)
    for book, stem, gold in gemini_pages():
        if len(gold) >= 200 and (ROOT / f"{book}_pages" / f"{stem}.jpg").exists():
            by_book[book].append((stem, gold))
    rng, model, engine = random.Random(seed), LayoutModel(), None
    todo = [(b, s, g) for b in sorted(by_book) for s, g in rng.sample(by_book[b], min(per_book, len(by_book[b])))]
    for k, (book, stem, gold) in enumerate(todo, 1):
        if (book, stem) in done:
            continue
        img = imread(ROOT / f"{book}_pages" / f"{stem}.jpg")
        engine = engine or HybridBanglaOCR()
        text = clean(app._local_page_json(engine, img, model.analyze(img)["photos"])["content_markdown"])
        with PAIRS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"book": book, "page": stem, "easyocr": text, "gemini": gold}, ensure_ascii=False) + "\n")
        print(f"  [{k}/{len(todo)}] {book}/{stem}  CER {cer(text, gold):.3f}", flush=True)
        time.sleep(sleep)  # keep a low-power CPU cool


def train(seed=0):
    rows = read_pairs()
    if len(rows) < 10:
        sys.exit(f"Need at least 10 page pairs, have {len(rows)}. Run: python -m bangla_ai_ocr.text_corrector collect")
    random.Random(seed).shuffle(rows)
    n = len(rows)
    tr, tu, te = rows[:int(.6 * n)], rows[int(.6 * n):int(.8 * n)], rows[int(.8 * n):]
    held = {(r["book"], r["page"]) for r in tu + te}
    vocab = Counter()
    for book, stem, gold in gemini_pages():
        if (book, stem) not in held:  # vocabulary never sees tune/test pages
            vocab.update(cores(gold))

    def score(cfg, rs):
        tc = TextCorrector(cfg)
        return statistics.mean(cer(tc.fix(r["easyocr"]), r["gemini"]) for r in rs)

    def make(subs_rows, mc, mp, mf):
        return {"subs": learn_subs(subs_rows, mc, mp), "vocab": {w: c for w, c in vocab.items() if c >= mf} if mf else {},
                "params": {"min_count": mc, "min_prob": mp, "min_vocab_freq": mf}}

    identity = {"subs": {}, "vocab": {}}
    grid = [(score(make(tr, mc, mp, mf), tu), mc, mp, mf)
            for mc in (1, 2, 3) for mp in (0.5, 0.7, 0.9) for mf in (0, 3, 10)]
    best_tune, mc, mp, mf = min(grid)
    final = make(tr + tu, mc, mp, mf)  # refit on train+tune with the chosen settings
    base_test, fixed_test = score(identity, te), score(final, te)
    keep = fixed_test < base_test
    MODEL_PATH.write_text(json.dumps(final if keep else identity, ensure_ascii=False), encoding="utf-8")
    report = {"pages": {"train": len(tr), "tune": len(tu), "test": len(te)},
              "chosen": final["params"], "kept": keep, "substitutions": len(final["subs"]), "vocab_words": len(final["vocab"]),
              "tune_cer": {"raw": round(score(identity, tu), 4), "corrected": round(best_tune, 4)},
              "test_cer": {"raw": round(base_test, 4), "corrected": round(fixed_test, 4)},
              "examples": dict(list(sorted(final["subs"].items(), key=lambda kv: len(kv[0])))[:20])}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--per-book", type=int, default=20)
    c.add_argument("--sleep", type=float, default=0.5)
    c.add_argument("--threads", type=int, default=2)
    sub.add_parser("train")
    a = ap.parse_args()
    collect(a.per_book, a.sleep, a.threads) if a.cmd == "collect" else train()
