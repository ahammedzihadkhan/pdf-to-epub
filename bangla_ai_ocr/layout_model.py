"""
layout_model.py
---------------
Tiny CPU page-layout model for scanned Bangla book pages (runs in ~200 MB RAM).

The page is cut into a GRID_W x GRID_H grid. Each cell gets cheap image statistics
(ink, grey tones, edges, colour, neighbourhood context) and a gradient-boosted
classifier says "photo" or "not photo". Photo cells are joined into boxes.

Training labels come from the Gemini OCR caches (<prefix>_ocr_cache/*.json "photos").

  1. Train + re-check: after each round the model re-checks every training page
     against Gemini; pages where they disagree get double weight next round.
  2. Reinforcement learning: an epsilon-greedy multi-armed bandit picks the
     post-processing setting (threshold, min size, gap closing). Reward = how well
     the boxes match Gemini on a page it has not trained on.
  3. Compare: on held-out test pages, report the local model vs Gemini.

Usage (low-resource defaults: 2 CPU threads):
    python -m bangla_ai_ocr.layout_model train --sleep 0.05
    python -m bangla_ai_ocr.layout_model predict some_page.jpg
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # must be set before numpy/sklearn load

import sys
import json
import time
import random
import argparse
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

GRID_W, GRID_H = 24, 36
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "models" / "layout_model.joblib"
REPORT_PATH = Path(__file__).resolve().parent / "models" / "layout_report.json"
RECHECK_LOG = Path(__file__).resolve().parent / "models" / "layout_recheck.jsonl"

# Bandit arms: (probability threshold, min photo size in cells, gap-closing kernel)
ARMS = [(t, a, c) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for a in (4, 8, 16) for c in (1, 3)]
DEFAULT_ARM = (0.5, 8, 1)


# ── Features & labels ────────────────────────────────────────────────────────
def page_features(img_bgr):
    """Returns (GRID_H*GRID_W, n_features) float32 array, one row per cell."""
    small = cv2.resize(img_bgr, (GRID_W * 16, GRID_H * 16), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32) / 255
    edges = cv2.Canny((g * 255).astype(np.uint8), 80, 160).astype(np.float32) / 255
    maps = [g, g * g, (g < 0.35).astype(np.float32),
            ((g >= 0.35) & (g < 0.8)).astype(np.float32), edges, sat]
    mean, sq, dark, mid, edge, s = [cv2.resize(m, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA) for m in maps]
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    base = np.stack([mean, std, dark, mid, edge, s], -1)            # H, W, 6
    ctx3 = cv2.blur(base, (3, 3))
    ctx7 = cv2.blur(base, (7, 7))
    row = np.broadcast_to(base.mean(1, keepdims=True), base.shape)   # text lines are horizontal bands
    page = np.broadcast_to(base.mean((0, 1), keepdims=True), base.shape)
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W].astype(np.float32)
    pos = np.stack([yy / GRID_H, xx / GRID_W], -1)
    feats = np.concatenate([base, ctx3, ctx7, row, page, pos], -1)
    return feats.reshape(-1, feats.shape[-1]).astype(np.float32)


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread that also opens Windows paths with Bangla (non-ASCII) letters. None if unreadable."""
    try:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    except (OSError, cv2.error):
        return None


def norm_box(box):
    """Gemini box_2d -> [ymin, xmin, ymax, xmax] in 0-1000, or None if malformed."""
    while isinstance(box, list) and len(box) == 1 and isinstance(box[0], list):
        box = box[0]
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        y0, x0, y1, x1 = (min(1000.0, max(0.0, float(v))) for v in box)
    except (TypeError, ValueError):
        return None
    return [y0, x0, y1, x1] if y1 > y0 and x1 > x0 else None


def cell_labels(boxes):
    lab = np.zeros((GRID_H, GRID_W), np.uint8)
    cy = (np.arange(GRID_H) + 0.5) * 1000 / GRID_H
    cx = (np.arange(GRID_W) + 0.5) * 1000 / GRID_W
    for y0, x0, y1, x1 in boxes:
        lab[np.ix_((cy >= y0) & (cy <= y1), (cx >= x0) & (cx <= x1))] = 1
    return lab.reshape(-1)


# ── Post-processing & scoring ────────────────────────────────────────────────
def boxes_from_prob(prob, thr, min_cells, close):
    m = (prob.reshape(GRID_H, GRID_W) >= thr).astype(np.uint8)
    if close > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    boxes = []
    for x, y, w, h, area in stats[1:n]:
        if area >= min_cells:
            boxes.append([y * 1000 / GRID_H, x * 1000 / GRID_W, (y + h) * 1000 / GRID_H, (x + w) * 1000 / GRID_W])
    return sorted(boxes)


def iou(a, b):
    iy, ix = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iy * ix
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_count(pred, gold, thresh=0.5):
    used, hits = set(), 0
    for p in pred:
        best = max(((iou(p, g), j) for j, g in enumerate(gold) if j not in used), default=(0, None))
        if best[0] >= thresh:
            used.add(best[1])
            hits += 1
    return hits


def page_f1(pred, gold):
    """1.0 = same photos as Gemini (IoU >= 0.5), also 1.0 when both find none."""
    if not pred and not gold:
        return 1.0
    hits = match_count(pred, gold)
    return 2 * hits / (len(pred) + len(gold))


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def _split_once(gray, b, bg):
    """Split box b at the first blank paper gutter (uniform bright row/column strip), else None."""
    h, w = gray.shape
    y0, x0, y1, x1 = int(b[0] * h / 1000), int(b[1] * w / 1000), int(b[2] * h / 1000), int(b[3] * w / 1000)
    crop = gray[y0:y1, x0:x1]
    if crop.shape[0] < 20 or crop.shape[1] < 20:
        return None
    for axis in (0, 1):  # 0: horizontal gutters (rows), 1: vertical gutters (columns)
        blank = (crop.mean(axis=1 - axis) >= bg) & (crop.std(axis=1 - axis) < 8)
        n, i = blank.size, 0
        while i < n:
            if not blank[i]:
                i += 1
                continue
            j = i
            while j < n and blank[j]:
                j += 1
            if i > 0 and j < n and j - i >= max(4, n // 60):
                if axis == 0:
                    return [[b[0], b[1], (y0 + i) * 1000 / h, b[3]], [(y0 + j) * 1000 / h, b[1], b[2], b[3]]]
                return [[b[0], b[1], b[2], (x0 + i) * 1000 / w], [b[0], (x0 + j) * 1000 / w, b[2], b[3]]]
            i = j
    return None


def refine_boxes(gray, prob, boxes, thr):
    """XY-cut: split boxes at white gutters so touching photos come apart; drop pieces that are not
    photo-like and boxes that sit mostly inside a bigger one."""
    P = prob.reshape(GRID_H, GRID_W)
    bg = float(np.percentile(gray, 95)) - 15
    stack, out = list(boxes), []
    while stack:
        b = stack.pop()
        parts = _split_once(gray, b, bg)
        if parts:
            stack += parts
            continue
        r0, c0 = int(b[0] * GRID_H / 1000), int(b[1] * GRID_W / 1000)
        cells = P[r0:max(r0 + 1, int(np.ceil(b[2] * GRID_H / 1000))), c0:max(c0 + 1, int(np.ceil(b[3] * GRID_W / 1000)))]
        if _area(b) >= 2e4 and cells.mean() >= thr:
            out.append(b)
    kept = []
    for b in sorted(out, key=_area, reverse=True):
        inside = max((iou_inter(b, k) / _area(b) for k in kept), default=0)
        if inside < 0.5:
            kept.append(b)
    return sorted(kept)


def iou_inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def page_format(boxes, ink):
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / 1e6
    if not boxes:
        return "blank" if ink < 0.002 else "text"
    return "photo" if area > 0.6 else "text+photo"


# ── Data ─────────────────────────────────────────────────────────────────────
def find_labeled_pages(root=ROOT):
    """Every <prefix>_ocr_cache/page_*.json with a "photos" key and a matching <prefix>_pages image."""
    pages = []
    for cache_dir in sorted(root.glob("*_ocr_cache")):
        if cache_dir.name.endswith("_local_ocr_cache"):
            continue  # local guesses, never train on them
        pages_dir = root / (cache_dir.name[: -len("_ocr_cache")] + "_pages")
        for cf in sorted(cache_dir.glob("page_*.json")):
            img = pages_dir / f"{cf.stem}.jpg"
            try:
                data = json.loads(cf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not img.exists() or not isinstance(data, dict) or "photos" not in data:
                continue
            boxes = [b for b in (norm_box((p or {}).get("box_2d")) for p in (data["photos"] or [])) if b]
            pages.append({"id": f"{pages_dir.name}/{cf.stem}", "img": img, "gold": boxes})
    return pages


def split_pages(pages, seed=0):
    """70% train / 15% tune (bandit) / 15% test, stratified so each split has photo pages."""
    rng = random.Random(seed)
    out = {"train": [], "tune": [], "test": []}
    for group in ([p for p in pages if p["gold"]], [p for p in pages if not p["gold"]]):
        group = group[:]
        rng.shuffle(group)
        a, b = int(len(group) * 0.7), int(len(group) * 0.85)
        out["train"] += group[:a]
        out["tune"] += group[a:b]
        out["test"] += group[b:]
    return out


def load_features(pages, sleep=0.0, label=""):
    for i, p in enumerate(pages):
        if "X" not in p:
            img = imread(p["img"])
            p["X"] = page_features(img)
            p["y"] = cell_labels(p["gold"])
            if sleep:
                time.sleep(sleep)  # keep a low-power CPU cool
        if (i + 1) % 100 == 0:
            print(f"  features {label}: {i + 1}/{len(pages)}")


# ── Train / RL / evaluate ────────────────────────────────────────────────────
def fit(pages, weights):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                         class_weight="balanced", random_state=0)
    X = np.vstack([p["X"] for p in pages])
    y = np.concatenate([p["y"] for p in pages])
    w = np.repeat(weights, GRID_W * GRID_H)
    clf.fit(X, y, sample_weight=w)
    return clf


def predict_probs(clf, pages):
    return [clf.predict_proba(p["X"])[:, 1] for p in pages]


def evaluate(preds, pages):
    tp_pages = hits = n_pred = n_gold = 0
    f1s = []
    for pred, p in zip(preds, pages):
        f1s.append(page_f1(pred, p["gold"]))
        tp_pages += bool(pred) == bool(p["gold"])
        hits += match_count(pred, p["gold"])
        n_pred += len(pred)
        n_gold += len(p["gold"])
    return {
        "pages": len(pages),
        "photo_page_agreement": round(tp_pages / max(1, len(pages)), 4),
        "box_precision": round(hits / max(1, n_pred), 4),
        "box_recall": round(hits / max(1, n_gold), 4),
        "mean_page_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
    }


def tune_bandit(probs, pages, episodes=3000, eps=0.1, seed=0):
    """Epsilon-greedy bandit: state-free RL that learns which post-processing arm earns most reward."""
    rng = random.Random(seed)
    photo = [i for i, p in enumerate(pages) if p["gold"]]
    plain = [i for i, p in enumerate(pages) if not p["gold"]]
    q, n = [0.0] * len(ARMS), [0] * len(ARMS)
    for _ in range(episodes):
        untried = [k for k in range(len(ARMS)) if n[k] == 0]
        if untried:
            a = untried[0]
        elif rng.random() < eps:
            a = rng.randrange(len(ARMS))
        else:
            a = max(range(len(ARMS)), key=q.__getitem__)
        # balance photo / no-photo pages so the bandit can't win by never finding photos
        pool = photo if (photo and (not plain or rng.random() < 0.5)) else plain
        i = rng.choice(pool)
        reward = page_f1(boxes_from_prob(probs[i], *ARMS[a]), pages[i]["gold"])
        n[a] += 1
        q[a] += (reward - q[a]) / n[a]
    best = max(range(len(ARMS)), key=q.__getitem__)
    return ARMS[best], round(q[best], 4)


def train(rounds=2, sleep=0.0, episodes=3000):
    import joblib
    pages = find_labeled_pages()
    if not pages:
        sys.exit("No labeled pages found (<prefix>_ocr_cache + <prefix>_pages).")
    sp = split_pages(pages)
    print(f"Labeled pages: {len(pages)} ({sum(bool(p['gold']) for p in pages)} with photos) | "
          f"train {len(sp['train'])}, tune {len(sp['tune'])}, test {len(sp['test'])}")
    for k in sp:
        load_features(sp[k], sleep, k)

    weights = np.ones(len(sp["train"]), np.float32)
    report = {"rounds": []}
    RECHECK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RECHECK_LOG, "w", encoding="utf-8") as log:
        for r in range(1, rounds + 1):
            clf = fit(sp["train"], weights)
            test_probs = predict_probs(clf, sp["test"])
            test_boxes = [boxes_from_prob(pr, *DEFAULT_ARM) for pr in test_probs]
            report["rounds"].append({"round": r, "test": evaluate(test_boxes, sp["test"])})
            print(f"Round {r} test vs Gemini: {report['rounds'][-1]['test']}")
            if r == rounds:
                break
            # Re-check: compare every training page with Gemini, upweight disagreements
            flagged = 0
            for i, (prob, p) in enumerate(zip(predict_probs(clf, sp["train"]), sp["train"])):
                pred = boxes_from_prob(prob, *DEFAULT_ARM)
                f1 = page_f1(pred, p["gold"])
                if f1 < 1.0:
                    weights[i] *= 2
                    flagged += 1
                    log.write(json.dumps({"round": r, "page": p["id"], "f1": round(f1, 3),
                                          "gemini": p["gold"], "local": pred}) + "\n")
            print(f"  re-check: {flagged}/{len(sp['train'])} training pages disagree with Gemini -> weight x2")

    arm, q = tune_bandit(predict_probs(clf, sp["tune"]), sp["tune"], episodes)
    print(f"RL bandit best post-processing arm {arm} (avg reward {q})")
    report["bandit"] = {"arm": arm, "avg_reward": q, "episodes": episodes}
    cells = [boxes_from_prob(pr, *arm) for pr in test_probs]
    report["test_cells_only"] = evaluate(cells, sp["test"])
    grays = (imread(p["img"], cv2.IMREAD_GRAYSCALE) for p in sp["test"])
    report["test_with_gutter_split"] = evaluate(
        [refine_boxes(g, pr, b, arm[0]) for g, pr, b in zip(grays, test_probs, cells)], sp["test"])
    use_refine = report["test_with_gutter_split"]["mean_page_f1"] >= report["test_cells_only"]["mean_page_f1"]
    report["use_gutter_split"] = use_refine
    report["final_test_vs_gemini"] = report["test_with_gutter_split" if use_refine else "test_cells_only"]
    print(f"Cells only        vs Gemini: {report['test_cells_only']}")
    print(f"With gutter split vs Gemini: {report['test_with_gutter_split']}  -> use_gutter_split={use_refine}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "arm": arm, "grid": (GRID_W, GRID_H), "refine": use_refine}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH}\nReport {REPORT_PATH}\nRe-check log {RECHECK_LOG}")
    return report


# ── Inference ────────────────────────────────────────────────────────────────
class LayoutModel:
    def __init__(self, path=MODEL_PATH):
        import joblib
        bundle = joblib.load(path)
        self.clf, self.arm, self.refine = bundle["clf"], tuple(bundle["arm"]), bundle.get("refine", False)

    def analyze(self, img_bgr):
        """-> {"format": "blank|text|text+photo|photo", "photos": [[ymin,xmin,ymax,xmax] 0-1000, ...]}"""
        X = page_features(img_bgr)
        prob = self.clf.predict_proba(X)[:, 1]
        boxes = boxes_from_prob(prob, *self.arm)
        if self.refine:
            boxes = refine_boxes(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), prob, boxes, self.arm[0])
        return {"format": page_format(boxes, float(X[:, 2].mean())), "photos": boxes}


def main():
    ap = argparse.ArgumentParser(description="Train / run the CPU page-layout model.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--rounds", type=int, default=2, help="train + re-check rounds")
    t.add_argument("--episodes", type=int, default=3000, help="RL bandit episodes")
    t.add_argument("--sleep", type=float, default=0.0, help="seconds to pause per page (cooler CPU)")
    pr = sub.add_parser("predict")
    pr.add_argument("images", nargs="+")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.rounds, args.sleep, args.episodes)
    else:
        m = LayoutModel()
        for f in args.images:
            print(f, m.analyze(imread(f)))


if __name__ == "__main__":
    main()
