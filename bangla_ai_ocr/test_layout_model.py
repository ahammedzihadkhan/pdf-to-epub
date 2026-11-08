"""Self-check for layout_model.py.  Run:  python -m pytest bangla_ai_ocr/test_layout_model.py  (or python -m bangla_ai_ocr.test_layout_model)"""
import numpy as np
import cv2

from bangla_ai_ocr.layout_model import (
    GRID_W, GRID_H, page_features, norm_box, cell_labels, boxes_from_prob, page_f1, page_format, refine_boxes,
)


def test_norm_box_handles_gemini_quirks():
    assert norm_box([[100, 200, 300, 400]]) == [100, 200, 300, 400]
    assert norm_box([0, 0, 1200, 500]) == [0, 0, 1000, 500]
    assert norm_box([300, 0, 100, 500]) is None
    assert norm_box("x") is None and norm_box(None) is None


def test_labels_round_trip_to_boxes():
    box = [250, 250, 750, 750]
    prob = cell_labels([box]).astype(float)
    pred = boxes_from_prob(prob, 0.5, 4, 1)
    assert len(pred) == 1 and page_f1(pred, [box]) == 1.0


def test_page_f1_and_format():
    assert page_f1([], []) == 1.0
    assert page_f1([[0, 0, 100, 100]], []) == 0.0
    assert page_format([], 0.0) == "blank"
    assert page_format([], 0.05) == "text"
    assert page_format([[0, 0, 900, 900]], 0.05) == "photo"
    assert page_format([[0, 0, 300, 300]], 0.05) == "text+photo"


def test_features_shape():
    img = np.full((1300, 836, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (700, 500), (90, 90, 90), -1)
    X = page_features(img)
    assert X.shape[0] == GRID_W * GRID_H and np.isfinite(X).all()


def test_gutter_split_separates_touching_photos():
    gray = np.full((1000, 1000), 250, np.uint8)
    rng = np.random.default_rng(0)
    gray[100:400, 100:900] = rng.integers(40, 200, (300, 800))   # photo A
    gray[430:800, 100:900] = rng.integers(40, 200, (370, 800))   # photo B, 30px white gutter
    merged = [[100, 100, 800, 900]]
    prob = cell_labels(merged).astype(float)
    out = refine_boxes(gray, prob, merged, 0.5)
    assert page_f1(out, [[100, 100, 400, 900], [430, 100, 800, 900]]) == 1.0
    # a box fully inside a bigger one is dropped
    assert len(refine_boxes(gray, prob, [[100, 100, 400, 900], [150, 150, 300, 500]], 0.5)) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("layout_model self-check OK")
