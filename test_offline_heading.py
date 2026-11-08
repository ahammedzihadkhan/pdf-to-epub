"""Self-check for the offline chapter-title rule.  Run:  python test_offline_heading.py"""
from bangla_pdf_to_epub import _split_heading


def line(y, h, text, x=100, w=600):
    return [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]], text, 0.9]


def body(y0=400, n=8):
    return [line(y0 + 40 * k, 25, f"body {k}") for k in range(n)]


def test_big_short_title_under_page_number():
    head, rest = _split_heading([line(30, 20, "১২", x=390, w=20), line(200, 40, "বাড়ী থেকে পালিয়ে", x=300, w=200)] + body(), 1300)
    assert head == "বাড়ী থেকে পালিয়ে" and len(rest) == 9


def test_title_split_into_two_boxes_joins_left_to_right():
    head, _ = _split_heading([line(200, 40, "পালিয়ে", x=420, w=100), line(202, 40, "বাড়ী থেকে", x=300, w=110)] + body(), 1300)
    assert head == "বাড়ী থেকে পালিয়ে"


def test_body_size_title_that_starts_low():
    page = [line(400, 25, "গল্পের শুরু", x=300, w=150)] + body(470)
    assert _split_heading(page, 1300)[0] == "গল্পের শুরু"
    assert _split_heading(page, 1300, low_start_ok=False)[0] is None  # e.g. a caption under a photo


def test_no_heading_cases():
    assert _split_heading(body(), 1300) == (None, body())
    assert _split_heading([line(80, 25, "বইয়ের নাম", x=300, w=150)] + body(130), 1300)[0] is None  # running header
    assert _split_heading(body() + [line(1000, 60, "x", x=300, w=100)], 1300)[0] is None           # tall line low down
    assert _split_heading([line(200, 40, "ক" * 80, x=300, w=200)] + body(), 1300)[0] is None       # too long
    assert _split_heading([line(200, 40, "ক", x=300, w=200)] + body(n=2), 1300)[0] is None          # too few lines


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("offline heading self-check OK")
