"""Self-check for text_corrector.py.  Run:  python -m bangla_ai_ocr.test_text_corrector"""
from bangla_ai_ocr.text_corrector import TextCorrector, clean, split_word, learn_subs, cer


def test_clean_and_split():
    assert clean("[PHOTO_0]\n\n**কিস্তি : ১**  &amp; ক") == "কিস্তি : ১ & ক"
    assert split_word("“যাত্রী,”") == ("“", "যাত্রী", ",”")
    assert split_word("।") == ("।", "", "")


def test_learns_repeated_misread_but_not_usually_right_words():
    rows = [{"easyocr": "আমি যত্রী হলাম", "gemini": "আমি যাত্রী হলাম"}] * 3 + \
           [{"easyocr": "সে দিন", "gemini": "সেই দিন"}, {"easyocr": "সে দিন", "gemini": "সে দিন"}, {"easyocr": "সে দিন", "gemini": "সে দিন"}]
    assert learn_subs(rows, 2, 0.7) == {"যত্রী": "যাত্রী"}


def test_fix_keeps_punctuation_and_uses_vocab():
    tc = TextCorrector({"subs": {"যত্রী": "যাত্রী"}, "vocab": {"কৈলাস": 50, "পাহাড়": 40}})
    assert tc.fix("“যত্রী,” কৈলস\nপাহাড়।") == "“যাত্রী,” কৈলাস\nপাহাড়।"
    assert tc.fix("অজানাশব্দ") == "অজানাশব্দ"          # nothing close: unchanged
    assert TextCorrector({}).fix("যত্রী  ক") == "যত্রী  ক"  # empty model is identity


def test_cer():
    assert cer("abc", "abc") == 0 and cer("abd", "abc") == 1 / 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("text_corrector self-check OK")
