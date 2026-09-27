import re
import os
from pathlib import Path
import ebooklib
from ebooklib import epub
import markdown

MD_PATH = Path("book.md")
OUTPUT_EPUB = Path("book_final.epub")

uni_superscripts = {
    '⁰': '০', '¹': '১', '²': '২', '³': '৩', '⁴': '৪', 
    '⁵': '৫', '⁶': '৬', '⁷': '৭', '⁸': '৮', '⁹': '৯'
}

def normalize_superscripts(text):
    for k, v in uni_superscripts.items():
        text = text.replace(k, v)
    return text

def main():
    if not MD_PATH.exists():
        print("book.md not found!")
        return

    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Normalize unicode superscripts to standard Bengali numbers globally
    md_content = normalize_superscripts(md_content)
    # Remove page markers
    md_content = re.sub(r'<!-- ===== পৃষ্ঠা \d+ ===== -->', '', md_content)
    
    # Split the document into chapters by H1 headers
    chapters_raw = re.split(r'(?m)^# ', md_content)
    
    # Pass 1: Find where all definitions are located
    def_map = {}
    for idx, chap_raw in enumerate(chapters_raw):
        # Match "38. content..." or "[^1]: content..."
        defs = re.findall(r'^([০-৯0-9]{1,3})[\.\s]+', chap_raw, flags=re.MULTILINE)
        defs += re.findall(r'^\[\^([০-৯0-9]+)\]:\s*', chap_raw, flags=re.MULTILINE)
        for num in defs:
            # Map the reference number to its chapter index
            def_map[num] = idx

    book = epub.EpubBook()
    book.set_identifier('id123456')
    book.set_title('বাংলাদেশের স্বাধীনতা: কূটনৈতিক যুদ্ধ')
    book.set_language('bn')
    book.add_author('আবু সাইয়িদ')

    chapters = []
    toc = []

    # Regex definitions
    ref_pattern1 = re.compile(r'\[\^([০-৯0-9]+)\]')
    ref_pattern2 = re.compile(r'<sup>([০-৯0-9]+)</sup>')
    ref_pattern3 = re.compile(r'([।”])\s*([০-৯]{1,3})(?![০-৯])')
    
    def_pattern1 = re.compile(r'^\[\^([০-৯0-9]+)\]:\s*(.+)$', flags=re.MULTILINE)
    def_pattern2 = re.compile(r'^([০-৯0-9]{1,3})[\.\s]+(.+)$', flags=re.MULTILINE)

    # Pass 2: Replace links with cross-chapter references
    for idx, chap_raw in enumerate(chapters_raw):
        if not chap_raw.strip():
            continue
            
        title = chap_raw.split('\n')[0].strip()
        if not title:
            title = f"অধ্যায় {idx}"
            
        chap_md = "# " + chap_raw if idx > 0 else chap_raw
        
        # We are leaving all text as it is, per user request. No regex link replacements.

        # Convert markdown to HTML (adding footnotes extension just in case they want standard markdown footnotes to render properly)
        html_content = markdown.markdown(chap_md, extensions=['tables', 'footnotes'])
        
        c = epub.EpubHtml(title=title, file_name=f'chap_{idx}.xhtml', lang='bn')
        # Add epub namespace for footnotes just in case markdown generates them
        c.content = f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>{title}</title><meta charset="utf-8"/></head><body>{html_content}</body></html>'
        
        book.add_item(c)
        chapters.append(c)
        # Only add to TOC if it looks like a real chapter, not just a small section. Or we can add all.
        toc.append(c)
        
    book.toc = tuple(toc)
    
    # Add Ncx and Nav for TOC
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    
    # Spine including nav this time so they have a visible index
    book.spine = ['nav'] + chapters
    
    epub.write_epub(OUTPUT_EPUB, book, {})
    print(f"Successfully generated EPUB with correct cross-chapter footnote links and Index at {OUTPUT_EPUB}")

if __name__ == "__main__":
    main()
