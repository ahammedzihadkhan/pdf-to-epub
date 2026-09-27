"""
FlipHTML5 Book Scraper - High-Resolution Screenshot to PDF
Book : Liberation War Bangladesh - Kutnitiik Juddho
URL  : https://online.fliphtml5.com/lzrut/jfhq/
"""

import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from PIL import Image
from playwright.async_api import async_playwright

# ── CONFIG ───────────────────────────────────────────────────────────────────
BOOK_BASE  = "https://online.fliphtml5.com/lzrut/jfhq/"
OUTPUT_DIR = Path(__file__).parent / "pages"
PDF_OUT    = Path(__file__).parent / "book.pdf"

VIEWPORT_W   = 1920
VIEWPORT_H   = 1080
DEVICE_SCALE = 2          # 2× = 3840×2160 effective pixels per shot
LOAD_WAIT_MS = 5000       # ms to wait after each page navigation
# ─────────────────────────────────────────────────────────────────────────────


async def detect_total_pages(page) -> int:
    """
    Load page 1 of the FlipHTML5 viewer and extract the total page count
    from the DOM using multiple detection strategies.
    """
    print("  Opening book to read total page count …")
    await page.goto(BOOK_BASE + "#p=1", wait_until="networkidle", timeout=60000)
    await page.wait_for_timeout(6000)

    total = await page.evaluate("""
    () => {
        // ── Strategy 1: common FlipHTML5 class/id selectors ──────────────
        const sels = [
            '.total-page', '.totalPage', '#total_page', '#totalPage',
            '[class*="total"][class*="page"]', '[id*="total"][id*="page"]',
            '.page-count', '.pageCount', '#pageCount',
        ];
        for (const s of sels) {
            const el = document.querySelector(s);
            if (el) {
                const n = parseInt(el.innerText.replace(/[^0-9]/g, ''), 10);
                if (n > 0) return n;
            }
        }

        // ── Strategy 2: scan every element for  "/ NNN" pattern ──────────
        for (const el of document.querySelectorAll('*')) {
            if (el.children.length > 5) continue;          // skip containers
            const t = (el.innerText || '').trim();
            const m = t.match(/^\\/\\s*(\\d+)$/) || t.match(/(\\d+)\\s*\\/\\s*(\\d+)/);
            if (m) {
                const n = parseInt(m[m.length - 1], 10);
                if (n > 1) return n;
            }
        }

        // ── Strategy 3: check global JS variables ─────────────────────────
        const candidates = [
            window.totalPage, window.pageNum, window.total_page,
            window.bookConfig && window.bookConfig.totalPage,
            window.BOOK_CONFIG && window.BOOK_CONFIG.totalPage,
        ];
        for (const c of candidates) {
            const n = parseInt(c, 10);
            if (n > 0) return n;
        }

        // ── Strategy 4: count <img> tags whose src looks like page images ─
        const imgs = [...document.querySelectorAll('img')]
            .map(i => i.src)
            .filter(s => /\\/(\\d+)\\.jpg/i.test(s));
        if (imgs.length > 0) {
            const nums = imgs.map(s => parseInt(s.match(/\\/(\\d+)\\.jpg/i)[1], 10));
            return Math.max(...nums);
        }

        return null;
    }
    """)

    if total and total > 0:
        return total

    # Strategy 5: try navigating to a huge page number and see what the
    # viewer redirects to — FlipHTML5 caps to last page
    print("  DOM detection failed. Trying navigation probe …")
    await page.goto(BOOK_BASE + "#p=9999", wait_until="networkidle", timeout=30000)
    await page.wait_for_timeout(3000)

    total = await page.evaluate("""
    () => {
        // After clamping, look for current page indicator
        const sels = [
            '.current-page', '.currentPage', '#current_page',
            '[class*="current"][class*="page"]',
        ];
        for (const s of sels) {
            const el = document.querySelector(s);
            if (el) {
                const n = parseInt(el.innerText.replace(/[^0-9]/g, ''), 10);
                if (n > 1) return n;
            }
        }
        // Scan for biggest visible page number
        let max = 0;
        for (const el of document.querySelectorAll('*')) {
            if (el.children.length > 3) continue;
            const n = parseInt((el.innerText || '').trim(), 10);
            if (n > max && n < 10000) max = n;
        }
        return max > 1 ? max : null;
    }
    """)

    return total


async def screenshot_one_page(page, num: int, out_path: Path):
    """Navigate to page `num` and save a high-res JPEG screenshot."""
    await page.goto(BOOK_BASE + f"#p={num}",
                    wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(LOAD_WAIT_MS)

    # Hide all UI chrome so only the book content is captured
    await page.evaluate("""
    () => {
        const selectors = [
            '.flipbook-navbar', '.navbar', '.toolbar', '#toolbar',
            '.page-nav', '.page-navigation', '.page-number-wrap',
            '#page-number', '.top-bar', '.bottom-bar',
            '.btn-toc', '.hp-menu', '.share-btn', '.zoom-btn',
            '#loading', '.loading', '.ui-dialog',
            '.page-turn-btn', '.turn-page-btn',
        ];
        selectors.forEach(s =>
            document.querySelectorAll(s).forEach(el => {
                el.style.setProperty('display', 'none', 'important');
            })
        );
    }
    """)
    await page.wait_for_timeout(400)

    # Try to find the book canvas / container for tight cropping
    clip = None
    for sel in [
        '#flipbook-container', '.flipbook-container',
        '#canvas', '.canvas', '.magazine',
        '#book', '.book', '.flipbook', '#flipbook',
        'canvas',
    ]:
        el = await page.query_selector(sel)
        if el:
            box = await el.bounding_box()
            if box and box['width'] > 200 and box['height'] > 200:
                clip = {k: box[k] for k in ('x', 'y', 'width', 'height')}
                break

    await page.screenshot(
        path=str(out_path),
        full_page=(clip is None),
        clip=clip,
        type='jpeg',
        quality=95,
    )


async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                '--ignore-certificate-errors',
                '--disable-web-security',
                '--allow-running-insecure-content',
            ]
        )
        ctx = await browser.new_context(
            viewport={'width': VIEWPORT_W, 'height': VIEWPORT_H},
            device_scale_factor=DEVICE_SCALE,
            ignore_https_errors=True,
        )
        page = await ctx.new_page()

        # ── Step 1: detect total pages ────────────────────────────────────
        print("=" * 60)
        print("STEP 1 — Detecting total pages")
        print("=" * 60)

        total = await detect_total_pages(page)

        if not (total and total > 0):
            print("[!] Auto-detection failed. Defaulting to 400 pages.")
            total = 400

        print(f"    -> Total pages: {total}\n")

        # ── Step 2: screenshot each page ─────────────────────────────────
        print("=" * 60)
        print("STEP 2 — Capturing pages")
        print("=" * 60)

        failed = []
        for p in range(1, total + 1):
            out = OUTPUT_DIR / f"page_{p:04d}.jpg"
            if out.exists():
                print(f"  SKIP {p:04d}/{total}  (already saved)")
                continue
            try:
                await screenshot_one_page(page, p, out)
                size_kb = out.stat().st_size // 1024
                print(f"  OK   {p:04d}/{total}  ({size_kb} KB)")
            except Exception as e:
                print(f"  FAIL {p:04d}/{total}  {e}")
                failed.append(p)

        await browser.close()

    if failed:
        print(f"\n[!] {len(failed)} pages failed: {failed}")
        print("    Re-run the script to retry them.\n")

    # ── Step 3: combine into PDF ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 3 — Building PDF")
    print("=" * 60)

    jpg_files = sorted(OUTPUT_DIR.glob("page_*.jpg"))
    if not jpg_files:
        print("[!] No images found – nothing to combine.")
        sys.exit(1)

    print(f"  Loading {len(jpg_files)} images …")
    images = []
    for f in jpg_files:
        try:
            images.append(Image.open(f).convert("RGB"))
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")

    if not images:
        print("[!] All images failed to open.")
        sys.exit(1)

    images[0].save(
        PDF_OUT,
        save_all=True,
        append_images=images[1:],
    )

    mb = PDF_OUT.stat().st_size / 1_048_576
    print(f"\n  PDF saved → {PDF_OUT}")
    print(f"  Pages : {len(images)}")
    print(f"  Size  : {mb:.1f} MB")
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
