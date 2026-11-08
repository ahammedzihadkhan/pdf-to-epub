"""Bangla scanned PDF -> EPUB desktop app. Offline by default; Gemini is optional.

Run:  python pdf_to_epub_gui.py
"""
import json
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk


class QueueWriter:
    """Copies stdout/stderr into the Log tab. A windowed .exe has no console, so the original may be None."""
    def __init__(self, q, orig):
        self.q, self.orig = q, orig

    def write(self, s):
        if self.orig:
            try:
                self.orig.write(s)
            except Exception:
                pass
        if s and "\r" not in s:  # skip tqdm progress redraws
            self.q.put(("log", s))

    def flush(self):
        if self.orig:
            try:
                self.orig.flush()
            except Exception:
                pass


EVENTS = queue.Queue()  # worker thread -> UI thread
sys.stdout, sys.stderr = QueueWriter(EVENTS, sys.stdout), QueueWriter(EVENTS, sys.stderr)

import bangla_pdf_to_epub as pipeline  # noqa: E402  (imported after stdout is redirected)
from bangla_ai_ocr.layout_model import norm_box  # noqa: E402


class StopRequested(Exception):
    pass


STAGES = {"0": "Reading PDF pages", "1": "Reading text with Gemini", "2": "Finding photos and reading text",
          "3": "Cropping photos", "6": "Building the EPUB"}
STEPS = ["Pages", "Text & photos", "Crop photos", "EPUB"]
STEP_OF = {"0": 0, "1": 1, "2": 1, "3": 2, "6": 3}
HINTS = {"Offline": "Runs on this PC with no internet. About 1 minute per page on a laptop CPU. "
                    "Uses the trained layout model and text corrector.",
         "Gemini": "Sends page images to Google Gemini. Faster, more accurate text. Needs an API key and internet."}
BN = "Nirmala UI"  # Windows font with Bangla glyphs
PHOTO_RGB = (230, 57, 70)
# (light, dark) colour pairs
BG, CARD, SIDE = ("#eef2f7", "#10141b"), ("#ffffff", "#1a2029"), ("#f8fafc", "#151a22")
ACCENT, ACCENT_HI = ("#4f46e5", "#6366f1"), ("#4338ca", "#4f46e5")
OK, OK_HI = ("#16a34a", "#22c55e"), ("#15803d", "#16a34a")
WARN, WARN_HI = ("#d97706", "#f59e0b"), ("#b45309", "#d97706")
DANGER, DANGER_HI = ("#dc2626", "#ef4444"), ("#b91c1c", "#dc2626")
MUTED, OFF = ("gray40", "gray62"), ("#e2e8f0", "#2a3240")
TEXT = ("gray10", "gray90")
FIELD = dict(height=36, corner_radius=8, border_width=1)

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def F(size=13, bold=False):
    return ctk.CTkFont(family=BN, size=size, weight="bold" if bold else "normal")


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


class PdfToEpubApp(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=BG)
        self.title("Bangla PDF to EPUB")
        self.geometry("1180x760")
        self.minsize(920, 580)
        if self.winfo_screenheight() < 900:
            self.after(50, lambda: self.state("zoomed"))  # fill small screens instead of running off the bottom
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.running = self.active = False  # active: preview shows the current book's page folder
        self.page_idx, self.epub_path = 0, None
        self.run_t0 = self.stage_t0 = 0.0
        self.pause_t = None  # when the run was paused; None while running
        self.stage_text = "Working…"
        self.unpaused, self.stop_req = threading.Event(), threading.Event()
        self._pdf_preview, self._pdf_pages, self._render_key, self._tk_img = None, 0, None, None
        self._build_sidebar()
        self._build_main()
        self.bind("<Left>", lambda e: self._key_step(-1))
        self.bind("<Right>", lambda e: self._key_step(1))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(200, self._drain)
        self.after(1000, self._tick)

    # ── layout helpers ───────────────────────────────────────────────────────
    def _card(self, parent, row, num, title):
        c = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12)
        c.grid(row=row, column=0, sticky="ew", padx=14, pady=6)
        c.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(c, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(head, text=num, width=24, height=24, corner_radius=12, fg_color=ACCENT, text_color="white",
                     font=F(12, True)).pack(side="left")
        ctk.CTkLabel(head, text=title, font=F(14, True)).pack(side="left", padx=8)
        return c

    def _slider(self, parent, row, label, lo, hi, steps, value, fmt):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=3)
        f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f, text=label, width=124, anchor="w", font=F(12)).grid(row=0, column=0)
        val = ctk.CTkLabel(f, text=fmt.format(value), width=40, font=F(12, True), text_color=ACCENT)
        s = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=steps, button_color=ACCENT, button_hover_color=ACCENT_HI,
                          progress_color=ACCENT, command=lambda v: val.configure(text=fmt.format(v)))
        s.set(value)
        s.grid(row=0, column=1, sticky="ew")
        val.grid(row=0, column=2)
        return s

    def _hint(self, parent, row, text):
        lbl = ctk.CTkLabel(parent, text=text, text_color=MUTED, anchor="w", justify="left", wraplength=280, font=F(11))
        lbl.grid(row=row, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="ew")
        return lbl

    def _button(self, parent, text, color, hover, command, **kw):
        b = ctk.CTkButton(parent, text=text, fg_color=color, hover_color=hover, corner_radius=10,
                          font=F(13, True), command=command, **kw)
        b.on_color = color
        return b

    def _enable(self, btn, on):
        btn.configure(state="normal" if on else "disabled", fg_color=btn.on_color if on else OFF)

    def _build_sidebar(self):
        left = ctk.CTkFrame(self, corner_radius=0, fg_color=SIDE)
        left.grid(row=0, column=0, sticky="nsw")
        left.grid_rowconfigure(0, weight=1)
        sb = ctk.CTkScrollableFrame(left, width=330, corner_radius=0, fg_color=SIDE)  # settings scroll on short screens
        sb.grid(row=0, column=0, sticky="ns")
        sb.grid_columnconfigure(0, weight=1)
        hero = ctk.CTkFrame(sb, fg_color=ACCENT, corner_radius=14)
        hero.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        ctk.CTkLabel(hero, text="📖  বাংলা PDF → EPUB", font=F(21, True), text_color="white").pack(
            anchor="w", padx=16, pady=(14, 0))
        ctk.CTkLabel(hero, text="Scanned book in, e-book out. Page 1 becomes the cover.", font=F(12),
                     text_color="#e0e7ff", wraplength=280, justify="left").pack(anchor="w", padx=16, pady=(2, 14))

        book = self._card(sb, 1, "1", "Book")
        self.pdf_entry = ctk.CTkEntry(book, placeholder_text="Choose a scanned PDF…", font=F(12), **FIELD)
        self.pdf_entry.grid(row=1, column=0, padx=(12, 6), pady=3, sticky="ew")
        self._button(book, "📂 Browse", ACCENT, ACCENT_HI, self.browse_pdf, width=92, height=36).grid(
            row=1, column=1, padx=(0, 12))
        self.epub_entry = ctk.CTkEntry(book, placeholder_text="Save EPUB as…", font=F(12), **FIELD)
        self.epub_entry.grid(row=2, column=0, padx=(12, 6), pady=3, sticky="ew")
        self._button(book, "💾 Save as", ACCENT, ACCENT_HI, self.browse_epub, width=92, height=36).grid(
            row=2, column=1, padx=(0, 12))
        self.title_entry = ctk.CTkEntry(book, placeholder_text="Book title (optional)", font=F(13), **FIELD)
        self.title_entry.grid(row=3, column=0, columnspan=2, padx=12, pady=3, sticky="ew")
        self.author_entry = ctk.CTkEntry(book, placeholder_text="Author (optional)", font=F(13), **FIELD)
        self.author_entry.grid(row=4, column=0, columnspan=2, padx=12, pady=(3, 12), sticky="ew")

        reader = self._card(sb, 2, "2", "Text reader")
        self.mode = ctk.CTkSegmentedButton(reader, values=["Offline", "Gemini"], command=self._mode_changed, height=34,
                                           font=F(13, True), selected_color=ACCENT, selected_hover_color=ACCENT_HI)
        self.mode.set("Offline")
        self.mode.grid(row=1, column=0, columnspan=2, padx=12, pady=4, sticky="ew")
        self.mode_hint = self._hint(reader, 2, "")
        self.api_entry = ctk.CTkEntry(reader, placeholder_text="Gemini API key", show="•", **FIELD)
        self.api_entry.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="ew")
        key = pipeline.get_api_key()
        if key and key != "YOUR_API_KEY":
            self.api_entry.insert(0, key)  # only sent when the Gemini mode is chosen

        load = self._card(sb, 3, "3", "PC load")
        self.threads = self._slider(load, 1, "CPU threads", 1, 4, 3, 2, "{:.0f}")
        self.cool = self._slider(load, 2, "Rest per page (s)", 0, 2, 8, 0.5, "{:.2f}")
        self._hint(load, 3, "Fewer threads and longer rests keep a laptop cool, but run slower.")

        ctl = ctk.CTkFrame(left, fg_color="transparent")  # run controls stay visible, never scrolled away
        ctl.grid(row=1, column=0, padx=14, pady=(6, 12), sticky="ew")
        ctl.grid_columnconfigure((0, 1), weight=1)
        self.start_btn = self._button(ctl, "▶  Start conversion", ACCENT, ACCENT_HI, self.start_conversion, height=46)
        self.start_btn.configure(font=F(15, True))
        self.start_btn.grid(row=0, column=0, columnspan=2, pady=(0, 8), sticky="ew")
        self.pause_btn = self._button(ctl, "❚❚  Pause", WARN, WARN_HI, self.toggle_pause, height=38)
        self.pause_btn.grid(row=1, column=0, padx=(0, 4), sticky="ew")
        self.stop_btn = self._button(ctl, "■  Stop", DANGER, DANGER_HI, self.stop_conversion, height=38)
        self.stop_btn.grid(row=1, column=1, padx=(4, 0), sticky="ew")
        self.open_btn = self._button(ctl, "📘  Open EPUB", OK, OK_HI, lambda: self._open(self.epub_path), height=38)
        self.open_btn.grid(row=2, column=0, padx=(0, 4), pady=(8, 0), sticky="ew")
        self.folder_btn = self._button(ctl, "📁  Open folder", OK, OK_HI,
                                       lambda: self._open(os.path.dirname(os.path.abspath(self.epub_path))), height=38)
        self.folder_btn.grid(row=2, column=1, padx=(4, 0), pady=(8, 0), sticky="ew")
        for b in (self.pause_btn, self.stop_btn, self.open_btn, self.folder_btn):
            self._enable(b, False)
        foot = ctk.CTkFrame(sb, fg_color="transparent")
        foot.grid(row=4, column=0, padx=14, pady=(10, 14), sticky="ew")
        ctk.CTkLabel(foot, text="Finished pages are saved. Stop or close at any time; the next run resumes.",
                     text_color=MUTED, anchor="w", justify="left", wraplength=290, font=F(11)).pack(anchor="w")
        theme = ctk.CTkOptionMenu(foot, values=["System", "Light", "Dark"], width=110, fg_color=ACCENT,
                                  button_color=ACCENT_HI,
                                  command=lambda m: (ctk.set_appearance_mode(m), self.render(force=True)))
        theme.pack(anchor="w", pady=(8, 0))
        self._mode_changed("Offline")

    def _tile(self, parent, col, caption):
        t = ctk.CTkFrame(parent, fg_color=OFF, corner_radius=10)
        t.grid(row=0, column=col, sticky="ew", padx=4)
        ctk.CTkLabel(t, text=caption, font=F(11), text_color=MUTED).pack(anchor="w", padx=12, pady=(6, 0))
        val = ctk.CTkLabel(t, text="—", font=F(16, True))
        val.pack(anchor="w", padx=12, pady=(0, 6))
        return val

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=16, pady=14)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=14)
        card.grid(row=0, column=0, sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        self.stage_label = ctk.CTkLabel(card, text="Ready", font=F(18, True), anchor="w")
        self.stage_label.grid(row=0, column=0, padx=16, pady=(12, 0), sticky="w")
        self.pct_label = ctk.CTkLabel(card, text="", font=F(26, True), text_color=ACCENT)
        self.pct_label.grid(row=0, column=1, rowspan=2, padx=16, pady=(8, 0), sticky="e")
        steps = ctk.CTkFrame(card, fg_color="transparent")
        steps.grid(row=1, column=0, padx=12, pady=(6, 0), sticky="w")
        self.step_lbls = []
        for i, name in enumerate(STEPS):
            lbl = ctk.CTkLabel(steps, text=f"{i + 1}  {name}", corner_radius=13, height=26, width=116,
                               fg_color=OFF, text_color=MUTED, font=F(12, True))
            lbl.grid(row=0, column=i, padx=4)
            self.step_lbls.append(lbl)
        self.progress = ctk.CTkProgressBar(card, height=14, corner_radius=7, progress_color=ACCENT, fg_color=OFF)
        self.progress.set(0)
        self.progress.grid(row=2, column=0, columnspan=2, padx=16, pady=(12, 10), sticky="ew")
        stats = ctk.CTkFrame(card, fg_color="transparent")
        stats.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 14), sticky="ew")
        stats.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.t_pages, self.t_elapsed = self._tile(stats, 0, "Pages done"), self._tile(stats, 1, "Elapsed")
        self.t_left, self.t_speed = self._tile(stats, 2, "Time left"), self._tile(stats, 3, "Speed")

        tabs = ctk.CTkTabview(main, fg_color=CARD, corner_radius=14, segmented_button_selected_color=ACCENT,
                              segmented_button_selected_hover_color=ACCENT_HI)
        tabs.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        prev, log = tabs.add("Preview"), tabs.add("Log")
        prev.grid_columnconfigure(0, weight=1)
        prev.grid_rowconfigure(1, weight=1)
        self.page_info = ctk.CTkLabel(prev, text="Choose a PDF to see its pages here.", font=F(14, True), anchor="w")
        self.page_info.grid(row=0, column=0, sticky="ew", padx=6)
        self.canvas = tk.Canvas(prev, highlightthickness=0, bd=0, cursor="hand2")  # a Canvas never resizes to its image
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=6)
        self.canvas.bind("<Button-1>", lambda e: self.open_page_image())
        self.canvas.bind("<Configure>", lambda e: self.render(force=True))
        nav = ctk.CTkFrame(prev, fg_color="transparent")
        nav.grid(row=2, column=0, sticky="ew")
        nav.grid_columnconfigure(1, weight=1)
        self._button(nav, "◀", ACCENT, ACCENT_HI, lambda: self.step(-1), width=40).grid(row=0, column=0)
        self.page_slider = ctk.CTkSlider(nav, from_=0, to=1, number_of_steps=1, command=self._slid, button_color=ACCENT,
                                         button_hover_color=ACCENT_HI, progress_color=ACCENT)
        self.page_slider.set(0)
        self.page_slider.grid(row=0, column=1, sticky="ew", padx=8)
        self._button(nav, "▶", ACCENT, ACCENT_HI, lambda: self.step(1), width=40).grid(row=0, column=2)
        self.follow = ctk.CTkSwitch(nav, text="Follow live", progress_color=ACCENT, font=F(12))
        self.follow.select()
        self.follow.grid(row=0, column=3, padx=(12, 0))
        ctk.CTkLabel(prev, text="Red boxes are the photos found on each page. Click a page to open it full size. "
                                "Use the ← → keys to flip pages.", text_color=MUTED, anchor="w", font=F(11)).grid(
            row=3, column=0, sticky="ew", padx=6)
        self.log_box = ctk.CTkTextbox(log, font=ctk.CTkFont(family="Consolas", size=12), state="disabled",
                                      corner_radius=10)
        self.log_box.pack(fill="both", expand=True)

    def _set_step(self, cur):
        for i, lbl in enumerate(self.step_lbls):
            if i < cur:
                lbl.configure(fg_color=OK, text_color="white", text=f"✓  {STEPS[i]}")
            else:
                lbl.configure(fg_color=ACCENT if i == cur else OFF, text_color="white" if i == cur else MUTED,
                              text=f"{i + 1}  {STEPS[i]}")

    def _reset_stats(self):
        self.progress.set(0)
        self.pct_label.configure(text="0%")
        for t in (self.t_pages, self.t_left, self.t_speed):
            t.configure(text="—")

    # ── inputs ───────────────────────────────────────────────────────────────
    def _mode_changed(self, mode):
        self.mode_hint.configure(text=HINTS[mode])
        self.api_entry.grid() if mode == "Gemini" else self.api_entry.grid_remove()

    def browse_pdf(self):
        f = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if f:
            self.load_pdf(f)

    def load_pdf(self, f):
        for entry, value in ((self.pdf_entry, f), (self.epub_entry, os.path.splitext(f)[0] + ".epub")):
            entry.delete(0, tk.END)
            entry.insert(0, value)
        if not self.title_entry.get().strip():
            self.title_entry.insert(0, os.path.splitext(os.path.basename(f))[0])
        self.active, self.page_idx = False, 0
        try:
            import fitz
            doc = fitz.open(f)
            pix = doc[0].get_pixmap(dpi=60)
            self._pdf_preview = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            self._pdf_pages = len(doc)
            self.page_info.configure(text=f"{self._pdf_pages} pages. Page 1, shown here, will be the cover.")
        except Exception as e:
            self._pdf_preview = None
            self.page_info.configure(text=f"Could not open this PDF: {e}")
        self.render(force=True)

    def browse_epub(self):
        f = filedialog.asksaveasfilename(defaultextension=".epub", filetypes=[("EPUB files", "*.epub")])
        if f:
            self.epub_entry.delete(0, tk.END)
            self.epub_entry.insert(0, f)

    # ── run / pause / stop ───────────────────────────────────────────────────
    def start_conversion(self):
        pdf, out, mode = self.pdf_entry.get().strip(), self.epub_entry.get().strip(), self.mode.get()
        if not pdf or not out:
            return messagebox.showwarning("Missing fields", "Please choose the PDF and where to save the EPUB.")
        if not os.path.exists(pdf):
            return messagebox.showerror("File error", "The selected PDF file does not exist.")
        key = self.api_entry.get().strip() if mode == "Gemini" else None
        if mode == "Gemini" and not key:
            return messagebox.showwarning("No API key", "Paste a Gemini API key, or switch to Offline.")
        args = dict(api_key=key, epub_out=out, title=self.title_entry.get().strip() or None,
                    author=self.author_entry.get().strip(), threads=int(self.threads.get()),
                    sleep=round(self.cool.get(), 2), progress_callback=lambda k, n: EVENTS.put(("progress", k, n)),
                    checkpoint=self._checkpoint)
        self.running = self.active = True
        self.epub_path, self.pause_t = out, None
        self.unpaused.set()
        self.stop_req.clear()
        self._enable(self.start_btn, False)
        self.start_btn.configure(text="⏳  Converting…")
        self._enable(self.pause_btn, True)
        self._enable(self.stop_btn, True)
        self.pause_btn.configure(text="❚❚  Pause")
        self._enable(self.open_btn, False)
        self._enable(self.folder_btn, False)
        self.progress.configure(progress_color=ACCENT)
        self._reset_stats()
        self.pct_label.configure(text_color=ACCENT)
        self.run_t0 = self.stage_t0 = time.time()
        self.stage_label.configure(text="Starting…", text_color=TEXT)
        self._set_step(0)
        self.follow.select()
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        threading.Thread(target=self._worker, args=(pdf, args), daemon=True).start()

    def _checkpoint(self):
        """Worker thread, before each page: wait while paused, end the run on Stop."""
        while not self.unpaused.wait(0.3):
            pass
        if self.stop_req.is_set():
            raise StopRequested

    def toggle_pause(self):
        if not self.running or self.stop_req.is_set():
            return
        if self.pause_t is None:
            self.pause_t = time.time()
            self.unpaused.clear()
            self.pause_btn.configure(text="▶  Resume")
            self.stage_label.configure(text="❚❚  Paused (the current page finishes first)", text_color=WARN)
            self.progress.configure(progress_color=WARN)
        else:
            d = time.time() - self.pause_t
            self.run_t0 += d  # paused time does not count toward elapsed time or speed
            self.stage_t0 += d
            self.pause_t = None
            self.unpaused.set()
            self.pause_btn.configure(text="❚❚  Pause")
            self.stage_label.configure(text=self.stage_text, text_color=TEXT)
            self.progress.configure(progress_color=ACCENT)

    def stop_conversion(self):
        if not self.running:
            return
        self.stop_req.set()
        self.unpaused.set()  # wake a paused worker so it can stop
        self._enable(self.pause_btn, False)
        self._enable(self.stop_btn, False)
        self.stage_label.configure(text="■  Stopping after the current page…", text_color=DANGER)

    def _worker(self, pdf, args):
        try:
            EVENTS.put(("done", str(pipeline.run(pdf, **args))))
        except StopRequested:
            EVENTS.put(("stopped", ""))
        except Exception as e:
            EVENTS.put(("error", f"{type(e).__name__}: {e}"))

    def _drain(self):
        logs = []
        try:
            while True:
                ev = EVENTS.get_nowait()
                if ev[0] == "log":
                    logs.append(ev[1])
                    m = re.search(r"\[STAGE (\d)\]", ev[1])
                    if m and self.running:
                        self.stage_text = STAGES.get(m.group(1), "Working…")
                        self._set_step(STEP_OF.get(m.group(1), 0))
                        if self.pause_t is None and not self.stop_req.is_set():
                            self.stage_label.configure(text=self.stage_text)
                        self.stage_t0 = time.time()
                        self._reset_stats()
                elif ev[0] == "progress":
                    k, n = ev[1], ev[2]
                    spent = (self.pause_t or time.time()) - self.stage_t0
                    self.progress.set(k / n if n else 0)
                    self.pct_label.configure(text=f"{100 * k // max(n, 1)}%")
                    self.t_pages.configure(text=f"{k} / {n}")
                    self.t_left.configure(text=fmt_time(spent / max(k, 1) * (n - k)))
                    self.t_speed.configure(text=f"{spent / max(k, 1):.1f} s/page")
                else:
                    self._finish(ev[0], ev[1])
        except queue.Empty:
            pass
        if logs:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", "".join(logs))
            if int(self.log_box.index("end").split(".")[0]) > 4000:
                self.log_box.delete("1.0", "1000.0")  # keep the log light on low-RAM PCs
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(200, self._drain)

    def _finish(self, kind, msg):
        self.running, self.pause_t = False, None
        self.t_elapsed.configure(text=fmt_time(time.time() - self.run_t0))  # final run time stays visible
        self._enable(self.start_btn, True)
        self.start_btn.configure(text="▶  Resume conversion" if kind == "stopped" else "▶  Start conversion")
        self._enable(self.pause_btn, False)
        self._enable(self.stop_btn, False)
        self.pause_btn.configure(text="❚❚  Pause")
        if kind == "done":
            self.epub_path = msg
            self.progress.set(1)
            self.progress.configure(progress_color=OK)
            self._set_step(len(STEPS))
            self.pct_label.configure(text="100%", text_color=OK)
            self.stage_label.configure(text="✓  Done: the EPUB is saved", text_color=OK)
            self.t_left.configure(text="—")
            self._enable(self.open_btn, True)
            self._enable(self.folder_btn, True)
            self.bell()
        elif kind == "stopped":
            self.progress.configure(progress_color=WARN)
            self.stage_label.configure(text="■  Stopped. Finished pages are saved; press Resume to continue.",
                                       text_color=WARN)
            self.t_left.configure(text="—")
        else:
            self.progress.configure(progress_color=DANGER)
            self.stage_label.configure(text="✕  Stopped with an error (see the Log tab)", text_color=DANGER)
            messagebox.showerror("Conversion failed", msg)

    def on_close(self):
        if self.running and not messagebox.askyesno(
                "Conversion running",
                "A conversion is still running. Close anyway?\nFinished pages are saved and the next run resumes."):
            return
        self.destroy()

    # ── live tick: elapsed clock + preview ───────────────────────────────────
    def _tick(self):
        if self.running:
            self.t_elapsed.configure(text=fmt_time((self.pause_t or time.time()) - self.run_t0))
        pages = self._pages()
        if pages:
            last = len(pages) - 1
            if self.running and self.follow.get():
                done = [i for i, p in enumerate(pages) if i and any(
                    d and (d / f"{p.stem}.json").exists() for d in (pipeline.LOCAL_CACHE, pipeline.OCR_CACHE))]
                self.page_idx = done[-1] if done else last
            self.render()
        self.after(1000, self._tick)

    def _pages(self):
        d = pipeline.PAGES_DIR
        return sorted(d.glob("page_*.jpg")) if self.active and d and d.exists() else []

    def _meta(self, stem):
        """Merged page info: Gemini cache wins for photos/heading; the local cache adds page_format."""
        merged, found = {}, False
        for d in (pipeline.LOCAL_CACHE, pipeline.OCR_CACHE):
            f = d / f"{stem}.json" if d else None
            if f and f.exists():
                try:
                    merged.update(json.loads(f.read_text(encoding="utf-8")))
                    found = True
                except (ValueError, OSError):
                    pass  # being written right now
        return merged if found else None

    def _slid(self, v):
        self.follow.deselect()
        self.page_idx = int(round(v))
        self.render()

    def step(self, d):
        self.follow.deselect()
        self.page_idx += d
        self.render()

    def _key_step(self, d):
        if not isinstance(self.focus_get(), tk.Entry):  # arrows still move the cursor inside text boxes
            self.step(d)

    def open_page_image(self):
        pages = self._pages()
        if pages:
            self._open(str(pages[min(self.page_idx, len(pages) - 1)]))

    def _open(self, path):
        try:
            os.startfile(path)
        except (OSError, AttributeError) as e:
            messagebox.showerror("Cannot open", str(e))

    def render(self, force=False):
        w, h = max(60, self.canvas.winfo_width()), max(60, self.canvas.winfo_height())
        mode = ctk.get_appearance_mode()
        dark = mode == "Dark"
        self.canvas.configure(bg=CARD[1] if dark else CARD[0])
        pages, meta = self._pages(), None
        if pages:
            self.page_idx = min(max(0, self.page_idx), len(pages) - 1)
            path = pages[self.page_idx]
            meta = self._meta(path.stem) if self.page_idx else None
            key = (str(path), w, h, json.dumps(meta, sort_keys=True, ensure_ascii=False) if meta else None, mode)
        elif self._pdf_preview is not None:
            self.page_idx, key = 0, ("pdf", id(self._pdf_preview), w, h, mode)
        else:
            return
        if key == self._render_key and not force:
            return
        self._render_key = key
        img = Image.open(path).convert("RGB") if pages else self._pdf_preview.copy()
        photos = [b for b in (norm_box(p.get("box_2d")) for p in (meta or {}).get("photos") or [] if isinstance(p, dict)) if b]
        draw, (W, H) = ImageDraw.Draw(img), img.size
        for y0, x0, y1, x1 in photos:
            draw.rectangle([x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000], outline=PHOTO_RGB,
                           width=max(3, W // 180))
        img.thumbnail((w - 24, h - 24))
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        iw, ih = img.size
        x0, y0 = (w - iw) // 2, (h - ih) // 2
        self.canvas.create_rectangle(x0 + 5, y0 + 6, x0 + iw + 5, y0 + ih + 6, width=0,
                                     fill="#05070a" if dark else "#cbd5e1")  # page shadow
        self.canvas.create_image(x0, y0, image=self._tk_img, anchor="nw")
        if pages:
            self.page_slider.configure(to=max(1, len(pages) - 1), number_of_steps=max(1, len(pages) - 1))
            self.page_slider.set(self.page_idx)
            n = f"Page {self.page_idx + 1} / {len(pages)}"
            if self.page_idx == 0:
                info = f"{n}  ·  COVER (used whole as the EPUB cover)"
            elif meta is None:
                info = f"{n}  ·  not read yet"
            else:
                fmt = meta.get("page_format") or ("text+photo" if photos else "text")
                info = f"{n}  ·  {fmt}  ·  {len(photos)} photo(s)" + (
                    f"  ·  chapter: {meta['heading']}" if meta.get("heading") else "")
            self.page_info.configure(text=info)


if __name__ == "__main__":
    PdfToEpubApp().mainloop()
