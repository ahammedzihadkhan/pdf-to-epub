import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import sys
import os

from run_gemini_pipeline import run_pipeline, get_api_key

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class PdfToEpubApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PDF to EPUB AI Converter")
        self.geometry("700x550")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # Title Label
        self.title_label = ctk.CTkLabel(self, text="PDF to EPUB Converter (Gemini OCR)", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 10))

        # API Key
        self.api_label = ctk.CTkLabel(self, text="Gemini API Key:")
        self.api_label.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        
        self.api_entry = ctk.CTkEntry(self, placeholder_text="Enter your Gemini API Key")
        self.api_entry.grid(row=1, column=1, columnspan=2, padx=20, pady=10, sticky="ew")
        # Load default API key if exists
        default_key = get_api_key()
        if default_key and default_key != "YOUR_API_KEY":
            self.api_entry.insert(0, default_key)

        # PDF Path
        self.pdf_label = ctk.CTkLabel(self, text="Input PDF:")
        self.pdf_label.grid(row=2, column=0, padx=20, pady=10, sticky="w")
        
        self.pdf_entry = ctk.CTkEntry(self, placeholder_text="Select a PDF file...")
        self.pdf_entry.grid(row=2, column=1, padx=(20, 0), pady=10, sticky="ew")
        
        self.pdf_btn = ctk.CTkButton(self, text="Browse", command=self.browse_pdf, width=100)
        self.pdf_btn.grid(row=2, column=2, padx=20, pady=10)

        # EPUB Path
        self.epub_label = ctk.CTkLabel(self, text="Output EPUB:")
        self.epub_label.grid(row=3, column=0, padx=20, pady=10, sticky="w")
        
        self.epub_entry = ctk.CTkEntry(self, placeholder_text="Save EPUB as...")
        self.epub_entry.grid(row=3, column=1, padx=(20, 0), pady=10, sticky="ew")
        
        self.epub_btn = ctk.CTkButton(self, text="Browse", command=self.browse_epub, width=100)
        self.epub_btn.grid(row=3, column=2, padx=20, pady=10)

        # Start Button
        self.start_btn = ctk.CTkButton(self, text="Start Conversion", command=self.start_conversion, fg_color="green", hover_color="darkgreen")
        self.start_btn.grid(row=4, column=0, columnspan=3, padx=20, pady=20, sticky="ew")

        # Progress Bar & Status
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.grid(row=5, column=0, columnspan=3, padx=20, pady=(0, 10), sticky="ew")
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.status_label.grid(row=6, column=0, columnspan=3, padx=20, pady=(0, 20))

    def browse_pdf(self):
        filename = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if filename:
            self.pdf_entry.delete(0, tk.END)
            self.pdf_entry.insert(0, filename)
            # auto-fill epub
            default_epub = os.path.splitext(filename)[0] + ".epub"
            self.epub_entry.delete(0, tk.END)
            self.epub_entry.insert(0, default_epub)

    def browse_epub(self):
        filename = filedialog.asksaveasfilename(defaultextension=".epub", filetypes=[("EPUB files", "*.epub")])
        if filename:
            self.epub_entry.delete(0, tk.END)
            self.epub_entry.insert(0, filename)

    def update_progress(self, completed, total):
        pct = completed / total if total > 0 else 0
        # Schedule GUI update from thread
        self.after(0, lambda: self.progress_bar.set(pct))
        self.after(0, lambda: self.status_label.configure(text=f"OCR Progress: {completed}/{total} pages ({int(pct*100)}%)"))

    def conversion_thread(self, pdf, api_key, epub_out):
        try:
            self.after(0, lambda: self.status_label.configure(text="Extracting images and starting OCR..."))
            run_pipeline(pdf, api_key, epub_out, progress_callback=self.update_progress)
            self.after(0, lambda: self.status_label.configure(text="Conversion Completed Successfully!", text_color="green"))
            self.after(0, lambda: messagebox.showinfo("Success", f"EPUB saved at:\n{epub_out}"))
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Error: {str(e)}", text_color="red"))
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: self.start_btn.configure(state="normal", text="Start Conversion"))

    def start_conversion(self):
        pdf = self.pdf_entry.get().strip()
        epub_out = self.epub_entry.get().strip()
        api_key = self.api_entry.get().strip()

        if not pdf or not epub_out or not api_key:
            messagebox.showwarning("Missing Fields", "Please fill in all the fields (API Key, PDF, EPUB).")
            return

        if not os.path.exists(pdf):
            messagebox.showerror("File Error", "The selected PDF file does not exist.")
            return

        self.start_btn.configure(state="disabled", text="Processing...")
        self.progress_bar.set(0)
        self.status_label.configure(text="Starting...", text_color="gray")

        # Run in background thread to keep UI responsive
        threading.Thread(target=self.conversion_thread, args=(pdf, api_key, epub_out), daemon=True).start()

if __name__ == "__main__":
    app = PdfToEpubApp()
    app.mainloop()
