"""
voice_gui_optimized.py
Combined, optimized GUI for Resemblyzer-based voice comparison and UMAP visualization.

Features:
- Add files, choose reference, clear list
- Converts non-wav files to 16k mono wav (ffmpeg) with cache
- Loads VoiceEncoder once, embeds files, computes similarities
- In-memory UMAP projection, displayed inside the same Tkinter window (matplotlib)
- Saves CSV and PNG to 'plots/' (optional via buttons)
"""

import os
import json
import hashlib
import shutil
import threading
from pathlib import Path
import subprocess
import sys
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Audio / ML imports (external libs)
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import umap
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from resemblyzer import VoiceEncoder, preprocess_wav
except Exception as e:
    print("Missing dependency:", e)
    print("Install required packages: pip install resemblyzer umap-learn numpy pandas matplotlib")
    raise

# -------------------- Config --------------------
WORK_DIR = Path("my_audio")
PLOTS_DIR = Path("plots")
CACHE_FILE = Path("conversion_cache.json")
WORK_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# -------------------- Utilities --------------------
def file_hash(path):
    """Return a short SHA1 hash of file contents (first 12 hex chars)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]

def load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print("Failed to save cache:", e)

CONVERSION_CACHE = load_cache()


def find_ffmpeg():
    # Prefer local file ffmpeg.exe (Windows) or system ffmpeg
    if Path("ffmpeg.exe").exists():
        return "ffmpeg.exe"
    return "ffmpeg"


def convert_to_wav(input_path, output_dir):
    """
    Convert any audio/video file to mono 16k WAV.
    Uses a conversion cache keyed by absolute input path.
    Returns the output wav path (string) or None on failure.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_key = str(input_path.resolve())
    hash_value = file_hash(input_path)

    if file_key in CONVERSION_CACHE:
        stored_hash, wav_output = CONVERSION_CACHE[file_key]
        if stored_hash == hash_value and Path(wav_output).exists():
            return wav_output

    # Build output path
    stem = input_path.stem
    out = (output_dir / f"{stem}.wav").resolve()

    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        str(out)
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        CONVERSION_CACHE[file_key] = (hash_value, str(out))
        save_cache(CONVERSION_CACHE)
        return str(out)
    except Exception:
        return None


# -------------------- GUI --------------------
class VoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Comparison — Optimized")
        self.root.geometry("900x700")
        self.file_list = []            # list of original paths
        self.reference_wav = None      # name of wav in WORK_DIR (basename)
        self.encoder = None            # lazy-loaded VoiceEncoder
        self.plot_canvas = None
        self.fig = None

        # Top controls
        top_frame = tk.Frame(root)
        top_frame.pack(pady=8, fill=tk.X)

        tk.Button(top_frame, text="Add Audio Files", command=self.add_files).grid(row=0, column=0, padx=6)
        tk.Button(top_frame, text="Choose Reference Voice", command=self.choose_reference).grid(row=0, column=1, padx=6)
        tk.Button(top_frame, text="Clear List", command=self.clear_list).grid(row=0, column=2, padx=6)
        tk.Button(top_frame, text="Run Comparison", command=self.run_process_thread, bg="#4CAF50", fg="white").grid(row=0, column=3, padx=8)
        tk.Button(top_frame, text="Save CSV", command=self.save_last_csv).grid(row=0, column=4, padx=6)
        tk.Button(top_frame, text="Save PNG", command=self.save_last_png).grid(row=0, column=5, padx=6)

        # File list + Reference label
        middle = tk.Frame(root)
        middle.pack(pady=6, fill=tk.X)

        self.listbox = tk.Listbox(middle, width=120, height=8)
        self.listbox.pack(side=tk.LEFT, padx=(10,0))

        scrollbar = tk.Scrollbar(middle, command=self.listbox.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        right_frame = tk.Frame(middle)
        right_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        self.ref_label = tk.Label(right_frame, text="Reference: (none)", fg="blue")
        self.ref_label.pack(anchor="n")

        # Log box
        self.log = tk.Text(root, height=12, width=120)
        self.log.pack(pady=6)
        self.log.insert(tk.END, "Ready.\n")

        # Plot area
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Internal state for saving last results
        self._last_df = None
        self._last_fig = None

    # ---------- File operations ----------
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[
                ("Audio/Video", "*.wav;*.mp3;*.mp4;*.m4a;*.mov;*.flac"),
                ("All files", "*.*")
            ]
        )
        for p in paths:
            if p not in self.file_list:
                self.file_list.append(p)
                self.listbox.insert(tk.END, p)
        self.log_insert(f"Added {len(paths)} file(s).\n")

    def choose_reference(self):
        path = filedialog.askopenfilename(
            title="Choose reference voice (audio/video)",
            filetypes=[
                ("Audio/Video", "*.wav;*.mp3;*.mp4;*.m4a;*.mov;*.flac"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return
        # Convert the chosen ref to WORK_DIR and store basename
        wav = convert_to_wav(path, WORK_DIR)
        if wav:
            self.reference_wav = Path(wav).name
            self.ref_label.config(text=f"Reference: {self.reference_wav}")
            self.log_insert(f"Reference set: {self.reference_wav}\n")
        else:
            self.log_insert("ERROR: Failed to convert reference.\n")

    def clear_list(self):
        self.file_list.clear()
        self.listbox.delete(0, tk.END)
        self.log_insert("Cleared file list.\n")
        self.reference_wav = None
        self.ref_label.config(text="Reference: (none)")

    # ---------- Logging ----------
    def log_insert(self, text):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    # ---------- Background thread runner ----------
    def run_process_thread(self):
        t = threading.Thread(target=self.run_process, daemon=True)
        t.start()

    # ---------- Main processing ----------
    def run_process(self):
        try:
            self.log_insert("\n--- Starting processing ---\n")
            WORK_DIR.mkdir(exist_ok=True)
            PLOTS_DIR.mkdir(exist_ok=True)

            # 1) Convert / copy input files into WORK_DIR
            wav_paths = []
            for src in self.file_list:
                src_path = Path(src)
                ext = src_path.suffix.lower()
                if ext == ".wav":
                    dst = WORK_DIR / src_path.name
                    # Copy only if different
                    if not dst.exists() or file_hash(src_path) != file_hash(dst):
                        shutil.copy(src_path, dst)
                        self.log_insert(f"Copied WAV: {src_path.name}\n")
                    else:
                        self.log_insert(f"WAV cached (copied earlier): {src_path.name}\n")
                    wav_paths.append(dst)
                else:
                    converted = convert_to_wav(src_path, WORK_DIR)
                    if converted:
                        wav_paths.append(Path(converted))
                        self.log_insert(f"Converted: {src_path.name} → {Path(converted).name}\n")
                    else:
                        self.log_insert(f"FAILED conversion: {src_path.name}\n")

            if not wav_paths:
                self.log_insert("No valid audio files to process. Aborting.\n")
                return

            # 2) Initialize encoder (if not loaded)
            if self.encoder is None:
                self.log_insert("Loading Resemblyzer model (this may take a moment)...\n")
                self.encoder = VoiceEncoder()
                self.log_insert("VoiceEncoder loaded.\n")

            # 3) Compute embeddings (in-memory)
            embeddings = {}
            for wav in wav_paths:
                self.log_insert(f"Embedding {wav.name} ...\n")
                try:
                    wav_arr = preprocess_wav(wav)  # loads & preprocess
                    emb = self.encoder.embed_utterance(wav_arr)
                    embeddings[wav.name] = emb
                    self.log_insert(f"Processed: {wav.name}\n")
                except Exception as e:
                    self.log_insert(f"ERROR processing {wav.name}: {e}\n")

            names = list(embeddings.keys())
            emb_matrix = np.vstack([embeddings[n] for n in names])

            # 4) Compute similarities & assemble DataFrame
            results = []
            if self.reference_wav and self.reference_wav in embeddings:
                ref_emb = embeddings[self.reference_wav]
                self.log_insert(f"\nUsing reference: {self.reference_wav}\n")
                for n in names:
                    if n == self.reference_wav:
                        continue
                    score = float(np.dot(ref_emb, embeddings[n]))
                    status = "Closer" if score > 0.7 else "Distant"
                    results.append({"filename": n, "similarity_to_reference": score, "status": status})
                    self.log_insert(f"{n:30s} → Similarity: {score:.3f}  ({status})\n")
            else:
                self.log_insert("\nNo reference or reference not in set; computing pairwise similarities.\n")
                for i, n1 in enumerate(names):
                    for j, n2 in enumerate(names):
                        if j <= i:
                            continue
                        score = float(np.dot(embeddings[n1], embeddings[n2]))
                        results.append({"filename": f"{n1} vs {n2}", "similarity": score})
                        self.log_insert(f"{n1} vs {n2} → Similarity: {score:.3f}\n")

            # 5) UMAP projection
            self.log_insert("\nComputing UMAP projection...\n")
            reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, random_state=42)
            proj = reducer.fit_transform(emb_matrix)

            # Build DataFrame for saving
            df = pd.DataFrame({
                "filename": names,
                "umap_x": proj[:, 0],
                "umap_y": proj[:, 1],
            })
            self._last_df = df  # store for Save CSV

            # 6) Plot (embedded in Tkinter)
            self.log_insert("Rendering plot...\n")
            fig, ax = plt.subplots(figsize=(6.5, 5))
            ax.scatter(proj[:, 0], proj[:, 1], s=110, alpha=0.9)
            for i, n in enumerate(names):
                color = "red" if n == self.reference_wav else "black"
                ax.text(proj[i, 0] + 0.01, proj[i, 1] + 0.01, n, fontsize=9, color=color)
            ax.set_title("UMAP — Voice Embeddings")
            ax.grid(alpha=0.35)

            # remove previous
            if self.plot_canvas:
                try:
                    self.plot_canvas.get_tk_widget().destroy()
                except Exception:
                    pass

            self.fig = fig
            self._last_fig = fig  # for Save PNG

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
            self.plot_canvas = canvas

            # 7) Save defaults (CSV + PNG) for convenience
            csv_path = PLOTS_DIR / "embeddings_gui.csv"
            png_path = PLOTS_DIR / "umap_gui.png"
            try:
                df.to_csv(csv_path, index=False)
                fig.savefig(png_path, dpi=180)
                self.log_insert(f"\nSaved CSV: {csv_path}\nSaved PNG: {png_path}\n")
            except Exception as e:
                self.log_insert(f"Warning: failed to save default files: {e}\n")

            self.log_insert("\n--- Processing complete ---\n")
        except Exception as e:
            self.log_insert("Unexpected error during processing:\n")
            self.log_insert(traceback.format_exc() + "\n")

    # ---------- Save helper buttons ----------
    def save_last_csv(self):
        if self._last_df is None:
            messagebox.showinfo("No Data", "No results to save. Run comparison first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")], initialdir=str(PLOTS_DIR))
        if not path:
            return
        try:
            self._last_df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"CSV saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV: {e}")

    def save_last_png(self):
        if self._last_fig is None:
            messagebox.showinfo("No Plot", "No plot to save. Run comparison first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")], initialdir=str(PLOTS_DIR))
        if not path:
            return
        try:
            self._last_fig.savefig(path, dpi=180)
            messagebox.showinfo("Saved", f"PNG saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PNG: {e}")


# -------------- Run app --------------
def main():
    root = tk.Tk()
    app = VoiceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
