import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import shutil
import hashlib
import json


# =========================================================
# FAST HASHING UTILITY (avoid reloading/reconverting)
# =========================================================
def file_hash(path):
    """Return a short hash of a file's contents."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:12]


# =========================================================
# CONVERSION CACHE
# =========================================================
CACHE_FILE = "conversion_cache.json"
if Path(CACHE_FILE).exists():
    with open(CACHE_FILE, "r") as f:
        CONVERSION_CACHE = json.load(f)
else:
    CONVERSION_CACHE = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(CONVERSION_CACHE, f, indent=2)


# =========================================================
# FAST CONVERSION: ANY ⇒ WAV (with caching)
# =========================================================
def convert_to_wav(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    input_path = Path(input_path)
    file_key = str(input_path.resolve())
    hash_value = file_hash(input_path)

    # If unchanged since last time → use cached result
    if file_key in CONVERSION_CACHE:
        stored_hash, wav_output = CONVERSION_CACHE[file_key]
        if stored_hash == hash_value and Path(wav_output).exists():
            return wav_output  # no reconvert needed

    # Convert now
    stem = input_path.stem
    output_path = output_dir / f"{stem}.wav"

    ffmpeg = "ffmpeg.exe" if Path("ffmpeg.exe").exists() else "ffmpeg"

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Save cache
        CONVERSION_CACHE[file_key] = (hash_value, str(output_path))
        save_cache()
        return str(output_path)
    except Exception:
        return None


# =========================================================
# GUI CLASS
# =========================================================
class VoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Comparison GUI")
        self.root.geometry("750x570")
        self.file_list = []
        self.reference_wav = None

        # ============ Buttons Row ============
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)

        tk.Button(top_frame, text="Add Audio Files", command=self.add_files).grid(row=0, column=0, padx=10)
        tk.Button(top_frame, text="Choose Reference Voice", command=self.choose_reference).grid(row=0, column=1, padx=10)
        tk.Button(top_frame, text="Clear List", command=self.clear_list).grid(row=0, column=2, padx=10)

        # ============ File List ============
        self.listbox = tk.Listbox(root, width=95, height=12)
        self.listbox.pack(pady=10)

        # ============ Run Button ============
        tk.Button(root, text="Run Voice Comparison", command=self.run_process_thread).pack(pady=10)

        # ============ Log =============
        self.log = tk.Text(root, height=15, width=95)
        self.log.pack(pady=10)
        self.log.insert(tk.END, "Ready.\n")

    # =========================================================
    # ADD FILES
    # =========================================================
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[
                ("Audio/Video", "*.wav;*.mp3;*.mp4;*.m4a;*.mov"),
                ("All files", "*.*")
            ]
        )

        for p in paths:
            if p not in self.file_list:
                self.file_list.append(p)
                self.listbox.insert(tk.END, p)

        self.log.insert(tk.END, f"Added {len(paths)} files.\n")

    # =========================================================
    # CHOOSE REFERENCE VOICE
    # =========================================================
    def choose_reference(self):
        path = filedialog.askopenfilename(
            title="Choose reference voice",
            filetypes=[
                ("Audio/Video", "*.wav;*.mp3;*.mp4;*.m4a;*.mov"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        # Convert to WAV + store
        wav = convert_to_wav(path, "my_audio")
        if wav:
            self.reference_wav = Path(wav).name
            self.log.insert(tk.END, f"Reference set: {self.reference_wav}\n")
        else:
            self.log.insert(tk.END, "ERROR: Failed to convert reference.\n")

    # =========================================================
    # CLEAR LIST
    # =========================================================
    def clear_list(self):
        self.file_list.clear()
        self.listbox.delete(0, tk.END)
        self.log.insert(tk.END, "Cleared file list.\n")

    # =========================================================
    # RUN IN BACKGROUND
    # =========================================================
    def run_process_thread(self):
        t = threading.Thread(target=self.run_process)
        t.start()

    # =========================================================
    # MAIN PROCESS
    # =========================================================
    def run_process(self):
        # Ensure working directory exists
        work_dir = Path("my_audio")
        work_dir.mkdir(exist_ok=True)

        # Convert all files (using cache)
        for src in self.file_list:
            ext = Path(src).suffix.lower()
            if ext == ".wav":
                # Only copy if the content changed
                dst = work_dir / Path(src).name
                if not dst.exists() or file_hash(src) != file_hash(dst):
                    shutil.copy(src, dst)
                    self.log.insert(tk.END, f"Copied WAV: {src}\n")
            else:
                wav = convert_to_wav(src, work_dir)
                if wav:
                    self.log.insert(tk.END, f"Converted: {src} → {wav}\n")
                else:
                    self.log.insert(tk.END, f"FAILED conversion: {src}\n")

        # ------------------ RUN YOUR SCRIPT ------------------
        cmd = [
            "python",
            "optimized_resemblyzer_cli.py",
            "--audio-dir", "my_audio",
            "--output-dir", "plots",
            "--workers", "4"
        ]

        if self.reference_wav:
            cmd += ["--reference", self.reference_wav]

        self.log.insert(tk.END, "\nRunning comparison...\n")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in proc.stdout:
            self.log.insert(tk.END, line)
            self.log.see(tk.END)

        self.log.insert(tk.END, "\nDone.\n")


# =========================================================
# RUN APPLICATION
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceGUI(root)
    root.mainloop()
