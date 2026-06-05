import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
import librosa
import joblib
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from resemblyzer import VoiceEncoder, preprocess_wav

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CUSTOM_MODEL_PATH = Path("models/custom_detector.pkl")

HF_MODELS = [
    "MelodyMachine/Deepfake-audio-detection-V2",
    "Jianf-MS/Audio-Deepfake-Detection-wav2vec2",
    "aumaree/ai-audio-detector"
]

VALID_EXT = [".wav", ".mp3", ".mp4"]

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
print("Loading Resemblyzer...")
encoder = VoiceEncoder()

print("Loading logistic regression model...")
try:
    clf = joblib.load(CUSTOM_MODEL_PATH)
    print("Custom model loaded successfully.")
except:
    clf = None
    print("⚠ No custom model found. Detection will use HF-only ensemble.")

print("Loading HF detectors...")
hf_extractors = []
hf_models = []

for model_id in HF_MODELS:
    print("Loading:", model_id)
    hf_extractors.append(AutoFeatureExtractor.from_pretrained(model_id))
    hf_models.append(AutoModelForAudioClassification.from_pretrained(model_id))

print("\nAll models loaded.\n")

# ---------------------------------------------------------
# FEATURE EXTRACTION PIPELINE
# ---------------------------------------------------------
def extract_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True)

    # --- Resemblyzer ---
    wav = preprocess_wav(path)
    emb = encoder.embed_utterance(wav)

    # --- HF detector scores ---
    detector_scores = []
    for extr, model in zip(hf_extractors, hf_models):
        inputs = extr(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        score = torch.softmax(logits, dim=1)[0][1].item()
        detector_scores.append(score)

    # --- Spectral features ---
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y, sr=16000))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y, sr=16000))
    flatness = np.mean(librosa.feature.spectral_flatness(y))

    return np.concatenate([
        emb,
        np.array(detector_scores),
        np.array([zcr, centroid, bandwidth, flatness])
    ])


def predict_ai_score(path):
    features = extract_features(path)

    # If no logistic regression model is trained
    if clf is None:
        # Pure HF ensemble fallback
        hf_part = features[len(encoder.embed_utterance(np.zeros(16000))):len(encoder.embed_utterance(np.zeros(16000)))+3]
        return float(np.mean(hf_part))

    # Use logistic regression
    proba = clf.predict_proba([features])[0][1]
    return float(proba)


# ---------------------------------------------------------
# TKINTER GUI
# ---------------------------------------------------------
class VoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Voice Detector — Upgraded")
        self.root.geometry("600x500")

        self.files = []

        tk.Label(root, text="AI Voice Detector", font=("Arial", 16, "bold")).pack(pady=10)

        tk.Button(root, text="Add Audio Files", command=self.add_files).pack(pady=5)
        tk.Button(root, text="Run Detection", command=self.run_detection).pack(pady=5)

        self.listbox = tk.Listbox(root, width=70, height=15)
        self.listbox.pack(pady=10)

    def add_files(self):
        selected = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[("Audio", "*.wav *.mp3 *.mp4")]
        )
        for f in selected:
            self.files.append(f)
            self.listbox.insert(tk.END, f)

    def run_detection(self):
        if not self.files:
            messagebox.showerror("Error", "No files selected.")
            return

        results = []
        for f in self.files:
            print("\nProcessing:", f)
            score = predict_ai_score(f)
            label = "AI" if score > 0.5 else "Human"

            results.append((Path(f).name, label, score))

        self.show_results(results)

    def show_results(self, results):
        result_win = tk.Toplevel(self.root)
        result_win.title("Detection Results")

        tk.Label(result_win, text="AI Detection Results", font=("Arial", 14, "bold")).pack(pady=10)

        for name, label, score in results:
            text = f"{name}: {label} ({score:.3f})"
            tk.Label(result_win, text=text, font=("Arial", 11)).pack(anchor="w", padx=20)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
root = tk.Tk()
app = VoiceGUI(root)
root.mainloop()
