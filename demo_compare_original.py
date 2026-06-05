"""
demo_compare_original.py
Baseline UMAP + similarity ranking using your 6 voice files.
Original.wav is the reference voice.
"""

from pathlib import Path
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from resemblyzer import VoiceEncoder, preprocess_wav
import umap

# ---------------- CONFIG ----------------
AUDIO_FOLDER = Path("my_audio")
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

reference_file = "Original.wav"  # focus voice

print("Starting voice similarity & UMAP baseline run")

# Validate reference exists
ref_path = AUDIO_FOLDER / reference_file
if not ref_path.exists():
    raise FileNotFoundError(f"Missing reference file: {ref_path}")

# Collect audio files
files = sorted([p for p in AUDIO_FOLDER.iterdir() if p.suffix.lower() == ".wav"])

if len(files) < 2:
    raise RuntimeError("Need at least 2 WAV files in my_audio folder.")

print(f"🎧 Found files:", [f.name for f in files])

# ---------------- RESSEMBLYZER ----------------
encoder = VoiceEncoder()

embeddings = {}
for f in files:
    wav = preprocess_wav(f)
    emb = encoder.embed_utterance(wav)
    embeddings[f.name] = emb
    print(f"Embedded {f.name}")

print("\n Reference voice:", reference_file)

ref_emb = embeddings[reference_file]

# Compute similarity (cosine = dot product since embeddings normalized)
scores = {}
for name, emb in embeddings.items():
    if name == reference_file:
        continue
    scores[name] = float(np.dot(ref_emb, emb))

# Sort highest similarity first
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\n Similarity to Original.wav (higher = closer):")
for name, score in ranked:
    print(f"{name}: {score:.3f}")

# ---------------- UMAP PLOT ----------------
emb_matrix = np.vstack([embeddings[n] for n in embeddings])
labels = list(embeddings.keys())

proj = umap.UMAP(
    n_neighbors=5,
    min_dist=0.1,
    n_components=2,
    random_state=42
).fit_transform(emb_matrix)

plt.figure(figsize=(8, 6))
plt.scatter(proj[:, 0], proj[:, 1], s=120)

for i, label in enumerate(labels):
    color = "red" if label == reference_file else "black"
    plt.text(proj[i, 0] + .01, proj[i, 1] + .01, label, fontsize=9, color=color)

plt.title("UMAP — Voice Similarity Baseline (Original = red)")
plt.xlabel("UMAP X")
plt.ylabel("UMAP Y")
plt.grid(alpha=.3)

img_path = OUTPUT_DIR / "umap_baseline.png"
plt.savefig(img_path, dpi=200)
print(f"\n UMAP saved:", img_path)

# Save CSV results
df = pd.DataFrame({
    "filename": labels,
    "umap_x": proj[:, 0],
    "umap_y": proj[:, 1],
})
df_path = OUTPUT_DIR / "embeddings_baseline.csv"
df.to_csv(df_path, index=False)
print(f"CSV saved:", df_path)

print("\n DONE — Baseline UMAP + similarity complete")
