"""
Baseline UMAP projection of your WAV files using the original Resemblyzer model,
with text output for voice-to-voice similarity results.
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

reference_file = "Original.wav"
encoder = VoiceEncoder()

# ---------------- CHECK FOLDER ----------------
print(f"📂 Using folder: {AUDIO_FOLDER.resolve()}")
if not AUDIO_FOLDER.exists():
    print("❌ Folder not found. Please make sure 'my_audio' exists inside the project folder.")
    exit(1)
else:
    print("📁 Folder exists: True")

valid_exts = [".wav", ".mp3"]
files = [p for p in AUDIO_FOLDER.iterdir() if p.suffix.lower() in valid_exts]

if not files:
    print("❌ No audio files found in folder!")
    exit(1)

print("🎧 Found files:", [f.name for f in files])

# ---------------- EMBEDDINGS ----------------
embeddings = {}
for f in files:
    wav = preprocess_wav(f)
    emb = encoder.embed_utterance(wav)
    embeddings[f.name] = emb
    print(f"✅ Processed: {f.name}")

# ---------------- SIMILARITY OUTPUT ----------------
print("\n🔍 Calculating pairwise similarities...\n")

names = list(embeddings.keys())
emb_matrix = np.vstack([embeddings[n] for n in names])
ref_emb = embeddings.get(reference_file, None)

if ref_emb is not None:
    print(f"🎯 Using '{reference_file}' as reference voice\n")
    for n in names:
        if n == reference_file:
            continue
        score = np.dot(ref_emb, embeddings[n])
        status = "✅ Closer (more similar)" if score > 0.7 else "⚠️ Distant"
        print(f"{n:15} → Similarity: {score:.3f}  {status}")
else:
    print("⚠️ Reference voice not found; printing all pairwise distances instead.\n")
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            score = np.dot(embeddings[n1], embeddings[n2])
            print(f"{n1:15} vs {n2:15} → Similarity: {score:.3f}")

# ---------------- UMAP REDUCTION ----------------
proj = umap.UMAP(
    n_neighbors=5,
    min_dist=0.1,
    n_components=2,
    random_state=42
).fit_transform(emb_matrix)

# ---------------- SAVE CSV ----------------
df = pd.DataFrame({
    "filename": names,
    "umap_x": proj[:, 0],
    "umap_y": proj[:, 1],
})
df_path = OUTPUT_DIR / "embeddings_baseline.csv"
df.to_csv(df_path, index=False)

# ---------------- PLOT ----------------
plt.figure(figsize=(8,6))
plt.scatter(proj[:,0], proj[:,1], s=120)

for i, n in enumerate(names):
    color = "red" if n == reference_file else "black"
    plt.text(proj[i,0]+0.01, proj[i,1]+0.01, n, fontsize=9, color=color)

plt.title("UMAP — Baseline Embeddings (Original in red)")
plt.grid(alpha=0.3)
img_path = OUTPUT_DIR / "umap_baseline.png"
plt.savefig(img_path, dpi=200)

print("\n✅ Baseline UMAP saved:", img_path)
print("📁 CSV saved:", df_path)
print("\n📊 Done — Similarity results printed above and plot saved.")
