#!/usr/bin/env python3
\"\"\"optimized_resemblyzer_cli.py

Level-3 optimized CLI for computing Resemblyzer embeddings and similarities.
Features:
 - Faster audio preprocessing with multiprocessing
 - Caching of preprocessed wavs and embeddings (npz)
 - Correct cosine normalization and similarity scores
 - Optional UMAP dimensionality reduction (skippable for large datasets)
 - Optional FAISS nearest-neighbor search for scalability
 - CLI with argparse, logging, and progress bars
 - Saves CSV of embeddings + UMAP coords and a scatter PNG

Usage example:
    python optimized_resemblyzer_cli.py --audio-dir my_audio --output-dir plots --reference Original.wav --workers 4 --use-faiss

Requires: resemblyzer, numpy, pandas, matplotlib, tqdm, umap-learn (optional), faiss (optional)
\"\"\"

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import json

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except Exception as e:
    raise RuntimeError(\"Please install resemblyzer (pip install resemblyzer)\") from e

# Optional imports
UMAP_AVAILABLE = True
try:
    import umap
except Exception:
    UMAP_AVAILABLE = False

FAISS_AVAILABLE = True
try:
    import faiss
except Exception:
    FAISS_AVAILABLE = False

# ---------------- utilities ----------------
def setup_logging(level=logging.INFO):
    fmt = \"%(asctime)s — %(levelname)s — %(message)s\"
    logging.basicConfig(format=fmt, level=level)

def find_audio_files(folder: Path, exts=None):
    if exts is None:
        exts = [\".wav\", \".mp3\", \".ogg\", \".flac\"]

    files = []
    for ext in exts:
        files.extend(sorted(folder.glob(f\"*{ext}\")))
    return files

def preprocess_one(path_str):
    \"\"\"Function used by multiprocessing pool: loads and preprocesses a single wav file.
    Returns (name, wav_array) where wav_array is a numpy float32 1D array.
    \"\"\"
    path = Path(path_str)
    try:
        wav = preprocess_wav(path)  # resemblyzer helper: loads, resamples, trims
        # keep first 6 seconds to speed up (configurable elsewhere)
        sr = 16000
        max_dur_sec = 6.0
        wav = wav[: int(sr * max_dur_sec)]
        return (path.name, wav)
    except Exception as e:
        return (path.name, None)

def l2_normalize(x, axis=1, eps=1e-10):
    if x.ndim == 1:
        norm = np.linalg.norm(x) + eps
        return x / norm
    norm = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / norm

# ---------------- main logic ----------------
def compute_embeddings(audio_paths, workers=0, cache_npz=None, skip_preprocess=False):
    \"\"\"Preprocess (parallel) and compute embeddings (sequential due to model)\"\"\"
    encoder = VoiceEncoder()

    # Preprocess with multiprocessing (CPU-bound)
    if not skip_preprocess:
        if workers and workers > 1:
            nproc = min(workers, cpu_count())
            with Pool(nproc) as p:
                results = list(tqdm(p.imap(preprocess_one, [str(p) for p in audio_paths]), total=len(audio_paths), desc=\"Preprocessing\"))
        else:
            results = [preprocess_one(str(p)) for p in audio_paths]
    else:
        # If skip_preprocess True, assume audio_paths are already (name, wav) tuples passed in
        results = audio_paths

    # Filter out failed loads
    wavs = []
    names = []
    for name, wav in results:
        if wav is None:
            logging.warning(f\"Failed to load {name}; skipping.\")
            continue
        names.append(name)
        wavs.append(wav.astype(np.float32))

    if len(wavs) == 0:
        raise RuntimeError(\"No valid audio loaded.\")

    # Check cache possibility
    if cache_npz and Path(cache_npz).exists():
        logging.info(f\"Loading embeddings from cache: {cache_npz}\")
        loaded = np.load(cache_npz, allow_pickle=True)
        cached_names = list(loaded['names'])
        embeddings = loaded['embs']
        # If cache contains everything we need and order matches, return early
        if cached_names == names:
            return names, embeddings, wavs
        # Otherwise, try to align and recompute missing
        name_to_idx = {n:i for i,n in enumerate(cached_names)}
        emb_list = []
        to_compute = []
        to_compute_idx = []
        for i, n in enumerate(names):
            if n in name_to_idx:
                emb_list.append(embeddings[name_to_idx[n]])
            else:
                emb_list.append(None)
                to_compute.append(wavs[i])
                to_compute_idx.append(i)
        # compute missing embeddings sequentially
        for ii, wav in enumerate(tqdm(to_compute, desc=\"Embedding (recompute missing)\")):
            emb = encoder.embed_utterance(wav)
            emb_list[to_compute_idx[ii]] = emb
        emb_matrix = np.vstack(emb_list)
    else:
        # compute all embeddings (sequentially — model inference is usually GPU-bound; multiprocessing is harmful)
        emb_matrix = []
        for wav in tqdm(wavs, desc=\"Embedding\", total=len(wavs)):
            emb = encoder.embed_utterance(wav)
            emb_matrix.append(emb)
        emb_matrix = np.vstack(emb_matrix)
        # save cache if requested
        if cache_npz:
            np.savez_compressed(cache_npz, names=np.array(names), embs=emb_matrix)
            logging.info(f\"Saved embedding cache to: {cache_npz}\")

    return names, emb_matrix, wavs

def compute_similarity_matrix(emb_matrix):
    # L2-normalize rows so dot product == cosine similarity
    emb_norm = l2_normalize(emb_matrix, axis=1)
    sim = np.dot(emb_norm, emb_norm.T)
    return sim, emb_norm

def nearest_neighbors_faiss(emb_norm, top_k=5):
    if not FAISS_AVAILABLE:
        raise RuntimeError(\"FAISS not available. Install faiss or unset --use-faiss\")
    dim = emb_norm.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product for normalized vectors == cosine
    index.add(emb_norm.astype(np.float32))
    D, I = index.search(emb_norm.astype(np.float32), top_k)
    return D, I

def reduce_umap(emb_norm, n_neighbors=5, min_dist=0.1, n_components=2, random_state=42):
    if not UMAP_AVAILABLE:
        raise RuntimeError(\"UMAP not available. pip install umap-learn\")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    proj = reducer.fit_transform(emb_norm)
    return proj

def save_outputs(output_dir: Path, names, emb_matrix, proj=None, sim_matrix=None, reference=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save CSV with embeddings (we'll save 256 dims as columns) and optional UMAP coords
    dim = emb_matrix.shape[1]
    df = pd.DataFrame(emb_matrix, columns=[f\"e_{i}\" for i in range(dim)])
    df.insert(0, \"filename\", names)
    if proj is not None:
        df[\"umap_x\"] = proj[:,0]
        df[\"umap_y\"] = proj[:,1]
    csv_path = output_dir / \"embeddings_full.csv\"
    df.to_csv(csv_path, index=False)
    logging.info(f\"Saved CSV: {csv_path}\")

    # Save similarity CSV if provided
    if sim_matrix is not None:
        sim_df = pd.DataFrame(sim_matrix, index=names, columns=names)
        sim_path = output_dir / \"similarity_matrix.csv\"
        sim_df.to_csv(sim_path)
        logging.info(f\"Saved similarity matrix CSV: {sim_path}\")

    # Save UMAP plot if proj provided
    if proj is not None:
        plt.figure(figsize=(9,7))
        plt.scatter(proj[:,0], proj[:,1], s=100)
        for i,n in enumerate(names):
            color = \"red\" if reference and n == reference else \"black\"
            plt.text(proj[i,0]+0.01, proj[i,1]+0.01, n, fontsize=9, color=color)
        plt.title(\"UMAP — Embeddings (reference in red)\")
        plt.grid(alpha=0.3)
        img_path = output_dir / \"umap_scatter.png\"
        plt.savefig(img_path, dpi=200, bbox_inches='tight')
        plt.close()
        logging.info(f\"Saved UMAP plot: {img_path}\")

    return csv_path

def classify_score(score):
    if score > 0.80:
        return \"💯 Near-identical voice\"
    if score > 0.68:
        return \"👍 Similar\"
    if score > 0.50:
        return \"⚠️ Weak similarity\"
    return \"❌ Not similar\"

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description=\"Optimized Resemblyzer CLI — embeddings, similarity, optional UMAP/FAISS\")
    p.add_argument(\"--audio-dir\", type=str, default=\"my_audio\", help=\"Folder containing audio files\")
    p.add_argument(\"--output-dir\", type=str, default=\"plots\", help=\"Output folder for CSV/plots/cache\") 
    p.add_argument(\"--reference\", type=str, default=None, help=\"Reference filename to compare against (exact filename)\")
    p.add_argument(\"--workers\", type=int, default=0, help=\"Number of preprocess workers (0 => no multiprocessing)\")
    p.add_argument(\"--cache\", type=str, default=None, help=\"Path to .npz cache for embeddings (will be created if missing)\")
    p.add_argument(\"--no-umap\", action='store_true', help=\"Skip UMAP dimensionality reduction and plotting\") 
    p.add_argument(\"--use-faiss\", action='store_true', help=\"Use FAISS for nearest neighbors (optional)\") 
    p.add_argument(\"--top-k\", type=int, default=5, help=\"Top-K neighbors to compute (used with FAISS)\")
    p.add_argument(\"--max-umap-samples\", type=int, default=200, help=\"Skip UMAP if #samples > this\") 
    return p.parse_args()

def main():
    args = parse_args()
    setup_logging()
    start_time = time.time()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    audio_dir_exists = audio_dir.exists() and audio_dir.is_dir()
    if not audio_dir_exists:
        logging.error(f\"Audio directory not found: {audio_dir.resolve()}\")
        return

    files = find_audio_files(audio_dir)
    if not files:
        logging.error(\"No audio files found in the audio directory.\")
        return
    logging.info(f\"Found {len(files)} audio files.\")

    # Compute embeddings (with caching & parallel preprocess)
    cache_path = args.cache
    names, emb_matrix, _ = compute_embeddings(files, workers=args.workers, cache_npz=cache_path)

    # similarity matrix and normalized embeddings
    sim_matrix, emb_norm = compute_similarity_matrix(emb_matrix)

    # Print pairwise or reference comparisons
    if args.reference and args.reference in names:
        ref_idx = names.index(args.reference)
        ref_row = sim_matrix[ref_idx]
        logging.info(f\"Using reference: {args.reference}\")
        print(\"\\nSimilarity to reference:\")
        for i, n in enumerate(names):
            if i == ref_idx:
                continue
            score = float(ref_row[i])
            print(f\"{n:25} → sim={score:.4f}  {classify_score(score)}\")
    else:
        print(\"\\nPairwise top similarities (top 3 for each):\")
        for i, n in enumerate(names):
            row = sim_matrix[i]
            top_idx = np.argsort(-row)
            # skip self (first is self)
            top_idx = [j for j in top_idx if j != i][:3]
            tops = [(names[j], float(row[j])) for j in top_idx]
            print(f\"{n:25} → {', '.join([f'{nm} ({s:.3f})' for nm,s in tops])}\")

    # Optional FAISS nearest neighbors
    if args.use_faiss:
        if not FAISS_AVAILABLE:
            logging.warning(\"FAISS not installed; skipping FAISS nearest-neighbors.\")
        else:
            D, I = nearest_neighbors_faiss(emb_norm, top_k=args.top_k)
            print(\"\\nFAISS nearest neighbors (similarity, filename):\")
            for i, row in enumerate(I):
                items = [(float(D[i, k]), names[int(row[k])]) for k in range(1, min(args.top_k, row.shape[0]))]
                print(f\"{names[i]:25} → {', '.join([f'{s:.3f}:{nm}' for s,nm in items])}\")

    # Optional UMAP
    proj = None
    if not args.no_umap and len(names) <= args.max_umap_samples:
        if UMAP_AVAILABLE:
            logging.info(\"Computing UMAP projection...\")
            proj = reduce_umap(emb_norm)
        else:
            logging.warning(\"UMAP not available (install umap-learn) — skipping projection.\")
    else:
        logging.info(\"Skipping UMAP (either disabled or too many samples).\")


    # Save outputs
    csv_path = save_outputs(output_dir, names, emb_matrix, proj=proj, sim_matrix=sim_matrix, reference=args.reference)

    elapsed = time.time() - start_time
    logging.info(f\"Done in {elapsed:.1f}s — outputs: {csv_path}\")

if __name__ == '__main__':
    main()
