# Voice Comparison Tool — A Resemblyzer-Based Analysis System

> A modern, optimized **GUI application** for voice similarity analysis and speaker verification using deep learning voice embeddings.

This is a **personal project built on top of Resemblyzer** (voice encoder by Resemble AI) that provides an intuitive desktop interface for comparing and analyzing voice recordings in real-time. It combines the power of pretrained neural voice encodings with an interactive visualization dashboard.

---

## 🎯 What This Project Does

This tool enables you to:

- **Compare multiple voice recordings** and compute similarity scores between them
- **Verify speaker identity** by comparing new audio against a reference voice profile
- **Visualize voice embeddings** in 2D space using UMAP dimensionality reduction
- **Convert any audio/video format** automatically (MP3, MP4, M4A, MOV, FLAC → 16kHz mono WAV)
- **Cache conversion results** to avoid redundant processing
- **Export results** as CSV (similarity scores + UMAP coordinates) and PNG plots
- **Analyze speaker similarity** across batches of files with confidence scoring

### Key Features

✅ **Universal Audio Support** — Accepts MP3, MP4, M4A, MOV, FLAC, WAV via automatic ffmpeg conversion  
✅ **Smart Caching** — Tracks converted files by hash to avoid re-processing  
✅ **Deep Learning Embeddings** — Uses Resemblyzer's 256-D d-vector voice encoder  
✅ **Interactive GUI** — Built with Tkinter for cross-platform compatibility  
✅ **UMAP Visualization** — 2D projection of voice embeddings clustered by speaker similarity  
✅ **Batch Processing** — Compare multiple voices simultaneously with reference-based similarity scoring  
✅ **Export Functionality** — Save results as CSV + PNG for further analysis  
✅ **Multi-threaded** — Long operations run in background without freezing UI  

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (officially tested with 3.10)
- **FFmpeg** (for audio format conversion) — [Download here](https://ffmpeg.org/download.html)
- **PyTorch** (CPU or GPU)

### Installation

1. **Clone this repository**
   ```bash
   cd C:\Users\[YourUsername]\Desktop
   git clone https://github.com/resemble-ai/Resemblyzer.git Resemblyzer_clone
   cd Resemblyzer_clone
   ```

2. **Create a Python virtual environment**
   ```bash
   py -3.10 -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch resemblyzer pydub numpy umap-learn matplotlib sounddevice soundfile
   ```

4. **Create working directories**
   ```bash
   mkdir my_audio
   mkdir plots
   ```

5. **Launch the application**
   ```bash
   python voice_gui_optimized.py
   ```

> 💡 **Alternative (Fast Setup):** See the interactive HTML simulator for a visual walkthrough:  
> Open `resemblyzer_simulator.html` in your browser for installation guidance and live terminal simulation.

---

## 📖 How to Use the GUI

### Main Window Layout

| Component | Purpose |
|-----------|---------|
| **Add Audio Files** | Select multiple audio/video files to compare |
| **Choose Reference Voice** | Pick a reference voice for similarity scoring |
| **Clear List** | Reset the file list and reference voice |
| **Run Comparison** | Execute embedding + similarity computation |
| **Save CSV** | Export similarity scores and UMAP coordinates |
| **Save PNG** | Save the 2D visualization plot as image |
| **File Listbox** | View all selected audio files |
| **Log Console** | Real-time processing status and scores |
| **UMAP Plot** | 2D scatter plot of voice embeddings (auto-updating) |

### Workflow Example

1. Click **"Add Audio Files"** → Select 5-10 voice recordings
2. Click **"Choose Reference Voice"** → Pick one voice to compare against
3. Click **"Run Comparison"** → 
   - Converts all files to 16kHz mono WAV (cached automatically)
   - Loads the VoiceEncoder model (~250MB)
   - Computes embeddings for each voice
   - Calculates similarity scores to the reference voice
   - Projects embeddings into 2D space with UMAP
   - Displays interactive plot with labels
4. Click **"Save CSV"** to export scores, or **"Save PNG"** to export the plot

### Interpreting Results

- **Similarity Score**: Range `[0.0, 1.0]`
  - `> 0.7` = "Closer" (high speaker similarity)
  - `≤ 0.7` = "Distant" (low speaker similarity)
- **UMAP Plot**: Voices that cluster together have similar acoustic characteristics
- **Reference Voice**: Marked in RED on the plot; all other voices in BLACK

---

## 🔧 Technical Details

### Core Scripts

| Script | Purpose |
|--------|---------|
| **`voice_gui_optimized.py`** | Main GUI application — Complete voice comparison dashboard |
| **`demo_utils.py`** | Utility functions for plotting, audio playback, and visualization |
| **`optimized_resemblyzer_cli.py`** | CLI interface for batch processing without GUI |
| **`demo01_similarity.py`** | Standalone demo: Cross-similarity matrix generation |
| **`demo02_diarization.py`** | Standalone demo: Speaker diarization (who is talking when) |
| **`demo03_projection_mine.py`** | Standalone demo: Custom UMAP projection with speaker clustering |
| **`demo05_fake_speech_detection.py`** | Standalone demo: Detect synthetic/fake speech |

### Data Flow

```
Audio Files (MP3, MP4, etc.)
    ↓
[Convert to 16kHz mono WAV] ← Cached by file hash
    ↓
[Load VoiceEncoder Model]
    ↓
[Compute 256-D Embeddings]
    ↓
[Calculate Similarity Scores] ← Reference-based or pairwise
    ↓
[UMAP Projection to 2D]
    ↓
[Visualize + Export CSV/PNG]
```

### File Conversion & Caching

- **Cache File**: `conversion_cache.json` — Maps input file path + hash → output WAV path
- **Work Directory**: `my_audio/` — Stores converted WAV files
- **Plots Directory**: `plots/` — Stores CSV and PNG exports

---

## 📊 Demo Scripts

Run any demo standalone for specific tasks:

```bash
# Cross-similarity matrix (compare multiple voices against each other)
python demo01_similarity.py

# Speaker diarization (determine who is talking in multi-speaker audio)
python demo02_diarization.py

# UMAP projection (visualize voice clusters)
python demo03_projection_mine.py

# Fake speech detection (identify synthetic/cloned voices)
python demo05_fake_speech_detection.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `resemblyzer` | Deep learning voice encoder (d-vector model) |
| `torch` | PyTorch backend for neural networks |
| `numpy` | Numerical computing |
| `pandas` | Data wrangling + CSV export |
| `matplotlib` | Plotting backend |
| `umap-learn` | Dimensionality reduction (2D projection) |
| `sounddevice` | Audio playback |
| `soundfile` | Audio file I/O |
| `pydub` | Audio format handling |

---

## 💾 Project Structure

```
661175VoiceComparison/
├── voice_gui_optimized.py        # Main GUI application ⭐
├── optimized_resemblyzer_cli.py  # CLI batch processing
├── demo*.py                       # Standalone demo scripts
├── demo_utils.py                 # Shared utilities
├── resemblyzer_simulator.html    # Interactive setup guide 📖
├── my_audio/                     # Working directory for audio files
├── plots/                        # Output directory for CSV + PNG
├── conversion_cache.json         # Cache of converted files
├── requirements_demos.txt        # Full dependencies
├── requirements_package.txt      # Minimal dependencies
└── README.md                     # This file
```

---

## 🎓 Background

This project builds upon **Resemblyzer** by Resemble AI, which provides a pretrained neural voice encoder for:
- **Speaker verification** — Verify if a person is who they claim to be
- **Speaker diarization** — Determine who is speaking at any given time in multi-speaker audio
- **Voice similarity** — Compute a numerical similarity metric between any two voices
- **Fake speech detection** — Identify synthetic or cloned speech

Original Resemblyzer: https://github.com/resemble-ai/Resemblyzer

---

## 📝 License

This project adapts code from Resemblyzer (licensed under Apache 2.0). See LICENSE for details.

---

## 🤝 Contributing

This is a personal academic/research project. Feel free to fork and adapt for your needs!

### Future Enhancements
- [ ] Real-time voice streaming comparison
- [ ] Speaker enrollment/database building
- [ ] Multi-language support
- [ ] Advanced filtering (noise removal, normalization)
- [ ] Web-based interface (FastAPI + React)

---

## ⚡ Troubleshooting

### Issue: "ffmpeg not found"
**Solution**: Install FFmpeg or place `ffmpeg.exe` in the project root directory.

### Issue: "CUDA out of memory"
**Solution**: The model will fall back to CPU. Ensure you have 8+ GB RAM available.

### Issue: "ModuleNotFoundError: No module named 'resemblyzer'"
**Solution**: Reinstall dependencies:
```bash
pip install --upgrade resemblyzer torch
```

### Issue: "Tkinter not available"
**Solution**: On Linux, install: `sudo apt-get install python3-tk`

---

## 📧 Questions?

Open an issue or refer to the original Resemblyzer documentation for more information about voice embeddings and the deep learning model.

---

**Happy voice analyzing!** 🎙️✨
