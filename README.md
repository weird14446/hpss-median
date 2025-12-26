# hpss-median

> **Harmonic/Percussive Source Separation using Median Filtering**

A Python implementation of the algorithm described in the paper *"Harmonic/Percussive Separation Using Median Filtering"* by Derry FitzGerald (2010). This project separates a monophonic audio signal into **Harmonic** (pitched instruments) and **Percussive** (drums, transients) components using simple image processing techniques on spectrograms.

## 📖 Overview

In an audio spectrogram:
- [cite_start]**Harmonic sounds** (e.g., piano, violin) appear as **horizontal lines** (stable frequency over time). [cite: 20, 21]
- [cite_start]**Percussive sounds** (e.g., drums) appear as **vertical lines** (broadband energy in short time). [cite: 20, 23]

This project utilizes **Median Filters** to exploit these geometric characteristics:
1.  [cite_start]**Horizontal Median Filter:** Suppresses vertical lines (percussive) to extract harmonics. [cite: 7]
2.  [cite_start]**Vertical Median Filter:** Suppresses horizontal lines (harmonic) to extract percussive elements. [cite: 7]
3.  [cite_start]**Soft Masking:** Combines the filtered results to generate high-quality separated audio. [cite: 154]

## 🚀 Features

- [cite_start]**No Deep Learning Required:** Pure algorithmic approach using STFT and median filtering. [cite: 16]
- [cite_start]**Fast & Lightweight:** Much faster than iterative optimization methods. [cite: 6]
- **Visualizations:** Includes scripts to visualize the original, harmonic, and percussive spectrograms.

## 🛠️ Tech Stack

- **Python 3.x**
- **Librosa:** For audio loading and STFT/iSTFT operations.
- **NumPy:** For matrix operations and masking logic.
- **SciPy:** For `ndimage.median_filter`.

## 📄 Reference

This implementation is based on the following paper:

> **FitzGerald, D. (2010).** "Harmonic/Percussive Separation Using Median Filtering". *Proc. of the 13th Int. [cite_start]Conference on Digital Audio Effects (DAFx-10), Graz, Austria.* [cite: 1]

---
*Developed by [Your Name] as a study project on Audio Signal Processing.*
