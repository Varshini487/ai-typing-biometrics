# AI Typing Biometrics — Keystroke Dynamics for Continuous Authentication

## Overview

A biometric authentication system that identifies users by their **typing patterns** — dwell time (key press duration), flight time (between keys), rhythm consistency, error patterns, and burst behavior. Combines a **simulated LSTM** (2-layer, implemented from scratch in NumPy) with **statistical template matching** (Mahalanobis-like distance) in an ensemble (60% LSTM + 40% Statistical).

## Key Features

- **32-dimensional biometric feature extraction** — dwell/flight statistics, per-key and per-digraph timing, error metrics, rhythm autocorrelation, burst patterns
- **LSTM from scratch** — forget gate, input gate, output gate, candidate gate, full BPTT (backpropagation through time), softmax classifier
- **Continuous authentication** — sliding window monitoring to detect unauthorized typist in real-time
- **20-user biometric system** — learns unique typing profiles per user with per-bigram digraph timing offsets
- **FAR/FRR analysis** — False Accept Rate and False Reject Rate evaluation with configurable authentication threshold

## Architecture

```
Keystroke Data → Feature Extraction (32-dim) + Sequence Extraction (dwell/flight)
                ↓                                        ↓
    Statistical Profiler (40%)              LSTM Network (60%)
    (Mahalanobis distance)                 (2-layer LSTM + softmax)
                ↓                                        ↓
                    Ensemble Fusion (weighted average)
                                ↓
                    User ID + Confidence Score
                                ↓
                    AUTHENTIC / ALERT (threshold-based)
```

## How It Works

1. **Keystroke Simulator**: Generates realistic typing data for 20 users with unique profiles (mean dwell, flight, rhythm consistency, error rate, burst tendency, pause frequency, per-bigram timing offsets)
2. **Feature Extraction**: 32 features per session — dwell/flight statistics, percentiles, coefficient of variation, typing speed, per-key dwell times (top common keys), digraph flight times (top 4 bigrams), error metrics, burst patterns, dwell-flight correlation, rhythm autocorrelation (lag-2)
3. **Statistical Profiler**: Template matching using Mahalanobis-like distance — each user has a mean template + per-feature standard deviation
4. **LSTM Network**: 2-layer LSTM with 64 hidden units — processes raw [dwell, flight] sequences, captures temporal patterns in typing rhythm, outputs softmax probabilities over 20 users
5. **Ensemble Fusion**: 60% LSTM + 40% Statistical — LSTM captures temporal dynamics, statistical model captures aggregate profile features
6. **Continuous Authentication**: Sliding window of 50 keystrokes, re-evaluates identity every 25 keystrokes, alerts if confidence drops below threshold

## Tech Stack

- **Python**, **NumPy** — LSTM (forward + backward + BPTT), softmax classifier, statistical profiling — all from scratch
- No PyTorch, TensorFlow, or scikit-learn

## Installation

```bash
pip install numpy
python app.py
```

## Interview Points

1. **"I built a biometric authentication system using keystroke dynamics with an ensemble of a from-scratch LSTM and statistical template matching — the LSTM processes raw [dwell, flight] timing sequences through 2 layers with forget/input/output/candidate gates and full backpropagation through time, capturing temporal rhythm patterns that aggregate features miss, while the statistical profiler provides an interpretable Mahalanobis-distance baseline using 32 hand-crafted biometric features."**

2. **"I designed a continuous authentication mode using sliding windows of 50 keystrokes that re-evaluates typist identity every 25 keystrokes — this is the key distinction from one-time authentication: it detects session hijacking where an impostor takes over mid-session, which is a real security threat in shared workstation environments."**

3. **"I extracted 32 biometric features including per-digraph timing offsets (bigram-specific flight times for 'th', 'he', 'in', 'er'), rhythm autocorrelation at lag-2, and burst pattern analysis — these capture the finding from keystroke dynamics research that users have consistent per-bigram timing patterns, not just overall typing speed, making the system more discriminative than naive mean-variance features."**
