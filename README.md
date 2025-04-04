# Audio Deepfake Detection

This repository provides an end-to-end framework for detecting deepfake audio using advanced feature extraction and deep learning techniques. The project leverages a hybrid CNN-BiLSTM model with an attention mechanism to robustly distinguish between real and manipulated audio samples.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [References](#references)
- [License](#license)

## Overview

Audio deepfakes have emerged as a significant threat to digital trust, with the potential to disrupt security and communications systems. This project develops a robust detection system by integrating classical feature extraction methods with modern deep learning architectures. By combining MFCC-based features, Mel spectrograms, and additional spectral features with a CNN-BiLSTM hybrid model enhanced by an attention mechanism, this approach offers a promising solution for both research and real-world applications.

## Features

- **Hybrid Feature Extraction:** Combines MFCC (40 coefficients), Mel spectrograms, and additional spectral features (e.g., spectral centroid, bandwidth, contrast, rolloff, chroma, tonnetz, zero-crossing rate, and RMSE).
- **Deep Learning Model:** A CNN-BiLSTM hybrid architecture that leverages an attention mechanism to focus on critical parts of the audio signal.
- **Data Augmentation:** Uses techniques such as time-stretching, pitch-shifting, noise injection, and SpecAugment to improve model generalizability.
- **Balanced Dataset Preparation:** Automated oversampling to address class imbalance between real and fake audio samples.
- **Evaluation Metrics:** Detailed performance evaluation using accuracy, precision, recall, F1-score, and confusion matrices.

## Datasets

The primary dataset used for this project is the **SceneFake** dataset, available on Kaggle:

- [SceneFake Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/scenefake)

This dataset contains a variety of audio scenes that are manipulated (deepfake) and genuine, providing a diverse set of samples for training and evaluation.

## Installation

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- librosa
- scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn
- tqdm

### Clone the Repository
```bash
git clone https://github.com/yourusername/Audio-Deepfake-Detection.git
cd Audio-Deepfake-Detection
