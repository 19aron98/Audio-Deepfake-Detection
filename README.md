# Audio Deepfake Detection

This repository provides an end-to-end framework for detecting deepfake audio using advanced feature extraction and deep learning techniques. The project leverages a hybrid CNN-BiLSTM model with an attention mechanism to robustly distinguish between real and manipulated audio samples.

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

### Install Dependencies
```bash
pip install -r requirements.txt
```


### Model Architecture
The core of this project is a hybrid deep learning model that combines:
- CNN Layers: For local feature extraction from the input audio.
- Bidirectional LSTM Layers: To capture temporal dependencies in both forward and backward directions.
- Attention Mechanism: To focus on the most informative parts of the sequence, enhancing classification accuracy.
- Fully Connected Layers: With dropout regularization to prevent overfitting, followed by a final sigmoid layer for binary classification.

```bash
def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # CNN layer
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # BiLSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.1)))(x)
    x = Dropout(0.3)(x)

    # Attention mechanism
    attention = Attention()([x, x])
    attention = GlobalMaxPooling1D()(attention)

    # Fully connected layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(attention)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model
```
### Training & Evaluation
The model is trained using the Adam optimizer with binary cross-entropy loss. Key performance metrics include:
- Validation Accuracy: ~90.90%
```bash
precision    recall  f1-score   support

           0       0.89      0.94      0.91      3000
           1       0.93      0.88      0.91      3000

    accuracy                           0.91      6000
   macro avg       0.91      0.91      0.91      6000
weighted avg       0.91      0.91      0.91      6000
```

- Test Accuracy: ~90.38%
```bash
precision    recall  f1-score   support

           0       0.87      0.95      0.91      3000
           1       0.94      0.86      0.90      3000

    accuracy                           0.90      6000
   macro avg       0.91      0.90      0.90      6000
weighted avg       0.91      0.90      0.90      6000
```
Detailed Metrics: Precision, recall, and F1-scores for both classes are reported, ensuring balanced performance.


