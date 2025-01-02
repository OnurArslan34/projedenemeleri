"""
feature_extraction.py
---------------------

Bu modül, ses dosyası yollarını alarak librosa ile çeşitli özellikler (MFCC,
spektral özellikler vb.) hesaplar ve bunların istatistiksel özetlerini 
bir DataFrame olarak döndürür.
"""

import os
import logging
import numpy as np
import pandas as pd
import librosa

from scipy.stats import kurtosis, skew

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract_statistical_features(feature_array, feature_name):
    """
    Belirli bir özelliğin (feature_array) mean, std, min, max, 
    kurtosis ve skew istatistiklerini hesaplar.
    feature_array genellikle (1, n_frames) veya (n_mfcc, n_frames) boyutunda olur.
    """
    # Tek boyuta indir
    feature_array_flat = np.ravel(feature_array)

    return {
        f'{feature_name}_mean': np.mean(feature_array_flat),
        f'{feature_name}_std': np.std(feature_array_flat),
        f'{feature_name}_min': np.min(feature_array_flat),
        f'{feature_name}_max': np.max(feature_array_flat),
        f'{feature_name}_kurtosis': kurtosis(feature_array_flat),
        f'{feature_name}_skew': skew(feature_array_flat)
    }

def extract_features(file_path):
    """
    Verilen bir ses dosyası (file_path) üzerinden çeşitli
    spektral, temporal ve cepstral özellikleri çıkarır.
    Dönen sonuç bir sözlük (dict) formatındadır.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)  # Orijinal sample rate
    except Exception as e:
        logging.warning(f"Dosya okunamadı: {file_path}, Hata: {e}")
        return None

    # Çok kısa veya bozuk ses için
    if y is None or len(y) < 10:
        logging.warning(f"Ses dosyası çok kısa/bozuk: {file_path}")
        return None

    features = {}

    # 1) Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.update(extract_statistical_features(spectral_centroid, 'spectral_centroid'))

    # 2) Spectral Flux (onset_strength)
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    # Tek boyuta indir: reshape(1, -1) veya ravel
    features.update(extract_statistical_features(spectral_flux, 'spectral_flux'))

    # 3) Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.update(extract_statistical_features(spectral_bandwidth, 'spectral_bandwidth'))

    # 4) Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    features.update(extract_statistical_features(spectral_flatness, 'spectral_flatness'))

    # 5) Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.update(extract_statistical_features(spectral_rolloff, 'spectral_rolloff'))

    # 6) Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.update(extract_statistical_features(spectral_contrast, 'spectral_contrast'))

    # 7) Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    features.update(extract_statistical_features(zero_crossing_rate, 'zero_crossing_rate'))

    # 8) RMS (Root Mean Square)
    rms = librosa.feature.rms(y=y)
    features.update(extract_statistical_features(rms, 'rms'))

    # 9) Chromagram
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.update(extract_statistical_features(chroma_stft, 'chroma_stft'))

    # 10) Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    # 11) MFCC (örn. 20 adet)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(mfcc.shape[0]):
        features.update(extract_statistical_features(mfcc[i, :], f'mfcc_{i+1}'))

    return features

def create_features_dataframe(df_paths):
    """
    df_paths: 'file_path' ve 'label' sütunlarını içeren bir DataFrame.
    Bu fonksiyon, her dosya için extract_features yapar ve 
    hepsini bir araya getirerek bir DataFrame döndürür.
    """
    all_data = []
    for _, row in df_paths.iterrows():
        file_path = row["file_path"]
        label = row["label"]

        feat_dict = extract_features(file_path)
        if feat_dict is None:
            # Özellik çıkarmada hata oluştuysa atla
            continue

        feat_dict["file_path"] = file_path
        feat_dict["label"] = label
        all_data.append(feat_dict)

    return pd.DataFrame(all_data)
