"""
main.py
-------

Bu dosya, tüm modülleri kullanarak projenin uçtan uca akışını yönetir:
1) data_preparation (dosya yol + label),
2) feature_extraction (özellik çıkarma),
3) train (model eğitimi).
"""

import logging
from data_preparation import create_dataframe_from_directory, save_dataframe_to_csv
from feature_extraction import create_features_dataframe
from train import train_model
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    # 1) Train ve Test veri setini oluştur (path + label)
    train_paths_df = create_dataframe_from_directory("train_sound", limit=None)
    test_paths_df = create_dataframe_from_directory("test_sound", limit=None)

    # Kaydetmek isterseniz
    save_dataframe_to_csv(train_paths_df, "train_dataset.csv")
    save_dataframe_to_csv(test_paths_df, "test_dataset.csv")

    # 2) Özellik çıkarma
    train_features_df = create_features_dataframe(train_paths_df)
    test_features_df = create_features_dataframe(test_paths_df)

    # (Opsiyonel) CSV'ye kaydet
    train_features_df.to_csv("train_features.csv", index=False)
    test_features_df.to_csv("test_features.csv", index=False)
    logging.info("Özellik çıkarımı tamamlandı ve CSV'lere kaydedildi.")

    # 3) Model Eğitimi
    train_model(
        train_features_df=train_features_df,
        test_features_df=test_features_df,
        model_output_path="xgb_model.pkl",
        scaler_output_path="scaler.pkl",
        encoder_output_path="label_encoder.pkl",
        do_hyperparam_tuning=True  # GridSearchCV ile parametre araması yapar
    )

if __name__ == "__main__":
    main()
