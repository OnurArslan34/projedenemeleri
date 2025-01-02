"""
train.py
--------

Bu modül, train_features_df ve test_features_df DataFrame'lerini alarak 
XGBoost modelini eğitir, test seti üzerinde performansını değerlendirir. 
Arzu edilirse GridSearchCV ile hyperparameter tuning uygulanabilir.
"""

import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_model(
    train_features_df: pd.DataFrame,
    test_features_df: pd.DataFrame,
    model_output_path: str = "xgb_model.pkl",
    scaler_output_path: str = "scaler.pkl",
    encoder_output_path: str = "label_encoder.pkl",
    do_hyperparam_tuning: bool = False
):
    """
    train_features_df: Özellikleri çıkarılmış TRAIN DataFrame.
                      İçinde 'file_path', 'label' ve diğer sütunlar bulunmalı.
    test_features_df:  Özellikleri çıkarılmış TEST DataFrame.
    model_output_path: Eğitilen modelin kaydedileceği dosya adı.
    scaler_output_path: Ölçekleyicinin kaydedileceği dosya adı.
    encoder_output_path: Label encoder'ın kaydedileceği dosya adı.
    do_hyperparam_tuning: True ise GridSearchCV ile parametre araması yapılır.
    """

    if train_features_df.empty:
        logging.error("Train DataFrame boş. Eğitim gerçekleştirilemedi.")
        return None

    # Eğer test_features_df boşsa, sadece eğitim yapabilirsiniz
    if test_features_df.empty:
        logging.warning("Test DataFrame boş. Model test edilemeyecek.")

    # 1) X, y ayrıştırma
    X_train = train_features_df.drop(["file_path", "label"], axis=1)
    y_train = train_features_df["label"]

    X_test = test_features_df.drop(["file_path", "label"], axis=1)
    y_test = test_features_df["label"]

    # 2) Label Encoding
    # Tüm etiketlerin bir arada encode edilmesi genelde daha iyidir.
    combined_labels = pd.concat([y_train, y_test], ignore_index=True)
    label_encoder = LabelEncoder()
    label_encoder.fit(combined_labels)

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # 3) Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) XGBoost Model
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )

    if do_hyperparam_tuning:
        logging.info("Hyperparameter tuning başlatılıyor...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_scaled, y_train_encoded)
        best_model = grid_search.best_estimator_
        logging.info(f"En iyi parametreler: {grid_search.best_params_}")
    else:
        logging.info("Varsayılan XGBoost parametreleriyle model eğitiliyor...")
        xgb_model.fit(X_train_scaled, y_train_encoded)
        best_model = xgb_model

    # 5) Test Değerlendirme
    if not test_features_df.empty:
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        logging.info(f"Test Doğruluk Skoru: {accuracy:.4f}")

        # Sınıflandırma Raporu
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
        logging.info("Sınıflandırma Raporu:")
        logging.info("\n" + classification_report(y_test_decoded, y_pred_decoded))
    else:
        logging.warning("Test DataFrame olmadığı için performans ölçülemedi.")

    # 6) Modeli Kaydetme
    joblib.dump(best_model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    joblib.dump(label_encoder, encoder_output_path)

    logging.info(f"Model kaydedildi: {model_output_path}")
    logging.info(f"Scaler kaydedildi: {scaler_output_path}")
    logging.info(f"Label encoder kaydedildi: {encoder_output_path}")

    return best_model, scaler, label_encoder
    