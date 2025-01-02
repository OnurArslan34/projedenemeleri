"""
data_preparation.py
-------------------

Bu modül, belirtilen dizindeki ses dosyalarının tam yolunu ve etiketi 
(klasör adından) alıp bir DataFrame oluşturur.
"""

import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_dataframe_from_directory(directory_path, limit=None):
    """
    Verilen dizindeki (directory_path) ses dosyalarının tam yolunu alır,
    klasör adını etiket olarak atar. 'limit' parametresi ile alınacak maksimum
    dosya sayısını isteğe göre kısıtlayabilirsiniz (None ise sınırsız).
    """
    logging.info(f"create_dataframe_from_directory => {directory_path}")

    paths = []
    labels = []

    for dirname, _, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            label = os.path.basename(dirname).lower()  # klasör adından

            paths.append(file_path)
            labels.append(label)

            if limit is not None and len(paths) == limit:
                break
        if limit is not None and len(paths) == limit:
            break

    df = pd.DataFrame({"file_path": paths, "label": labels})
    return df

def save_dataframe_to_csv(df, csv_name):
    """
    Bir pandas DataFrame'ini CSV dosyasına kaydeder.
    """
    df.to_csv(csv_name, index=False)
    logging.info(f"DataFrame kaydedildi: {csv_name}")
