import gradio as gr
import numpy as np
import pandas as pd
import joblib
import librosa
import os
import uuid
import soundfile as sf

from feature_extraction import extract_features  # <-- Eğitimde kullandığınız fonksiyon
# Bu fonksiyonun tam set features (175 sütun vb.) ürettiğini varsayıyoruz.

MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

EMOTION_SCORES = {
    "angry": -2,
    "sad": -1,
    "calm": +1,
    "happy": +2
}

# 1) Model yükleme
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

# 2) Ses chunklama
def chunk_audio(y, sr, chunk_duration=3.0):
    audio_chunks = []
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    start = 0.0
    while start < total_duration:
        end = start + chunk_duration
        if end > total_duration:
            end = total_duration
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_chunk = y[start_sample:end_sample]
        audio_chunks.append((y_chunk, start, end))
        start += chunk_duration
    return audio_chunks

# 3) Her chunk için predict
def predict_chunk_emotion(y_chunk, sr):
    """
    Burada chunk verisini geçici dosyaya kaydedip,
    oradan 'extract_features' fonksiyonunu çağırıyoruz.
    """
    if len(y_chunk) < 10:
        return None

    # Temp wav dosyası
    temp_filename = f"temp_chunk_{uuid.uuid4()}.wav"
    sf.write(temp_filename, y_chunk, sr)

    feat_dict = extract_features(temp_filename)
    
    # Dosyayı temizle
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    if feat_dict is None:
        return None
    
    # Model pipeline
    df_features = pd.DataFrame([feat_dict])
    # Feature name uyarısını önlemek için, 
    # train aşamasındaki sütun isimleriyle aynı sırayı koruduğunuzdan emin olun.
    X_scaled = scaler.transform(df_features)  
    y_pred = model.predict(X_scaled)
    emotion = label_encoder.inverse_transform(y_pred)[0]
    return emotion

def analyze_audio(audio_file):
    if audio_file is None:
        return None, None
    
    y, sr = librosa.load(audio_file, sr=None)
    audio_chunks = chunk_audio(y, sr, 3.0)
    
    rows = []
    scores = []

    for (chunk_data, start_sec, end_sec) in audio_chunks:
        emotion = predict_chunk_emotion(chunk_data, sr)
        if emotion is None:
            continue
        rows.append({
            "Başlangıç (sn)": round(start_sec, 2),
            "Bitiş (sn)": round(end_sec, 2),
            "Duygu": emotion
        })
        score = EMOTION_SCORES.get(emotion.lower(), 0)
        scores.append(score)
    
    df = pd.DataFrame(rows)
    if len(scores) == 0:
        final_rating = 3
    else:
        mean_score = np.mean(scores)  # -2..+2
        rating = (mean_score + 2) / 4 * 4 + 1  
        final_rating = max(1, min(5, rating))
    final_rating = round(final_rating, 1)
    
    return df, final_rating

def create_star_string(rating):
    full_star = "★"
    empty_star = "☆"
    rounded = int(round(rating))
    return full_star * rounded + empty_star * (5 - rounded)

def inference_pipeline(audio_file):
    df, final_rating = analyze_audio(audio_file)
    if df is None:
        return (
            pd.DataFrame({"HATA": ["Ses dosyası yüklenmedi veya okunamadı"]}),
            "Bilgi Yok",
            "Bilgi Yok",
        )
    star_str = create_star_string(final_rating)
    rating_text = f"{final_rating} / 5"
    return df, rating_text, star_str

with gr.Blocks() as demo:
    gr.Markdown("# Müşteri Hizmetleri Görüşme Analizi (Chunk Bazlı)")
    
    with gr.Row():
        audio_input = gr.Audio(label="Ses Yükle", type="filepath")
        submit_btn = gr.Button("Analiz Et")
    
    result_dataframe = gr.Dataframe(
        headers=["Başlangıç (sn)", "Bitiş (sn)", "Duygu"], interactive=False
    )
    final_rating_str = gr.Textbox(label="Nihai Derecelendirme (Sayısal)")
    final_rating_stars = gr.Textbox(label="Nihai Derecelendirme (Yıldız)")
    
    submit_btn.click(
        fn=inference_pipeline,
        inputs=[audio_input],
        outputs=[result_dataframe, final_rating_str, final_rating_stars]
    )

if __name__ == "__main__":
    demo.launch()
