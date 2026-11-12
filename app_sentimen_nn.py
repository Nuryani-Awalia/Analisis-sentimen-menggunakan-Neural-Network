import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud

st.set_page_config(page_title="Sentimen Pemadaman Listrik", page_icon="💡", layout="wide")

# ------------------------------------------------------------------
# Konfigurasi: kamus kata & stopword (silakan tambah sendiri)
# ------------------------------------------------------------------
KATA_POSITIF = {
    "menyala", "nyala", "pulih", "normal", "kembali", "lancar", "aman", "stabil",
    "terima kasih", "apresiasi", "sigap", "cepat", "tertangani", "teratasi",
    "berhasil", "membaik", "bantuan", "lega", "siaga", "kompensasi", "handal",
    "puas", "semangat", "dukungan", "bersyukur", "alhamdulillah",
}
KATA_NEGATIF = {
    "padam", "mati", "gelap", "gangguan", "mengganggu", "terganggu", "parah",
    "lama", "kecewa", "marah", "rugi", "kerugian", "rusak", "gagal", "buruk",
    "lumpuh", "keluhan", "sengsara", "menderita", "panas", "byarpet", "drop",
    "protes", "kesal", "capek", "tersiksa", "sering",
}
STOPWORD = {
    "yang", "di", "ke", "dan", "dari", "untuk", "dengan", "ini", "itu", "pada",
    "adalah", "atau", "juga", "karena", "akan", "ada", "tidak", "sudah", "saya",
    "kami", "kita", "mereka", "dalam", "oleh", "bisa", "udah", "nya", "aja",
    "yg", "dgn", "utk", "tp", "sih", "lah", "kok", "ya", "kan", "rt", "amp",
    "para", "sebagai", "lagi", "masih", "banget", "gak", "nggak", "ga",
}
LABEL_ORDER = ["positif", "netral", "negatif"]
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}
WARNA_CLOUD = {"positif": "Greens", "netral": "Blues", "negatif": "Reds"}
WARNA_BAR = {"positif": "#2e9e5b", "netral": "#4a90d9", "negatif": "#d64545"}


# ------------------------------------------------------------------
# Langkah 3 - Pembersihan teks
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"@\w+", " ", text)                    # mention
    text = re.sub(r"#\w+", " ", text)                    # hashtag
    text = re.sub(r"\bRT\b", " ", text)                  # tanda retweet
    text = text.lower()                                  # case folding
    text = re.sub(r"[^a-z\s]", " ", text)                # simbol & angka
    text = re.sub(r"\s+", " ", text).strip()             # spasi berlebih
    return text


# ------------------------------------------------------------------
# Langkah 4 - Pelabelan rule-based
# ------------------------------------------------------------------
def hitung_kata(clean: str, kamus: set) -> int:
    return sum(len(re.findall(rf"\b{re.escape(k)}\b", clean)) for k in kamus)


def label_sentimen(clean: str) -> str:
    pos = hitung_kata(clean, KATA_POSITIF)
    neg = hitung_kata(clean, KATA_NEGATIF)
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"


ALIAS_KOLOM = {
    "created_at": ["created_at", "created at", "date", "tanggal", "waktu", "time", "timestamp", "datetime"],
    "username": ["username", "user_name", "user", "screen_name", "nama", "author", "name"],
    "full_text": ["full_text", "text", "tweet", "content", "isi", "komentar", "comment", "teks", "isi_tweet"],
}


def baca_file(file) -> pd.DataFrame:
    nama = str(getattr(file, "name", file)).lower()
    if nama.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    # sep=None: otomatis mendeteksi pemisah (koma, titik koma, tab)
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            return pd.read_csv(file, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("File tidak dapat dibaca (encoding tidak dikenali).")


@st.cache_data(show_spinner=False)
def muat_dan_proses(file) -> pd.DataFrame:
    df = baca_file(file)
    df.columns = [str(c).strip() for c in df.columns]
    peta = {}
    for target, alias in ALIAS_KOLOM.items():
        for c in df.columns:
            if c.lower() in alias:
                peta[c] = target
                break
    df = df.rename(columns=peta)
    kolom = {"created_at", "username", "full_text"}
    if "username" not in df.columns and "full_text" in df.columns:
        df["username"] = "tidak_diketahui"
    if "created_at" not in df.columns and "full_text" in df.columns:
        df["created_at"] = pd.NaT
    if not kolom.issubset(df.columns):
        raise ValueError(f"Kolom 'full_text' (isi tweet) tidak ditemukan. Kolom yang terbaca di file: {list(df.columns)}")
    df = df[["created_at", "username", "full_text"]].dropna(subset=["full_text"]).copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["clean_text"] = df["full_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].drop_duplicates(subset="clean_text")
    df["sentimen"] = df["clean_text"].apply(label_sentimen)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# Langkah 7 & 8 - Model Neural Network & evaluasi
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Melatih model Neural Network...")
def latih_model(texts: tuple, labels: tuple, hidden: int, max_iter: int):
    X_teks, y = list(texts), list(labels)
    counts = pd.Series(y).value_counts()
    strat = y if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_teks, y, test_size=0.2, random_state=42, stratify=strat)

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOPWORD))
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)

    model = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu",
                          solver="adam", max_iter=max_iter, random_state=42)
    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xte)

    labels_ada = [l for l in LABEL_ORDER if l in set(y)]
    p, r, f, _ = precision_recall_fscore_support(
        y_te, y_pred, average="weighted", zero_division=0)
    hasil = {
        "akurasi": accuracy_score(y_te, y_pred),
        "presisi": p, "recall": r, "f1": f,
        "cm": confusion_matrix(y_te, y_pred, labels=labels_ada),
        "labels": labels_ada,
        "report": pd.DataFrame(classification_report(
            y_te, y_pred, labels=labels_ada, output_dict=True, zero_division=0)).T,
        "n_train": len(X_tr), "n_test": len(X_te),
        "loss_curve": model.loss_curve_,
    }
    return model, vec, hasil


def buat_wordcloud(teks: str, cmap: str):
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap=cmap, stopwords=STOPWORD, max_words=100).generate(teks)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


# ------------------------------------------------------------------
# Langkah 2 - Antarmuka dashboard
# ------------------------------------------------------------------
st.title("💡 Analisis Sentimen Pemadaman Listrik Bergilir / Blackout")
st.caption("Pipeline: Crawling → Cleansing → Pelabelan Rule-Based → TF-IDF → Neural Network (MLP)")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    upload = st.file_uploader("Upload CSV / Excel (opsional)", type=["csv", "xlsx"])
    st.caption("Jika kosong, aplikasi memakai `tweet_data.csv` (atau `tweet_data.xlsx`) di folder yang sama.")
    hidden = st.slider("Neuron hidden layer", 10, 300, 100, step=10)
    max_iter = st.slider("Max iterasi", 100, 1000, 300, step=50)

PATH_DATA = r"C:\Users\lenovo\Downloads\praktikum_sentimen_listrik\Tweet_Data.csv"
if os.path.exists(PATH_DATA):
    sumber_default = PATH_DATA
elif os.path.exists("tweet_data.csv"):
    sumber_default = "tweet_data.csv"
else:
    sumber_default = "tweet_data.xlsx"
try:
    df = muat_dan_proses(upload if upload else sumber_default)
except FileNotFoundError:
    st.error("File `tweet_data.csv` tidak ditemukan. Lakukan crawling (lihat PANDUAN.md) "
             "atau jalankan `python generate_sample_data.py` untuk data contoh.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

if len(df) < 300:
    st.warning(f"Data hanya {len(df)} baris setelah dibersihkan; syarat praktikum minimal 300 tweet.")

tab_data, tab_dist, tab_wc, tab_eval, tab_pred = st.tabs(
    ["📄 Data", "📊 Distribusi", "☁️ WordCloud", "🧠 Model & Evaluasi", "✍️ Prediksi"])

# --- Data ---
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah tweet", len(df))
    c2.metric("Jumlah pengguna unik", df["username"].nunique())
    rentang = (f"{df['created_at'].min():%d %b %Y} – {df['created_at'].max():%d %b %Y}"
               if df["created_at"].notna().any() else "-")
    c3.metric("Rentang waktu", rentang)
    st.dataframe(df[["created_at", "username", "full_text", "clean_text", "sentimen"]],
                 use_container_width=True, height=400)
    st.download_button("⬇️ Unduh data berlabel", df.to_csv(index=False).encode("utf-8"),
                       "tweet_berlabel.csv", "text/csv")

# --- Distribusi ---
with tab_dist:
    dist = df["sentimen"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
    warna = [WARNA_BAR[l] for l in dist.index]
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        bars = ax.bar(dist.index, dist.values, color=warna)
        ax.bar_label(bars)
        ax.set_title("Jumlah Tweet per Sentimen")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(dist.values, labels=dist.index, colors=warna, autopct="%1.1f%%", startangle=90)
        ax.set_title("Proporsi Sentimen")
        st.pyplot(fig)
    st.table(dist.rename("jumlah").to_frame())

# --- WordCloud ---
with tab_wc:
    cols = st.columns(3)
    for col, lab in zip(cols, LABEL_ORDER):
        with col:
            st.subheader(f"{EMOJI[lab]} {lab.capitalize()}")
            teks = " ".join(df.loc[df["sentimen"] == lab, "clean_text"])
            if teks.strip():
                st.pyplot(buat_wordcloud(teks, WARNA_CLOUD[lab]))
            else:
                st.info("Tidak ada data pada kelas ini.")

# --- Model & Evaluasi ---
if df["sentimen"].nunique() < 2:
    with tab_eval:
        st.error("Label hanya 1 kelas, model tidak dapat dilatih. Perbaiki kamus kata atau tambah data.")
    st.stop()

model, vec, hasil = latih_model(tuple(df["clean_text"]), tuple(df["sentimen"]), hidden, max_iter)

with tab_eval:
    st.write(f"Data latih: **{hasil['n_train']}** | Data uji: **{hasil['n_test']}** (rasio 80:20)")
    m = st.columns(4)
    m[0].metric("Akurasi", f"{hasil['akurasi']:.3f}")
    m[1].metric("Presisi (weighted)", f"{hasil['presisi']:.3f}")
    m[2].metric("Recall (weighted)", f"{hasil['recall']:.3f}")
    m[3].metric("F1-Score (weighted)", f"{hasil['f1']:.3f}")

    ringkas = pd.DataFrame({
        "Metrik": ["Akurasi", "Presisi", "Recall", "F1-Score"],
        "Nilai": [hasil["akurasi"], hasil["presisi"], hasil["recall"], hasil["f1"]]})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Grafik Metrik")
        fig, ax = plt.subplots()
        bars = ax.bar(ringkas["Metrik"], ringkas["Nilai"], color="#4a90d9")
        ax.bar_label(bars, fmt="%.3f")
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)
    with c2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm, labs = hasil["cm"], hasil["labels"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labs)), labs)
        ax.set_yticks(range(len(labs)), labs)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        st.pyplot(fig)

    st.subheader("Laporan per Kelas")
    st.dataframe(hasil["report"].round(3), use_container_width=True)

    with st.expander("Kurva loss training"):
        fig, ax = plt.subplots()
        ax.plot(hasil["loss_curve"])
        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Loss")
        st.pyplot(fig)

    st.info("Catatan: label dibuat dari aturan kamus kata, sehingga model belajar meniru aturan "
            "tersebut. Akurasi tinggi tidak berarti model memahami sentimen manusia sebenarnya.")

# --- Prediksi ---
with tab_pred:
    teks_baru = st.text_area("Masukkan teks (misal tweet baru):",
                             "Listrik sudah menyala kembali, warga lega dan terima kasih kepada petugas PLN")
    if st.button("Prediksi", type="primary"):
        bersih = clean_text(teks_baru)
        if not bersih:
            st.warning("Teks kosong setelah dibersihkan.")
        else:
            proba = model.predict_proba(vec.transform([bersih]))[0]
            kelas = model.classes_
            pred = kelas[int(np.argmax(proba))]
            st.markdown(f"### {EMOJI[pred]} Sentimen: **{pred.upper()}**")
            st.caption(f"Teks bersih: `{bersih}`")
            dfp = pd.DataFrame({"sentimen": kelas, "probabilitas": proba})
            dfp["sentimen"] = dfp["sentimen"].map(lambda s: f"{EMOJI[s]} {s}")
            st.dataframe(dfp.style.format({"probabilitas": "{:.2%}"}), hide_index=True)
            st.bar_chart(dfp.set_index("sentimen"))