import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud


# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Sentimen Pemadaman Listrik",
    page_icon="💡",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================================
# KAMUS KATA DAN STOPWORD
# ==========================================================
KATA_POSITIF = {
    "menyala", "nyala", "pulih", "normal", "kembali", "lancar",
    "aman", "stabil", "terima kasih", "apresiasi", "sigap", "cepat",
    "tertangani", "teratasi", "berhasil", "membaik", "bantuan",
    "lega", "siaga", "kompensasi", "handal", "puas", "semangat",
    "dukungan", "bersyukur", "alhamdulillah",
}

KATA_NEGATIF = {
    "padam", "mati", "gelap", "gangguan", "mengganggu", "terganggu",
    "parah", "lama", "kecewa", "marah", "rugi", "kerugian", "rusak",
    "gagal", "buruk", "lumpuh", "keluhan", "sengsara", "menderita",
    "panas", "byarpet", "drop", "protes", "kesal", "capek",
    "tersiksa", "sering",
}

STOPWORD = {
    "yang", "di", "ke", "dan", "dari", "untuk", "dengan", "ini", "itu",
    "pada", "adalah", "atau", "juga", "karena", "akan", "ada", "tidak",
    "sudah", "saya", "kami", "kita", "mereka", "dalam", "oleh", "bisa",
    "udah", "nya", "aja", "yg", "dgn", "utk", "tp", "sih", "lah",
    "kok", "ya", "kan", "rt", "amp", "para", "sebagai", "lagi",
    "masih", "banget", "gak", "nggak", "ga",
}

LABEL_ORDER = ["positif", "netral", "negatif"]
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}
WARNA_CLOUD = {"positif": "Greens", "netral": "Blues", "negatif": "Reds"}
WARNA_BAR = {
    "positif": "#2e9e5b",
    "netral": "#4a90d9",
    "negatif": "#d64545",
}


# ==========================================================
# CLEANSING TEKS
# ==========================================================
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URL
    text = re.sub(r"@\w+", " ", text)                   # Mention
    text = re.sub(r"#\w+", " ", text)                   # Hashtag
    text = re.sub(r"\bRT\b", " ", text)                 # Retweet
    text = text.lower()                                 # Case folding
    text = re.sub(r"[^a-z\s]", " ", text)                # Simbol dan angka
    text = re.sub(r"\s+", " ", text).strip()             # Spasi berlebih
    return text


# ==========================================================
# PELABELAN RULE-BASED
# ==========================================================
def hitung_kata(clean: str, kamus: set) -> int:
    return sum(
        len(re.findall(rf"\b{re.escape(k)}\b", clean))
        for k in kamus
    )


def label_sentimen(clean: str) -> str:
    positif = hitung_kata(clean, KATA_POSITIF)
    negatif = hitung_kata(clean, KATA_NEGATIF)

    if positif > negatif:
        return "positif"
    if negatif > positif:
        return "negatif"
    return "netral"


# ==========================================================
# PEMETAAN NAMA KOLOM DATASET
# ==========================================================
ALIAS_KOLOM = {
    "created_at": [
        "created_at", "created at", "date", "tanggal", "waktu",
        "time", "timestamp", "datetime",
    ],
    "username": [
        "username", "user_name", "user", "screen_name",
        "nama", "author", "name",
    ],
    "full_text": [
        "full_text", "text", "tweet", "content", "isi",
        "komentar", "comment", "teks", "isi_tweet",
    ],
}


def baca_file(file) -> pd.DataFrame:
    nama = str(getattr(file, "name", file)).lower()

    if nama.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    # Mendeteksi pemisah CSV secara otomatis dan mencoba beberapa encoding.
    error_terakhir = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            return pd.read_csv(
                file,
                sep=None,
                engine="python",
                encoding=encoding,
            )
        except UnicodeDecodeError as error:
            error_terakhir = error
            continue

    raise ValueError(
        f"File tidak dapat dibaca karena encoding tidak dikenali: {error_terakhir}"
    )


@st.cache_data(show_spinner=False)
def muat_dan_proses(file) -> pd.DataFrame:
    df = baca_file(file)
    df.columns = [str(kolom).strip() for kolom in df.columns]

    peta = {}
    for target, alias in ALIAS_KOLOM.items():
        alias_lower = [nama.lower() for nama in alias]
        for kolom in df.columns:
            if kolom.lower() in alias_lower:
                peta[kolom] = target
                break

    df = df.rename(columns=peta)

    # Username dan tanggal boleh tidak tersedia dalam dataset.
    if "username" not in df.columns and "full_text" in df.columns:
        df["username"] = "tidak_diketahui"

    if "created_at" not in df.columns and "full_text" in df.columns:
        df["created_at"] = pd.NaT

    if "full_text" not in df.columns:
        raise ValueError(
            "Kolom teks tidak ditemukan. Gunakan salah satu nama kolom: "
            "full_text, text, tweet, content, isi, komentar, atau teks. "
            f"Kolom yang terbaca: {list(df.columns)}"
        )

    df = df[["created_at", "username", "full_text"]].dropna(
        subset=["full_text"]
    ).copy()

    df["full_text"] = df["full_text"].astype(str)
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
    )
    df["clean_text"] = df["full_text"].apply(clean_text)

    df = df[df["clean_text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset="clean_text")

    df["sentimen"] = df["clean_text"].apply(label_sentimen)
    return df.reset_index(drop=True)


# ==========================================================
# PELATIHAN NEURAL NETWORK DAN EVALUASI
# ==========================================================
@st.cache_resource(show_spinner="Melatih model Neural Network...")
def latih_model(
    texts: tuple,
    labels: tuple,
    hidden: int,
    max_iter: int,
):
    X_teks = list(texts)
    y = list(labels)

    if len(X_teks) < 5:
        raise ValueError(
            "Data terlalu sedikit untuk dibagi menjadi data latih dan data uji. "
            "Tambahkan setidaknya 5 data teks."
        )

    counts = pd.Series(y).value_counts()
    stratifikasi = y if len(counts) > 1 and counts.min() >= 2 else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_teks,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratifikasi,
    )

    if len(set(y_tr)) < 2:
        raise ValueError(
            "Data latih hanya memiliki satu kelas sentimen. "
            "Tambahkan data dengan variasi sentimen positif, netral, dan negatif."
        )

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(STOPWORD),
    )
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)

    if Xtr.shape[1] == 0:
        raise ValueError(
            "Tidak ada fitur teks yang dapat dipelajari. "
            "Periksa isi dataset setelah cleansing."
        )

    model = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
    )
    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xte)

    labels_ada = [label for label in LABEL_ORDER if label in set(y)]
    presisi, recall, f1, _ = precision_recall_fscore_support(
        y_te,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    hasil = {
        "akurasi": accuracy_score(y_te, y_pred),
        "presisi": presisi,
        "recall": recall,
        "f1": f1,
        "cm": confusion_matrix(y_te, y_pred, labels=labels_ada),
        "labels": labels_ada,
        "report": pd.DataFrame(
            classification_report(
                y_te,
                y_pred,
                labels=labels_ada,
                output_dict=True,
                zero_division=0,
            )
        ).T,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "loss_curve": model.loss_curve_,
    }

    return model, vec, hasil


# ==========================================================
# WORDCLOUD
# ==========================================================
def buat_wordcloud(teks: str, cmap: str):
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=cmap,
        stopwords=STOPWORD,
        max_words=100,
    ).generate(teks)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


# ==========================================================
# ANTARMUKA DASHBOARD
# ==========================================================
st.title("💡 Analisis Sentimen Pemadaman Listrik Bergilir / Blackout")
st.caption(
    "Pipeline: Crawling → Cleansing → Pelabelan Rule-Based "
    "→ TF-IDF → Neural Network (MLP)"
)

with st.sidebar:
    st.header("⚙️ Pengaturan")
    upload = st.file_uploader(
        "Upload CSV / Excel (opsional)",
        type=["csv", "xlsx", "xls"],
    )
    st.caption(
        "Jika tidak upload file, aplikasi akan mencari tweet_data.csv "
        "atau tweet_data.xlsx di repository."
    )
    hidden = st.slider("Neuron hidden layer", 10, 300, 100, step=10)
    max_iter = st.slider("Max iterasi", 100, 1000, 300, step=50)


# ==========================================================
# MEMBACA DATASET - MENDUKUNG STREAMLIT CLOUD DAN LOKAL
# ==========================================================
# Tidak menggunakan path absolut dari laptop Windows.
# Nama file diperiksa satu per satu karena Linux membedakan huruf besar/kecil.
daftar_file = [
    "tweet_data.csv",
    "Tweet_Data.csv",
    "tweet_data.xlsx",
    "Tweet_Data.xlsx",
    os.path.join("data", "tweet_data.csv"),
    os.path.join("data", "Tweet_Data.csv"),
    os.path.join("data", "tweet_data.xlsx"),
    os.path.join("data", "Tweet_Data.xlsx"),
]

sumber_default = None
for nama_file in daftar_file:
    lokasi = os.path.join(BASE_DIR, nama_file)
    if os.path.isfile(lokasi):
        sumber_default = lokasi
        break

try:
    if upload is not None:
        df = muat_dan_proses(upload)
    elif sumber_default is not None:
        df = muat_dan_proses(sumber_default)
    else:
        st.error("File dataset tidak ditemukan di repository GitHub.")
        st.info(
            "Unggah file CSV/Excel melalui bagian Upload di sidebar, "
            "atau tambahkan tweet_data.csv ke repository GitHub "
            "sejajar dengan app.py."
        )
        st.stop()

except (FileNotFoundError, OSError) as error:
    st.error(f"Dataset tidak dapat ditemukan atau dibuka: {error}")
    st.stop()
except ValueError as error:
    st.error(f"Kesalahan format dataset: {error}")
    st.stop()
except Exception as error:
    st.error(f"Terjadi kesalahan saat membaca dataset: {error}")
    st.stop()


if df.empty:
    st.error(
        "Tidak ada data teks yang tersisa setelah proses cleansing. "
        "Periksa kembali isi dataset."
    )
    st.stop()

if len(df) < 300:
    st.warning(
        f"Data hanya {len(df)} baris setelah dibersihkan; "
        "syarat praktikum minimal 300 tweet."
    )


tab_data, tab_dist, tab_wc, tab_eval, tab_pred = st.tabs(
    [
        "📄 Data",
        "📊 Distribusi",
        "☁️ WordCloud",
        "🧠 Model & Evaluasi",
        "✍️ Prediksi",
    ]
)


# ==========================================================
# TAB DATA
# ==========================================================
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah tweet", len(df))
    c2.metric("Jumlah pengguna unik", df["username"].nunique())

    rentang = (
        f"{df['created_at'].min():%d %b %Y} – "
        f"{df['created_at'].max():%d %b %Y}"
        if df["created_at"].notna().any()
        else "-"
    )
    c3.metric("Rentang waktu", rentang)

    st.dataframe(
        df[
            [
                "created_at",
                "username",
                "full_text",
                "clean_text",
                "sentimen",
            ]
        ],
        use_container_width=True,
        height=400,
    )

    st.download_button(
        "⬇️ Unduh data berlabel",
        df.to_csv(index=False).encode("utf-8"),
        "tweet_berlabel.csv",
        "text/csv",
    )


# ==========================================================
# TAB DISTRIBUSI
# ==========================================================
with tab_dist:
    dist = (
        df["sentimen"]
        .value_counts()
        .reindex(LABEL_ORDER)
        .fillna(0)
        .astype(int)
    )
    warna = [WARNA_BAR[label] for label in dist.index]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        bars = ax.bar(dist.index, dist.values, color=warna)
        ax.bar_label(bars)
        ax.set_title("Jumlah Tweet per Sentimen")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.pie(
            dist.values,
            labels=dist.index,
            colors=warna,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Proporsi Sentimen")
        st.pyplot(fig)
        plt.close(fig)

    st.table(dist.rename("jumlah").to_frame())


# ==========================================================
# TAB WORDCLOUD
# ==========================================================
with tab_wc:
    cols = st.columns(3)

    for col, label in zip(cols, LABEL_ORDER):
        with col:
            st.subheader(f"{EMOJI[label]} {label.capitalize()}")
            teks = " ".join(
                df.loc[df["sentimen"] == label, "clean_text"]
            )

            if teks.strip():
                try:
                    fig = buat_wordcloud(teks, WARNA_CLOUD[label])
                    st.pyplot(fig)
                    plt.close(fig)
                except ValueError:
                    st.info("Kosakata tidak cukup untuk membuat WordCloud.")
            else:
                st.info("Tidak ada data pada kelas ini.")


# ==========================================================
# TAB MODEL DAN EVALUASI
# ==========================================================
if df["sentimen"].nunique() < 2:
    with tab_eval:
        st.error(
            "Label hanya memiliki satu kelas. Model tidak dapat dilatih. "
            "Perbaiki kamus kata atau tambahkan data."
        )
    st.stop()

try:
    model, vec, hasil = latih_model(
        tuple(df["clean_text"]),
        tuple(df["sentimen"]),
        hidden,
        max_iter,
    )
except ValueError as error:
    with tab_eval:
        st.error(str(error))
    st.stop()
except Exception as error:
    with tab_eval:
        st.error(f"Model gagal dilatih: {error}")
    st.stop()


with tab_eval:
    st.write(
        f"Data latih: **{hasil['n_train']}** | "
        f"Data uji: **{hasil['n_test']}** (rasio 80:20)"
    )

    metrik = st.columns(4)
    metrik[0].metric("Akurasi", f"{hasil['akurasi']:.3f}")
    metrik[1].metric("Presisi (weighted)", f"{hasil['presisi']:.3f}")
    metrik[2].metric("Recall (weighted)", f"{hasil['recall']:.3f}")
    metrik[3].metric("F1-Score (weighted)", f"{hasil['f1']:.3f}")

    ringkas = pd.DataFrame(
        {
            "Metrik": ["Akurasi", "Presisi", "Recall", "F1-Score"],
            "Nilai": [
                hasil["akurasi"],
                hasil["presisi"],
                hasil["recall"],
                hasil["f1"],
            ],
        }
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Grafik Metrik")
        fig, ax = plt.subplots()
        bars = ax.bar(
            ringkas["Metrik"],
            ringkas["Nilai"],
            color="#4a90d9",
        )
        ax.bar_label(bars, fmt="%.3f")
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = hasil["cm"]
        labels = hasil["labels"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if cm[i, j] > cm.max() / 2
                        else "black"
                    ),
                )

        fig.colorbar(im)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Laporan per Kelas")
    st.dataframe(
        hasil["report"].round(3),
        use_container_width=True,
    )

    with st.expander("Kurva loss training"):
        fig, ax = plt.subplots()
        ax.plot(hasil["loss_curve"])
        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Loss")
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "Catatan: label dibuat dari aturan kamus kata, sehingga model "
        "belajar meniru aturan tersebut. Akurasi tinggi tidak berarti "
        "model memahami sentimen manusia sebenarnya."
    )


# ==========================================================
# TAB PREDIKSI
# ==========================================================
with tab_pred:
    teks_baru = st.text_area(
        "Masukkan teks (misal tweet baru):",
        "Listrik sudah menyala kembali, warga lega dan terima kasih "
        "kepada petugas PLN",
    )

    if st.button("Prediksi", type="primary"):
        bersih = clean_text(teks_baru)

        if not bersih:
            st.warning("Teks kosong setelah dibersihkan.")
        else:
            probabilitas = model.predict_proba(
                vec.transform([bersih])
            )[0]
            kelas = model.classes_
            prediksi = kelas[int(np.argmax(probabilitas))]

            st.markdown(
                f"### {EMOJI[prediksi]} Sentimen: "
                f"**{prediksi.upper()}**"
            )
            st.caption(f"Teks bersih: `{bersih}`")

            df_prediksi = pd.DataFrame(
                {
                    "sentimen": kelas,
                    "probabilitas": probabilitas,
                }
            )
            df_prediksi["sentimen"] = df_prediksi["sentimen"].map(
                lambda label: f"{EMOJI[label]} {label}"
            )

            st.dataframe(
                df_prediksi.style.format(
                    {"probabilitas": "{:.2%}"}
                ),
                hide_index=True,
            )
            st.bar_chart(
                df_prediksi.set_index("sentimen")
            )
