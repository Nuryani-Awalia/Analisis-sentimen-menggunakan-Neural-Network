import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud

st.set_page_config(page_title="Sentimen Listrik Pontianak", page_icon="⚡", layout="wide")

# ----------------------------------------------------------------------------
# Konstanta
# ----------------------------------------------------------------------------
LABEL_ORDER = ["positif", "netral", "negatif"]
WARNA = {"positif": "#2e9e5b", "netral": "#4a90d9", "negatif": "#d64545"}
CMAP = {"positif": "Greens", "netral": "Blues", "negatif": "Reds"}
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}

ALIAS_KOLOM = {
    "created_at": ["created_at", "created at", "date", "tanggal", "waktu", "time", "timestamp", "datetime"],
    "username": ["username", "user_name", "user", "screen_name", "nama", "author", "name"],
    "full_text": ["full_text", "text", "tweet", "content", "isi", "komentar", "comment", "teks", "isi_tweet"],
}

KAMUS_SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak",
    "engga": "tidak", "tdk": "tidak", "gk": "tidak", "tak": "tidak",
    "udah": "sudah", "udh": "sudah", "sdh": "sudah", "dah": "sudah",
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "untk": "untuk",
    "tp": "tapi", "krn": "karena", "karna": "karena", "krna": "karena",
    "bgt": "sekali", "banget": "sekali", "bngt": "sekali", "jg": "juga", "jga": "juga",
    "sy": "saya", "gue": "saya", "gw": "saya", "aku": "saya", "aja": "saja", "doang": "saja",
    "emg": "memang", "emang": "memang", "blm": "belum", "belom": "belum",
    "lg": "lagi", "lgi": "lagi", "skrg": "sekarang", "skrang": "sekarang",
    "bs": "bisa", "bsa": "bisa", "hrs": "harus", "org": "orang", "orng": "orang",
    "kmrn": "kemarin", "kalo": "kalau", "klo": "kalau", "kl": "kalau",
    "dr": "dari", "pd": "pada", "mlm": "malam", "pagi2": "pagi", "sm": "sama",
    "makasih": "terima kasih", "terimakasih": "terima kasih", "trims": "terima kasih",
    "thanks": "terima kasih", "thx": "terima kasih", "tq": "terima kasih",
    "padem": "padam", "mati2": "mati", "idup": "hidup", "nyala": "menyala",
    "byarpet": "byarpet", "bgmn": "bagaimana", "gmn": "bagaimana", "gimana": "bagaimana",
    "tlg": "tolong", "plis": "tolong", "pls": "tolong", "mohon2": "mohon",
    "pln": "pln", "pemadamannya": "pemadaman", "listriknya": "listrik",
    # tambahan sesuai kosakata Tweet_Data.csv
    "yaelah": "aduh", "yaa": "ya", "yaaa": "ya", "duh": "aduh", "sumpah": "sungguh",
    "info": "informasi", "update": "pembaruan", "syukurlah": "bersyukur", "hidup": "menyala",
    "deh": "", "sih": "", "dong": "", "nih": "", "tuh": "", "lho": "", "kok": "",
}

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


# ----------------------------------------------------------------------------
# Langkah 3: pemrosesan teks
# ----------------------------------------------------------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"@\w+", " ", text)                    # mention
    text = re.sub(r"#\w+", " ", text)                    # hashtag
    text = re.sub(r"\bRT\b", " ", text)                  # tanda retweet
    text = text.lower()                                  # case folding
    text = re.sub(r"[^a-z\s]", " ", text)                # simbol & angka
    text = re.sub(r"\s+", " ", text).strip()             # spasi berlebih
    return text


def normalisasi_text(text):
    text = re.sub(r"(.)\1{2,}", r"\1", text)             # paraaah -> parah
    kata = [KAMUS_SLANG.get(k, k) for k in text.split()]
    return re.sub(r"\s+", " ", " ".join(kata)).strip()


# ----------------------------------------------------------------------------
# Langkah 4: pelabelan rule-based
# ----------------------------------------------------------------------------
def hitung_kata(teks, kamus):
    return sum(len(re.findall(rf"\b{re.escape(k)}\b", teks)) for k in kamus)


def label_sentimen(teks):
    pos, neg = hitung_kata(teks, KATA_POSITIF), hitung_kata(teks, KATA_NEGATIF)
    if pos > neg:
        return "positif"
    if neg > pos:
        return "negatif"
    return "netral"


# ----------------------------------------------------------------------------
# Membaca data
# ----------------------------------------------------------------------------
def baca_file(sumber, nama):
    if nama.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(sumber)
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            if hasattr(sumber, "seek"):
                sumber.seek(0)
            return pd.read_csv(sumber, sep=None, engine="python", encoding=enc)  # pemisah otomatis
        except UnicodeDecodeError:
            continue
    raise ValueError("File tidak dapat dibaca.")


def cari_file_default():
    folder = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()] if "__file__" in globals() else [os.getcwd()]
    for d in dict.fromkeys(folder):
        for nama in sorted(os.listdir(d)):
            if nama.lower() in ("tweet_data.csv", "tweet_data.xlsx"):
                return os.path.join(d, nama)
    return None


def rapikan_kolom(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    peta = {}
    for target, alias in ALIAS_KOLOM.items():
        for c in df.columns:
            if c.lower() in alias:
                peta[c] = target
                break
    df = df.rename(columns=peta)
    if "full_text" not in df.columns:
        raise ValueError(f"Kolom isi tweet tidak ditemukan. Kolom terbaca: {list(df.columns)}")
    if "username" not in df.columns:
        df["username"] = "tidak_diketahui"
    if "created_at" not in df.columns:
        df["created_at"] = pd.NaT
    df = df[["created_at", "username", "full_text"]].dropna(subset=["full_text"])
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


@st.cache_data(show_spinner="Memproses teks...")
def proses_data(df_mentah, hapus_duplikat):
    df = df_mentah.copy()
    df["clean_text"] = df["full_text"].apply(clean_text)
    df["normal_text"] = df["clean_text"].apply(normalisasi_text)
    df = df[df["normal_text"].str.len() > 0]
    if hapus_duplikat:
        df = df.drop_duplicates(subset="normal_text")
    df = df.reset_index(drop=True)
    df["sentimen"] = df["normal_text"].apply(label_sentimen)
    return df


# ----------------------------------------------------------------------------
# Langkah 7: model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Melatih model Neural Network...")
def latih_model(teks, label, hidden, max_iter):
    X, y = pd.Series(list(teks)), pd.Series(list(label))
    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat)
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOPWORD))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
                          max_iter=max_iter, random_state=42)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    return {"vec": vec, "model": model, "y_test": y_test.tolist(), "y_pred": list(y_pred),
            "n_train": len(X_train), "n_test": len(X_test), "n_fitur": Xtr.shape[1]}


def prediksi(teks, hasil_model):
    dibersihkan = clean_text(teks)
    bersih = normalisasi_text(dibersihkan)
    model, vec = hasil_model["model"], hasil_model["vec"]
    proba = model.predict_proba(vec.transform([bersih]))[0]
    label = model.classes_[int(np.argmax(proba))]
    return dibersihkan, bersih, label, dict(zip(model.classes_, proba))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def tampil_tabel(data, **kw):
    """st.dataframe lebar penuh; kompatibel dengan Streamlit lama maupun baru."""
    try:
        st.dataframe(data, width="stretch", **kw)
    except Exception:  # Streamlit versi lama belum mengenal width="stretch"
        st.dataframe(data, use_container_width=True, **kw)


st.title("⚡ Analisis Sentimen Pemadaman Listrik")
st.caption("Pipeline: cleansing → normalisasi → pelabelan rule-based → TF-IDF → Neural Network (MLP)")

with st.sidebar:
    st.header("Pengaturan")
    unggahan = st.file_uploader("Unggah dataset (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    hapus_duplikat = st.checkbox("Hapus tweet duplikat", value=True,
                                 help="Duplikat yang jatuh di data latih dan uji membuat evaluasi terlalu optimis.")
    hidden = st.slider("Neuron hidden layer", 10, 300, 100, step=10)
    max_iter = st.slider("Maksimum iterasi", 100, 1000, 300, step=50)
    st.caption("Aktivasi ReLU · Optimizer Adam · Rasio data latih:uji = 80:20")

# ----------------------------------------------------------------------------
# Memuat data
# ----------------------------------------------------------------------------
try:
    if unggahan is not None:
        mentah = baca_file(unggahan, unggahan.name)
        sumber = unggahan.name
    else:
        path = cari_file_default()
        if path is None:
            st.warning("Tweet_Data.csv tidak ditemukan. Unggah dataset lewat sidebar.")
            st.stop()
        mentah = baca_file(path, path)
        sumber = os.path.basename(path)
    mentah = rapikan_kolom(mentah)
except Exception as e:  # noqa: BLE001
    st.error(f"Gagal membaca data: {e}")
    st.stop()

df = proses_data(mentah, hapus_duplikat)
hasil_model = latih_model(tuple(df["normal_text"]), tuple(df["sentimen"]), hidden, max_iter)

tab_data, tab_dist, tab_wc, tab_eval, tab_pred = st.tabs(
    ["📄 Data & Teks", "📊 Distribusi", "☁️ WordCloud", "🧠 Model & Evaluasi", "🔮 Prediksi"])

# ----------------------------------------------------------------------------
# Tab 1: data + pemrosesan (Langkah 3 & 4)
# ----------------------------------------------------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Tweet dibaca", f"{len(mentah):,}".replace(",", "."))
    c2.metric("Tweet dipakai", f"{len(df):,}".replace(",", "."))
    c3.metric("Duplikat dibuang", f"{len(mentah) - len(df):,}".replace(",", "."))
    st.caption(f"Sumber data: {sumber}")
    st.subheader("Hasil cleansing, normalisasi, dan pelabelan")
    tampil_tabel(df[["full_text", "clean_text", "normal_text", "sentimen"]].head(50))
    st.download_button("⬇️ Unduh data hasil proses (CSV)",
                       df.to_csv(sep=";", index=False).encode("utf-8-sig"),
                       file_name="Tweet_Data_normalized.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Tab 2: distribusi (Langkah 5)
# ----------------------------------------------------------------------------
with tab_dist:
    dist = df["sentimen"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
    kolom = st.columns(3)
    for k, lab in zip(kolom, LABEL_ORDER):
        k.metric(f"{EMOJI[lab]} {lab.capitalize()}", int(dist[lab]),
                 f"{dist[lab] / dist.sum():.2%}", delta_color="off")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bars = axes[0].bar(dist.index, dist.values, color=[WARNA[l] for l in dist.index])
    axes[0].bar_label(bars)
    axes[0].set_title("Jumlah Tweet per Sentimen")
    axes[1].pie(dist.values, labels=dist.index, colors=[WARNA[l] for l in dist.index],
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Proporsi Sentimen")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------------
# Tab 3: WordCloud (Langkah 6)
# ----------------------------------------------------------------------------
with tab_wc:
    kolom = st.columns(3)
    for k, lab in zip(kolom, LABEL_ORDER):
        with k:
            st.subheader(f"{EMOJI[lab]} {lab.capitalize()}")
            teks = " ".join(df.loc[df["sentimen"] == lab, "normal_text"])
            if teks.strip():
                wc = WordCloud(width=600, height=400, background_color="white", colormap=CMAP[lab],
                               stopwords=STOPWORD, max_words=100).generate(teks)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Tidak ada data.")

# ----------------------------------------------------------------------------
# Tab 4: evaluasi (Langkah 7 & 8)
# ----------------------------------------------------------------------------
with tab_eval:
    y_test, y_pred = hasil_model["y_test"], hasil_model["y_pred"]
    labels_ada = [l for l in LABEL_ORDER if l in set(df["sentimen"])]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data latih (80%)", hasil_model["n_train"])
    c2.metric("Data uji (20%)", hasil_model["n_test"])
    c3.metric("Fitur TF-IDF", hasil_model["n_fitur"])
    c4.metric("Neuron hidden", hidden)

    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    ringkas = pd.DataFrame({
        "Metrik": ["Akurasi", "Presisi (weighted)", "Recall (weighted)", "F1-Score (weighted)"],
        "Nilai": [accuracy_score(y_test, y_pred), p, r, f]}).round(4)

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Tabel metrik")
        tampil_tabel(ringkas, hide_index=True)
        st.subheader("Laporan per kelas")
        tampil_tabel(pd.DataFrame(classification_report(
            y_test, y_pred, labels=labels_ada, output_dict=True, zero_division=0)).T.round(3))
    with kanan:
        st.subheader("Grafik metrik")
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        bars = ax.bar(ringkas["Metrik"], ringkas["Nilai"], color="#4a90d9")
        ax.bar_label(bars, fmt="%.3f")
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=labels_ada)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_ada)), labels_ada)
    ax.set_yticks(range(len(labels_ada)), labels_ada)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    for i in range(len(labels_ada)):
        for j in range(len(labels_ada)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "**Catatan:** label dibuat otomatis dari kamus kata positif/negatif, lalu kata-kata yang sama "
        "menjadi fitur TF-IDF. Karena itu akurasi yang sangat tinggi (bahkan 100%) belum membuktikan "
        "model memahami sentimen sebenarnya. Anggap ini evaluasi awal; untuk evaluasi yang kuat, beri "
        "label manual pada sebagian tweet.")

# ----------------------------------------------------------------------------
# Tab 5: prediksi teks baru (Langkah 9)
# ----------------------------------------------------------------------------
with tab_pred:
    st.subheader("Prediksi sentimen teks baru")
    teks_baru = st.text_area(
        "Masukkan teks", height=110, key="teks_baru",
        value="Listrik sudah menyala kembali, warga lega dan terima kasih kepada petugas PLN")
    if st.button("Prediksi Sentimen", type="primary", key="tombol_prediksi"):
        if not teks_baru.strip():
            st.warning("Teks masih kosong. Ketik kalimat terlebih dahulu.")
        else:
            dibersihkan, bersih, label, proba = prediksi(teks_baru, hasil_model)
            st.markdown(f"### {EMOJI[label]} {label.upper()}")
            st.write(f"**Setelah cleansing:** {dibersihkan}")
            st.write(f"**Setelah normalisasi:** {bersih}")
            for k in LABEL_ORDER:
                if k in proba:
                    st.write(f"{EMOJI[k]} {k.capitalize()} — {proba[k]:.2%}")
                    st.progress(float(proba[k]))
