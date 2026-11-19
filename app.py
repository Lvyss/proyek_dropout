import streamlit as st
import pandas as pd
import io
from eda import render_eda

from utils import guess_column_types, check_target
from preprocessing import render_preprocessing_tab        # BARU
from evaluation import render_evaluation_tab              # BARU

st.set_page_config(page_title="Klasifikasi Dropout Mahasiswa", layout="wide")


@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


st.title("Klasifikasi Dropout dan Keberhasilan Akademik Mahasiswa")

uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])

menu = st.radio(
    "Menu",
    ["Overview", "EDA", "Preprocessing", "Evaluasi"],
    horizontal=True,
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("Upload file CSV lewat sidebar untuk memulai.")
    st.stop()

try:
    df = load_csv(uploaded.getvalue())
except Exception as e:
    st.error(f"File tidak bisa dibaca: {e}")
    st.stop()

if df.empty or df.shape[1] < 2:
    st.error("File kosong atau kolomnya kurang dari 2. Periksa format CSV-nya.")
    st.stop()

# ---------------- Pengaturan data (sidebar) ----------------
st.sidebar.header("Pengaturan data")
file_id = f"{uploaded.name}_{uploaded.size}"

semua_kolom = list(df.columns)
default_target = "Target" if "Target" in semua_kolom else semua_kolom[-1]
target = st.sidebar.selectbox(
    "Kolom target",
    semua_kolom,
    index=semua_kolom.index(default_target),
    key=f"target_{file_id}",
)

# Baris tanpa label tidak bisa dipakai untuk klasifikasi
n_awal = len(df)
df = df[df[target].notna()].copy()
n_dibuang = n_awal - len(df)

errors, warnings = check_target(df[target])
if errors:
    for e in errors:
        st.error(e)
    st.stop()

fitur = [c for c in df.columns if c != target]
tebakan = guess_column_types(df[fitur])

kandidat_nominal = [c for c in fitur if tebakan[c] == "numerik"]
nominal_pilihan = st.sidebar.multiselect(
    "Kolom nominal berkode angka",
    kandidat_nominal,
    key=f"nominal_{file_id}_{target}",
    help="Kolom berisi angka yang sebenarnya kode kategori (mis. kode jurusan).",
)

col_types = {}
for c in fitur:
    col_types[c] = "nominal" if c in nominal_pilihan else tebakan[c]

# Simpan untuk dipakai tab lain
st.session_state["df"] = df
st.session_state["target"] = target
st.session_state["col_types"] = col_types

# ---------------- Isi tiap menu ----------------
if menu == "Overview":
    st.subheader("Overview Data")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah baris", f"{len(df):,}")
    c2.metric("Jumlah kolom", df.shape[1])
    c3.metric("Jumlah fitur", len(fitur))
    c4.metric("Ukuran file", f"{uploaded.size / 1024:.1f} KB")

    if n_dibuang > 0:
        st.warning(f"{n_dibuang} baris dibuang karena kolom target kosong.")
    for w in warnings:
        st.warning(w)

    st.write(f"Distribusi kelas pada kolom target: **{target}**")
    counts = df[target].value_counts()
    kiri, kanan = st.columns(2)
    tabel = pd.DataFrame({
        "kelas": counts.index,
        "jumlah": counts.values,
        "persen": (counts.values / counts.sum() * 100).round(1),
    })
    kiri.dataframe(tabel, width="stretch", hide_index=True)
    kanan.bar_chart(counts)

    st.write("Tipe kolom fitur")
    ringkas = pd.Series(col_types).value_counts()
    t1, t2, t3 = st.columns(3)
    t1.metric("Numerik", int(ringkas.get("numerik", 0)))
    t2.metric("Biner", int(ringkas.get("biner", 0)))
    t3.metric("Nominal", int(ringkas.get("nominal", 0)))

    tabel_tipe = pd.DataFrame({
        "kolom": fitur,
        "dtype": [str(df[c].dtype) for c in fitur],
        "nilai unik": [df[c].nunique() for c in fitur],
        "tipe dipakai": [col_types[c] for c in fitur],
    })
    st.dataframe(tabel_tipe, width="stretch", hide_index=True)

    with st.expander("Bantu putuskan kolom nominal (kolom bilangan bulat)"):
        bantu = [c for c in kandidat_nominal if pd.api.types.is_integer_dtype(df[c])]
        st.dataframe(
            pd.DataFrame({
                "kolom": bantu,
                "nilai unik": [df[c].nunique() for c in bantu],
                "min": [int(df[c].min()) for c in bantu],
                "maks": [int(df[c].max()) for c in bantu],
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Petunjuk: kode kategori biasanya punya nilai unik sedikit tetapi "
            "rentang angka lebar (mis. kode jurusan 33 sampai 9991)."
        )

    st.write("Data mentah")
    st.dataframe(df, width="stretch")

elif menu == "EDA":
    render_eda(df, target, col_types)

elif menu == "Preprocessing":
    render_preprocessing_tab(df, target, col_types)

elif menu == "Evaluasi":
    render_evaluation_tab(
        df, target,
        st.session_state.get("config"),
        col_types,
    )