# preprocessing.py
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


# ---------- Custom transformers ----------

class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorizing berbasis IQR (fit menyimpan batas, tidak mengubah y)."""
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=30):
        self.k = k

    def fit(self, X, y):
        n = X.shape[1]
        k = min(self.k, n) if isinstance(self.k, int) else self.k
        self.selector_ = SelectKBest(score_func=f_classif, k=k).fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self, indices=False):
        return self.selector_.get_support(indices=indices)

    def get_feature_names_out(self, input_features=None):
        return self.selector_.get_feature_names_out(input_features)


class SafePCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, X, y=None):
        n = X.shape[1]
        nc = min(self.n_components, n) if isinstance(self.n_components, int) else self.n_components
        self.pca_ = PCA(n_components=nc, random_state=42).fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


# ---------- Build transformer ----------
# col_types: dict FLAT {nama_kolom: "numerik"|"biner"|"nominal"}

def build_preprocessor(config, col_types):
    dropped = set(config.get("buang_kolom", []))

    numeric_cols = [c for c, t in col_types.items()
                    if t == "numerik" and c not in dropped]
    binary_cols = [c for c, t in col_types.items()
                   if t == "biner" and c not in dropped]
    nominal_cols = [c for c, t in col_types.items()
                    if t == "nominal" and c not in dropped]

    missing = config.get("missing", "median")
    outlier = config.get("outlier", "none")
    scaling = config.get("scaling", "standard")

    def make_steps(pakai_outlier):
        steps = []
        if missing in ("mean", "median", "modus"):
            strat = {"mean": "mean", "median": "median",
                     "modus": "most_frequent"}[missing]
            steps.append(("imputer", SimpleImputer(strategy=strat)))
        if pakai_outlier and outlier == "winsorize_iqr":
            steps.append(("winsorizer", IQRClipper()))
        if scaling == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaling == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        return steps

    num_steps = make_steps(pakai_outlier=True)    # numerik: boleh winsorize
    bin_steps = make_steps(pakai_outlier=False)   # biner: tanpa winsorize

    nom_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(num_steps) if num_steps else "passthrough", numeric_cols))
    if binary_cols:
        transformers.append(("bin", Pipeline(bin_steps) if bin_steps else "passthrough", binary_cols))
    if nominal_cols:
        transformers.append(("nom", Pipeline(nom_steps), nominal_cols))

    col_transformer = ColumnTransformer(transformers, remainder="drop")

    steps = [("columns", col_transformer)]
    seleksi = config.get("seleksi_fitur", "none")
    if seleksi == "selectkbest":
        steps.append(("selector", SafeSelectKBest(k=config.get("k_features") or 30)))
    elif seleksi == "pca":
        steps.append(("pca", SafePCA(n_components=config.get("pca_components") or 10)))

    return Pipeline(steps)


def build_pipeline(config, col_types, estimator):
    pre = build_preprocessor(config, col_types)

    if config.get("imbalance") == "smote":
        if not HAS_IMBLEARN:
            st.warning("imbalanced-learn tidak terpasang; SMOTE dilewati.")
            return Pipeline([("preprocessor", pre), ("classifier", estimator)])
        return ImbPipeline([
            ("preprocessor", pre),
            ("smote", SMOTE(random_state=config.get("random_state", 42))),
            ("classifier", estimator),
        ])

    return Pipeline([
        ("preprocessor", pre),
        ("classifier", estimator),
    ])


# ---------- Preview (fit di train saja) ----------

def preview_preprocessing(df, target_col, config, col_types):
    df = df.copy()
    if config.get("hapus_duplikat", True):
        df = df.drop_duplicates().reset_index(drop=True)
    if config.get("missing") == "drop":
        df = df.dropna().reset_index(drop=True)

    drop_cols = [c for c in config.get("buang_kolom", []) if c in df.columns]
    df_model = df.drop(columns=drop_cols)

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42), stratify=y_enc,
    )

    pre = build_preprocessor(config, col_types)
    pre.fit(X_tr, y_tr)
    X_tr_t = pre.transform(X_tr)

    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_tr_t.shape[1])]

    out = pd.DataFrame(np.asarray(X_tr_t), columns=feat_names)
    out["__target__"] = le.inverse_transform(y_tr)

    info = {
        "n_before": X_tr.shape[1],
        "n_after": X_tr_t.shape[1],
        "n_rows_train": len(X_tr),
        "n_rows_test": len(X_te),
        "classes": list(le.classes_),
    }
    return out, info


# ---------- UI tab Preprocessing ----------

def render_preprocessing_tab(df, target_col, col_types):
    st.subheader("⚙️ Preprocessing")
    st.caption(
        "Tab ini hanya **menyimpan konfigurasi**. Transformasi dijalankan di dalam pipeline "
        "saat tab Evaluasi, dan selalu di-fit pada data latih saja (mencegah data leakage)."
    )

    st.info(
        "Tipe kolom diambil dari sidebar **Pengaturan data**. "
        "Untuk mengubah klasifikasi kolom (mis. menandai kode jurusan sebagai nominal), "
        "atur lewat sidebar."
    )

    all_cols = [c for c in df.columns if c != target_col]

    with st.expander("Konfigurasi transformasi", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            hapus_duplikat = st.checkbox("Hapus baris duplikat", value=True)
            missing = st.selectbox("Missing value", ["median", "mean", "modus", "drop"])
            outlier = st.selectbox("Outlier", ["none", "winsorize_iqr"])
            scaling = st.selectbox("Scaling", ["standard", "minmax", "none"])
        with c2:
            seleksi = st.selectbox("Seleksi fitur", ["none", "selectkbest", "pca"])
            imbalance = st.selectbox("Imbalance", ["none", "class_weight", "smote"])
            test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
            random_state = st.number_input("Random state", 0, 9999, 42)

        k_features = None
        pca_components = None
        if seleksi == "selectkbest":
            k_features = st.slider("Jumlah fitur (SelectKBest)", 5, 150, 30)
        if seleksi == "pca":
            pca_components = st.slider("PCA components", 2, 50, 10)

        if imbalance == "smote" and not HAS_IMBLEARN:
            st.warning("`imbalanced-learn` tidak terpasang → SMOTE tidak akan aktif.")

    with st.expander("Buang kolom (mis. uji deteksi dini tanpa fitur semester)", expanded=False):
        search = st.text_input("Cari nama kolom", placeholder="mis. Curricular")
        cand = [c for c in all_cols if not search or search.lower() in c.lower()]
        buang_kolom = st.multiselect("Kolom yang dibuang", cand, default=[])

    config = {
        "hapus_duplikat": hapus_duplikat,
        "buang_kolom": buang_kolom,
        "missing": missing,
        "outlier": outlier,
        "scaling": scaling,
        "seleksi_fitur": seleksi,
        "k_features": k_features,
        "pca_components": pca_components,
        "imbalance": imbalance,
        "test_size": test_size,
        "random_state": int(random_state),
    }
    st.session_state["config"] = config

    st.divider()
    if st.button("👁️ Preview preprocessing (fit di train)", type="primary"):
        with st.spinner("Memproses preview..."):
            preview_df, info = preview_preprocessing(df, target_col, config, col_types)
        st.success(
            f"Fitur: **{info['n_before']} → {info['n_after']}** "
            f"| baris train/test: **{info['n_rows_train']} / {info['n_rows_test']}** "
            f"| kelas: {info['classes']}"
        )
        st.dataframe(preview_df.head(20), width="stretch")
        if config["imbalance"] == "smote":
            st.caption("Preview menampilkan data **sebelum** SMOTE "
                       "(SMOTE hanya diterapkan pada data latih saat pelatihan di tab Evaluasi).")