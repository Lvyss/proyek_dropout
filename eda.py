import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats


def tabel_outlier_iqr(df: pd.DataFrame, kolom: list) -> pd.DataFrame:
    """Hitung jumlah outlier per kolom dengan aturan IQR (Q1-1.5*IQR, Q3+1.5*IQR)."""
    baris = []
    for c in kolom:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < bawah) | (s > atas)).sum())
        baris.append({
            "kolom": c,
            "batas bawah": round(float(bawah), 2),
            "batas atas": round(float(atas), 2),
            "jumlah outlier": n_out,
            "persen": round(n_out / len(s) * 100, 2),
        })
    hasil = pd.DataFrame(baris)
    if hasil.empty:
        return hasil
    return hasil.sort_values("jumlah outlier", ascending=False).reset_index(drop=True)


def pasangan_korelasi_tertinggi(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Ambil n pasangan fitur dengan |korelasi| terbesar (segitiga atas saja)."""
    iu = np.triu_indices_from(corr, k=1)
    pasangan = pd.DataFrame({
        "fitur A": corr.index[iu[0]],
        "fitur B": corr.columns[iu[1]],
        "korelasi": corr.values[iu],
    }).dropna()
    urut = pasangan["korelasi"].abs().sort_values(ascending=False).index
    return pasangan.loc[urut].head(n).round(3).reset_index(drop=True)


def render_eda(df: pd.DataFrame, target: str, col_types: dict):
    st.subheader("Exploratory Data Analysis")

    fitur = list(col_types.keys())
    numerik = [c for c in fitur if col_types[c] == "numerik"]
    biner = [c for c in fitur if col_types[c] == "biner"]
    numerik_biner = numerik + biner

    # Target dijadikan teks agar plotly memperlakukannya sebagai kategori
    df_plot = df.copy()
    df_plot[target] = df_plot[target].astype(str)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Statistik & kualitas data",
        "Distribusi fitur",
        "Hubungan antar fitur",
        "Normalitas",
    ])

    # ---------- Tab 1: statistik deskriptif + kualitas data ----------
    with tab1:
        st.write("Statistik deskriptif (fitur numerik dan biner)")
        if numerik_biner:
            desc = df[numerik_biner].describe().T
            desc["skewness"] = df[numerik_biner].skew()
            st.dataframe(desc.round(2), width="stretch")
        else:
            st.info("Tidak ada fitur numerik/biner.")

        st.write("Kualitas data")
        n_missing = int(df[fitur].isna().sum().sum())
        n_dup = int(df.duplicated().sum())
        n_konstan = int(sum(df[c].nunique(dropna=True) <= 1 for c in fitur))
        k1, k2, k3 = st.columns(3)
        k1.metric("Total missing value", n_missing)
        k2.metric("Baris duplikat", n_dup)
        k3.metric("Kolom konstan", n_konstan)

        if n_missing > 0:
            miss = df[fitur].isna().sum()
            miss = miss[miss > 0].sort_values(ascending=False)
            st.dataframe(
                pd.DataFrame({"kolom": miss.index, "jumlah missing": miss.values,
                              "persen": (miss.values / len(df) * 100).round(2)}),
                width="stretch", hide_index=True,
            )
        else:
            st.success("Tidak ada missing value pada fitur.")

        if n_dup > 0:
            st.warning(
                f"Ada {n_dup} baris duplikat. Pertimbangkan menghapusnya di tab Preprocessing, "
                "kecuali duplikat memang data yang sah (mis. dua mahasiswa dengan profil sama)."
            )

        with st.expander("Deteksi outlier (aturan IQR, fitur numerik)"):
            out = tabel_outlier_iqr(df, numerik)
            if out.empty:
                st.info("Tidak ada fitur numerik.")
            else:
                st.dataframe(out, width="stretch", hide_index=True)
                st.caption(
                    "Outlier bukan otomatis salah. Kolom hitungan yang sebagian besar bernilai 0 "
                    "akan menandai banyak nilai sebagai outlier karena IQR-nya nol."
                )

    # ---------- Tab 2: distribusi fitur ----------
    with tab2:
        kolom = st.selectbox("Pilih fitur", fitur, key="eda_dist_col")
        if col_types[kolom] == "numerik":
            bins = st.slider("Jumlah bin histogram", 10, 100, 30, key="eda_bins")
            a, b = st.columns(2)
            fig_h = px.histogram(
                df_plot, x=kolom, color=target, nbins=bins,
                barmode="overlay", opacity=0.6, title=f"Histogram {kolom}",
            )
            fig_b = px.box(
                df_plot, x=target, y=kolom, color=target,
                title=f"Boxplot {kolom} per kelas",
            )
            a.plotly_chart(fig_h, width="stretch")
            b.plotly_chart(fig_b, width="stretch")
        else:
            teratas = df[kolom].value_counts().head(20).index
            sub = df_plot[df_plot[kolom].isin(teratas)]
            jumlah = df[kolom].value_counts().head(20)
            prop = pd.crosstab(sub[kolom], sub[target], normalize="index").mul(100).round(1)
            prop = prop.loc[[v for v in teratas if v in prop.index]]
            prop.index = prop.index.astype(str)

            a, b = st.columns(2)
            fig_j = px.bar(
                x=jumlah.index.astype(str), y=jumlah.values,
                labels={"x": kolom, "y": "jumlah"}, title=f"Jumlah per nilai {kolom}",
            )
            fig_j.update_xaxes(type="category")
            fig_p = px.bar(
                prop, barmode="stack",
                labels={"index": kolom, "value": "persen", "variable": target},
                title=f"Proporsi kelas {target} per nilai {kolom}",
            )
            fig_p.update_xaxes(type="category")
            a.plotly_chart(fig_j, width="stretch")
            b.plotly_chart(fig_p, width="stretch")
            if df[kolom].nunique() > 20:
                st.caption("Hanya 20 nilai paling sering yang ditampilkan.")

    # ---------- Tab 3: korelasi + scatter ----------
    with tab3:
        if len(numerik_biner) >= 2:
            metode = st.radio("Metode korelasi", ["pearson", "spearman"],
                              horizontal=True, key="eda_corr")
            corr = df[numerik_biner].corr(method=metode)
            fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                            aspect="auto", height=700, title=f"Heatmap korelasi ({metode})")
            st.plotly_chart(fig, width="stretch")

            st.write("10 pasangan fitur dengan korelasi tertinggi (kandidat fitur redundan)")
            st.dataframe(pasangan_korelasi_tertinggi(corr), width="stretch", hide_index=True)
        else:
            st.info("Butuh minimal 2 fitur numerik/biner untuk korelasi.")

        st.write("Scatter plot dua fitur numerik")
        if len(numerik) >= 2:
            c1, c2 = st.columns(2)
            x = c1.selectbox("Sumbu X", numerik, index=0, key="eda_sx")
            y = c2.selectbox("Sumbu Y", numerik, index=1, key="eda_sy")
            sampel = df_plot.sample(min(2000, len(df_plot)), random_state=42)
            fig = px.scatter(sampel, x=x, y=y, color=target, opacity=0.6,
                             title=f"{y} vs {x}")
            st.plotly_chart(fig, width="stretch")
            if len(df_plot) > 2000:
                st.caption("Ditampilkan sampel acak 2.000 baris agar tetap ringan.")
        else:
            st.info("Butuh minimal 2 fitur numerik untuk scatter plot.")

    # ---------- Tab 4: normalitas ----------
    with tab4:
        if numerik:
            kolom = st.selectbox("Pilih fitur numerik", numerik, key="eda_norm_col")
            s = df[kolom].dropna()
            if s.nunique() < 3:
                st.info("Fitur ini hampir konstan, uji normalitas tidak bermakna.")
            else:
                (osm, osr), (slope, intercept, _) = stats.probplot(s, dist="norm")
                fig = px.scatter(x=osm, y=osr, title=f"Q-Q plot {kolom}",
                                 labels={"x": "Kuantil teoretis (normal)", "y": "Kuantil data"})
                garis = np.array([osm.min(), osm.max()])
                fig.add_scatter(x=garis, y=slope * garis + intercept, mode="lines", name="garis normal")
                st.plotly_chart(fig, width="stretch")

                sampel = s.sample(min(5000, len(s)), random_state=42)
                w, p = stats.shapiro(sampel)
                m1, m2, m3 = st.columns(3)
                m1.metric("Skewness", f"{s.skew():.2f}")
                m2.metric("Shapiro-Wilk W", f"{w:.3f}")
                m3.metric("p-value", f"{p:.2e}")
                st.caption(
                    "Pada data besar, uji Shapiro-Wilk hampir selalu menolak normalitas karena sensitif "
                    "terhadap penyimpangan kecil. Andalkan Q-Q plot dan skewness sebagai pendamping."
                )
        else:
            st.info("Tidak ada fitur numerik.")

            