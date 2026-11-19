# evaluation.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc as sk_auc,
)

from preprocessing import build_pipeline
from models import MODELS, PARAM_SPECS, GRIDS, build_estimator


# ---------- Sidebar parameter per model ----------

def render_model_param_sidebar(selected_models):
    params = {}
    for key in selected_models:
        specs = PARAM_SPECS.get(key, [])
        if not specs:
            params[key] = {}
            continue
        with st.sidebar.expander(f"⚙️ {MODELS[key]['label']}", expanded=False):
            p = {}
            for s in specs:
                t = s["type"]
                if t == "int":
                    p[s["name"]] = st.slider(
                        s["label"], int(s["min"]), int(s["max"]),
                        int(s["default"]), int(s.get("step", 1)),
                        key=f"p_{key}_{s['name']}",
                    )
                elif t == "float_log":
                    p[s["name"]] = st.select_slider(
                        s["label"], options=s["options"], value=s["default"],
                        key=f"p_{key}_{s['name']}",
                    )
                elif t == "select":
                    opts = s["options"]
                    idx = opts.index(s["default"]) if s["default"] in opts else 0
                    p[s["name"]] = st.selectbox(
                        s["label"], opts, index=idx,
                        key=f"p_{key}_{s['name']}",
                    )
            params[key] = p
    return params


# ---------- Helper: siapkan X, y ----------

def _prepare_xy(df, target_col, config):
    df = df.copy()
    if config.get("hapus_duplikat", True):
        df = df.drop_duplicates().reset_index(drop=True)

    drop_cols = [c for c in config.get("buang_kolom", []) if c in df.columns]
    df = df.drop(columns=drop_cols)

    if config.get("missing") == "drop":
        df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le


def _make_estimator(key, model_params, config):
    cw = "balanced" if config.get("imbalance") == "class_weight" else None
    return build_estimator(key, model_params.get(key, {}), config["random_state"], class_weight=cw)


# ---------- Mode 1: Holdout ----------

def run_holdout(model_keys, df, target_col, config, col_types, model_params, do_tuning=False):
    X, y_enc, le = _prepare_xy(df, target_col, config)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=config["test_size"],
        random_state=config["random_state"], stratify=y_enc,
    )

    results = []
    progress = st.progress(0.0, text="Melatih model…")
    n = len(model_keys)

    for i, key in enumerate(model_keys):
        progress.progress(i / n, text=f"Melatih {MODELS[key]['label']} ({i+1}/{n})…")

        est = _make_estimator(key, model_params, config)
        pipe = build_pipeline(config, col_types, est)

        best_params = {}
        t0 = time.perf_counter()
        if do_tuning and GRIDS.get(key):
            tune_cv = StratifiedKFold(n_splits=3, shuffle=True,
                                      random_state=config["random_state"])
            search = GridSearchCV(pipe, GRIDS[key], scoring="f1_macro",
                                  cv=tune_cv, n_jobs=-1, refit=True)
            search.fit(X_tr, y_tr)
            pipe = search.best_estimator_
            best_params = search.best_params_
        else:
            pipe.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0   # termasuk waktu GridSearchCV bila tuning aktif

        y_pred = pipe.predict(X_te)
        y_tr_pred = pipe.predict(X_tr)

        y_proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_te)
            except Exception:
                y_proba = None

        acc         = accuracy_score(y_te, y_pred)
        acc_tr      = accuracy_score(y_tr, y_tr_pred)
        p_macro     = precision_score(y_te, y_pred, average="macro", zero_division=0)
        r_macro     = recall_score(y_te, y_pred, average="macro", zero_division=0)
        f1_macro    = f1_score(y_te, y_pred, average="macro", zero_division=0)
        f1_weight   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        f1_macro_tr = f1_score(y_tr, y_tr_pred, average="macro", zero_division=0)

        per_class_recall = recall_score(y_te, y_pred, average=None,
                                        zero_division=0).tolist()

        roc_auc = None
        if y_proba is not None and len(le.classes_) > 1:
            try:
                roc_auc = roc_auc_score(y_te, y_proba,
                                        multi_class="ovr", average="macro")
            except Exception:
                roc_auc = None

        cm = confusion_matrix(y_te, y_pred, labels=range(len(le.classes_)))

        results.append({
            "key": key,
            "model": MODELS[key]["label"],
            "accuracy": acc,
            "accuracy_train": acc_tr,
            "f1_macro": f1_macro,
            "f1_macro_train": f1_macro_tr,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_weighted": f1_weight,
            "roc_auc": roc_auc,
            "per_class_recall": per_class_recall,
            "cm": cm,
            "classes": list(le.classes_),
            "y_te": y_te,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "overfit_gap": f1_macro_tr - f1_macro,
            "best_params": best_params,
            "fit_time": fit_time,
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "mode": "holdout",
        })

    progress.progress(1.0, text="Selesai.")
    return results


# ---------- Mode 2: Stratified k-fold CV ----------

def run_cv(model_keys, df, target_col, config, col_types, model_params, k_folds=10):
    X, y_enc, le = _prepare_xy(df, target_col, config)

    cv = StratifiedKFold(n_splits=k_folds, shuffle=True,
                         random_state=config["random_state"])
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    results = []
    progress = st.progress(0.0, text="Cross-validating…")
    n = len(model_keys)

    for i, key in enumerate(model_keys):
        progress.progress(i / n, text=f"CV {MODELS[key]['label']} ({i+1}/{n})…")

        est = _make_estimator(key, model_params, config)
        pipe = build_pipeline(config, col_types, est)

        t0 = time.perf_counter()
        cv_res = cross_validate(
            pipe, X, y_enc, cv=cv, scoring=scoring,
            return_train_score=True, n_jobs=-1, error_score="raise",
        )
        elapsed = time.perf_counter() - t0

        results.append({
            "key": key,
            "model": MODELS[key]["label"],
            "accuracy_mean": cv_res["test_accuracy"].mean(),
            "accuracy_std":  cv_res["test_accuracy"].std(),
            "f1_macro_mean": cv_res["test_f1_macro"].mean(),
            "f1_macro_std":  cv_res["test_f1_macro"].std(),
            "f1_weighted_mean": cv_res["test_f1_weighted"].mean(),
            "precision_macro_mean": cv_res["test_precision_macro"].mean(),
            "recall_macro_mean": cv_res["test_recall_macro"].mean(),
            "f1_macro_train_mean": cv_res["train_f1_macro"].mean(),
            "overfit_gap": cv_res["train_f1_macro"].mean() - cv_res["test_f1_macro"].mean(),
            "cv_time": elapsed,
            "n_folds": k_folds,
            "mode": "cv",
        })

    progress.progress(1.0, text="Selesai.")
    return results


# ---------- Display ----------

def display_results(results):
    if not results:
        st.info("Belum ada hasil. Klik **Jalankan evaluasi**.")
        return

    mode = results[0]["mode"]
    st.subheader("📊 Hasil Evaluasi")

    if mode == "holdout":
        ordered = sorted(results, key=lambda r: -r["f1_macro"])
        table = pd.DataFrame([{
            "Model": r["model"],
            "Accuracy": round(r["accuracy"], 4),
            "Macro F1": round(r["f1_macro"], 4),
            "Macro P":  round(r["precision_macro"], 4),
            "Macro R":  round(r["recall_macro"], 4),
            "Weighted F1": round(r["f1_weighted"], 4),
            "ROC-AUC": round(r["roc_auc"], 4) if r["roc_auc"] is not None else None,
            "Train F1": round(r["f1_macro_train"], 4),
            "Overfit":  round(r["overfit_gap"], 4),
            "Waktu latih (s)": round(r["fit_time"], 2),
        } for r in ordered])
        st.dataframe(table, width="stretch", hide_index=True)

        st.markdown("**Perbandingan macro-F1 (test vs train)**")
        bar_df = pd.DataFrame([{
            "Model": r["model"], "Test (macro F1)": r["f1_macro"],
            "Train (macro F1)": r["f1_macro_train"],
        } for r in results]).set_index("Model")
        st.bar_chart(bar_df)

        st.markdown("**Recall per kelas** (perhatikan kelas minoritas *Enrolled*)")
        recall_rows = []
        for r in results:
            row = {"Model": r["model"]}
            for cls, rec in zip(r["classes"], r["per_class_recall"]):
                row[str(cls)] = round(rec, 3)
            recall_rows.append(row)
        st.dataframe(pd.DataFrame(recall_rows), width="stretch", hide_index=True)

        st.divider()
        st.markdown("**Detail per model**")
        choice = st.selectbox("Pilih model", [r["model"] for r in results], key="detail_model")
        r = next(x for x in results if x["model"] == choice)

        c1, c2 = st.columns([1, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(4.5, 3.6))
            sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=r["classes"], yticklabels=r["classes"], ax=ax,
                        cbar=False)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {r['model']}")
            st.pyplot(fig)
        with c2:
            if r["y_proba"] is not None:
                fig, ax = plt.subplots(figsize=(4.5, 3.6))
                for i, cls in enumerate(r["classes"]):
                    y_bin = (np.asarray(r["y_te"]) == i).astype(int)
                    fpr, tpr, _ = roc_curve(y_bin, r["y_proba"][:, i])
                    ax.plot(fpr, tpr, label=f"{cls} (AUC={sk_auc(fpr, tpr):.3f})")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
                ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                ax.set_title("ROC (One-vs-Rest)")
                ax.legend(fontsize=8)
                st.pyplot(fig)
            else:
                st.info("Model ini tidak menyediakan `predict_proba`.")

        if r.get("best_params"):
            st.caption(f"Best params: `{r['best_params']}`")
        st.caption(f"Train n={r['n_train']}, Test n={r['n_test']}")

    else:  # CV
        ordered = sorted(results, key=lambda r: -r["f1_macro_mean"])
        table = pd.DataFrame([{
            "Model": r["model"],
            "Accuracy (mean ± std)":   f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
            "Macro F1 (mean ± std)":   f"{r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f}",
            "Weighted F1": round(r["f1_weighted_mean"], 4),
            "Macro P": round(r["precision_macro_mean"], 4),
            "Macro R": round(r["recall_macro_mean"], 4),
            "Train F1 (mean)": round(r["f1_macro_train_mean"], 4),
            "Overfit (mean)": round(r["overfit_gap"], 4),
            "Time (s)": round(r["cv_time"], 2),
        } for r in ordered])
        st.dataframe(table, width="stretch", hide_index=True)

        st.markdown("**Macro F1 per model (CV)**")
        bar_df = pd.DataFrame([{
            "Model": r["model"], "Macro F1": r["f1_macro_mean"],
        } for r in results]).set_index("Model")
        st.bar_chart(bar_df)


# ---------- UI utama tab Evaluasi ----------

def render_evaluation_tab(df, target_col, config, col_types):
    st.subheader("🧪 Evaluasi")

    if not config:
        st.warning("Atur konfigurasi di tab **Preprocessing** dulu.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        selected = st.multiselect(
            "Pilih algoritma",
            options=list(MODELS.keys()),
            default=list(MODELS.keys()),
            format_func=lambda k: MODELS[k]["label"],
        )
    with c2:
        mode = st.radio("Mode validasi", ["Holdout", "Stratified k-fold CV"])

    k_folds = 10
    do_tuning = False
    if mode == "Holdout":
        do_tuning = st.checkbox(
            "Aktifkan GridSearchCV (kecil, 3-fold, di data latih)",
            value=False,
            help="Set uji tidak digunakan untuk tuning.",
        )
    else:
        k_folds = st.slider("Jumlah fold", 3, 10, 10)

    st.sidebar.markdown("### ⚙️ Parameter model")
    model_params = render_model_param_sidebar(selected)

    run = st.button("🚀 Jalankan evaluasi", type="primary",
                    disabled=len(selected) == 0)

    if run:
        try:
            with st.spinner("Menjalankan eksperimen…"):
                if mode == "Holdout":
                    results = run_holdout(selected, df, target_col, config,
                                          col_types, model_params, do_tuning)
                else:
                    results = run_cv(selected, df, target_col, config,
                                     col_types, model_params, k_folds)
            st.session_state["results"] = results
        except Exception as e:
            st.error(f"Evaluasi gagal: {e}")
            st.exception(e)

    display_results(st.session_state.get("results", []))


# ---------- Uji kewajaran (debug) ----------

def sanity_check(results):
    """Panggil dari console/notebook setelah run_holdout untuk memeriksa angka."""
    for r in results:
        if r["key"] == "zero_r":
            print(f"Zero-R: acc={r['accuracy']:.4f}  f1_macro={r['f1_macro']:.4f}")
            print("  Ekspektasi: accuracy ≈ 0.499 (proporsi Graduate), macro-F1 ≈ 0.22")
        if r["accuracy"] > 0.99:
            print(f"⚠️  {r['model']}: accuracy {r['accuracy']:.4f} mencurigakan — cek leakage.")
        if r["overfit_gap"] > 0.2:
            print(f"⚠️  {r['model']}: overfit gap {r['overfit_gap']:+.3f}.")