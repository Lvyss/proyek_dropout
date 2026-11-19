import pandas as pd

MIN_PER_CLASS = 10   # cukup untuk split terstratifikasi + 10-fold CV
MAX_CLASSES = 20     # lebih dari ini kemungkinan bukan kolom target klasifikasi


def guess_column_types(df_fitur: pd.DataFrame) -> dict:
    """Tebakan awal tipe kolom: 'nominal', 'biner', atau 'numerik'."""
    tipe = {}
    for col in df_fitur.columns:
        s = df_fitur[col]
        if not pd.api.types.is_numeric_dtype(s):
            tipe[col] = "nominal"
        elif s.nunique(dropna=True) == 2:
            tipe[col] = "biner"
        else:
            tipe[col] = "numerik"
    return tipe


def check_target(y: pd.Series):
    """Kembalikan (daftar_error, daftar_peringatan) untuk kolom target."""
    errors, warnings = [], []
    counts = y.value_counts()

    if len(counts) < 2:
        errors.append("Kolom target hanya punya 1 kelas (atau kosong). Klasifikasi butuh minimal 2 kelas.")
        return errors, warnings

    if len(counts) > MAX_CLASSES:
        errors.append(
            f"Kolom target punya {len(counts)} nilai unik, kemungkinan ini bukan kolom kelas. "
            "Pilih kolom target lain."
        )
        return errors, warnings

    kecil = counts[counts < MIN_PER_CLASS]
    if len(kecil) > 0:
        errors.append(
            f"Kelas {list(kecil.index)} punya kurang dari {MIN_PER_CLASS} sampel, "
            "terlalu sedikit untuk split terstratifikasi dan cross validation."
        )

    rasio = counts.max() / counts.min()
    if rasio > 1.5:
        warnings.append(
            f"Kelas tidak seimbang: kelas terbesar {rasio:.1f}x kelas terkecil. "
            "Gunakan macro-F1, jangan hanya accuracy."
        )
    return errors, warnings