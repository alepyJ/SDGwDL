from __future__ import annotations

import math

import numpy as np
import pandas as pd

# La similitud numérica se resume con estadísticas simples
def summarize_numeric_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    summary = {}
    for col in numeric_columns:
        real = pd.to_numeric(real_df[col], errors="coerce")
        synth = pd.to_numeric(synth_df[col], errors="coerce")
        summary[col] = {
            "real_mean": None if real.dropna().empty else float(real.mean()),
            "synth_mean": None if synth.dropna().empty else float(synth.mean()),
            "mean_abs_diff": None if real.dropna().empty or synth.dropna().empty else float(abs(real.mean() - synth.mean())),
            "real_std": None if real.dropna().empty else float(real.std()),
            "synth_std": None if synth.dropna().empty else float(synth.std()),
        }
    return summary

# Para variables categóricas se comparan distribuciones de frecuencia
# entre tablas reales y sintéticas.
def summarize_categorical_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame, categorical_columns: list[str]) -> dict:
    summary = {}
    for col in categorical_columns:
        real_freq = real_df[col].astype(str).value_counts(normalize=True, dropna=False)
        synth_freq = synth_df[col].astype(str).value_counts(normalize=True, dropna=False)
        categories = sorted(set(real_freq.index).union(set(synth_freq.index)))
        l1_distance = 0.0
        for category in categories:
            l1_distance += abs(float(real_freq.get(category, 0.0)) - float(synth_freq.get(category, 0.0)))
        summary[col] = {
            # La distancia L1 resume cuánto se separan ambas distribuciones categóricas.
            "l1_distance": float(l1_distance),
            "num_categories_real": int(real_freq.shape[0]),
            "num_categories_synth": int(synth_freq.shape[0]),
        }
    return summary

# También se comprueba si se conservan aproximadamente
# las relaciones entre variables numéricas.
def correlation_gap(real_df: pd.DataFrame, synth_df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    # Esta métrica solo tiene sentido si existen al menos dos variables numéricas.
    if len(numeric_columns) < 2:
        return {"mean_abs_corr_gap": None}
    real_corr = real_df[numeric_columns].corr(numeric_only=True).fillna(0)
    synth_corr = synth_df[numeric_columns].corr(numeric_only=True).fillna(0)
    gap = (real_corr - synth_corr).abs().values
    return {"mean_abs_corr_gap": float(np.mean(gap))}
