from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({pd.NA: np.nan})

# Reales y sintéticos se proyectan al mismo espacio de features
# antes de calcular métricas de cercanía.
def _privacy_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ),
    ])

# Distancias muy pequeñas a vecinos reales pueden indicar
# exceso de parecido o posible memorización.
def nearest_neighbor_privacy_metrics(
    real_train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    sample_size: int = 3000,
) -> dict:
    feature_columns = numeric_columns + categorical_columns
    # Se muestrea para mantener el cálculo de distancias manejable
    # en datasets más grandes.
    real_sample = _sanitize_features(real_train_df[feature_columns]).sample(min(sample_size, len(real_train_df)), random_state=42)
    synth_sample = _sanitize_features(synth_df[feature_columns]).sample(min(sample_size, len(synth_df)), random_state=42)

    preprocessor = _privacy_preprocessor(numeric_columns, categorical_columns)
    real_matrix = preprocessor.fit_transform(real_sample)
    synth_matrix = preprocessor.transform(synth_sample)
    
    # Se mide qué tan cerca queda cada fila sintética
    # de su fila real más próxima.
    dist = pairwise_distances(synth_matrix, real_matrix, metric="euclidean")
    min_dist = dist.min(axis=1)
    return {
        "min_distance_mean": float(np.mean(min_dist)),
        "min_distance_median": float(np.median(min_dist)),
        "min_distance_p05": float(np.quantile(min_dist, 0.05)),
    }

# Los duplicados exactos son una señal sencilla pero útil
# de posible memorización del conjunto real.
def exact_duplicate_rate(real_train_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    real_rows = {tuple(row) for row in real_train_df.astype(str).itertuples(index=False, name=None)}
    synth_rows = [tuple(row) for row in synth_df.astype(str).itertuples(index=False, name=None)]
    duplicates = sum(1 for row in synth_rows if row in real_rows)
    total = len(synth_rows) or 1
    return {
        "exact_duplicates": int(duplicates),
        "exact_duplicate_rate": float(duplicates / total),
    }

# Replicar categorías muy raras puede mejorar fidelity,
# pero también aumentar el riesgo de divulgación.
def rare_category_overlap(real_train_df: pd.DataFrame, synth_df: pd.DataFrame, categorical_columns: list[str], rare_threshold: float = 0.01) -> dict:
    overlap = {}
    for col in categorical_columns:
        real_freq = real_train_df[col].astype(str).value_counts(normalize=True, dropna=False)
        rare_categories = set(real_freq[real_freq <= rare_threshold].index)
        synth_values = set(synth_df[col].astype(str).unique())
        overlap[col] = {
            "rare_categories_real": int(len(rare_categories)),
            "rare_categories_reproduced_in_synth": int(len(rare_categories & synth_values)),
        }
    return overlap
