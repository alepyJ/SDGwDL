from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

# Agrupa los splits junto con la información de tipos de columnas
# para reutilizar el mismo esquema en etapas posteriores.
@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target: str
    numeric_columns: list[str]
    categorical_columns: list[str]


NUMERIC_DTYPES = ["number", "float", "int"]

# Los tipos de columna se infieren una sola vez a partir de los datos reales
# y luego se reutilizan en evaluación y modelado.
def infer_column_types(df: pd.DataFrame, target: str, id_columns: list[str] | None = None) -> tuple[list[str], list[str]]:
    id_columns = id_columns or []
    feature_df = df.drop(columns=[target] + id_columns, errors="ignore")
    numeric_columns = feature_df.select_dtypes(include=NUMERIC_DTYPES).columns.tolist()
    categorical_columns = [c for c in feature_df.columns if c not in numeric_columns]
    return numeric_columns, categorical_columns

# Limpieza mínima: solo se corrigen valores vacíos o inconsistentes
# sin transformar en exceso el dataset original.
def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    return out

# El split se hace antes de generar datos sintéticos para evaluar
# siempre contra un conjunto real no visto.
def train_test_split_stratified(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    seed: int = 42,
    id_columns: list[str] | None = None,
) -> SplitBundle:
    clean_df = basic_cleaning(df)
    numeric_columns, categorical_columns = infer_column_types(clean_df, target=target, id_columns=id_columns)
    # Se estratifica cuando el target lo permite para conservar mejor
    # la distribución de clases entre train y test.
    stratify = clean_df[target] if clean_df[target].nunique(dropna=False) <= 20 else None
    train_df, test_df = train_test_split(clean_df, test_size=test_size, random_state=seed, stratify=stratify)

    return SplitBundle(
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        target=target,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
