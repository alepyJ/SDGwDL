from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from ucimlrepo import fetch_ucirepo

# Estructura mínima para registrar cada dataset:
# origen, identificador y columna objetivo.
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    uci_id: int
    target: str
    task: str = "binary_classification"
    description: str = ""

# Conjunto inicial de datasets tabulares usados en los experimentos.
DATASETS: dict[str, DatasetSpec] = {
    "adult": DatasetSpec(
        name="adult",
        uci_id=2,
        target="income",
        description="Adult / Census Income dataset",
    ),
    "bank_marketing": DatasetSpec(
        name="bank_marketing",
        uci_id=222,
        target="y",
        description="Bank Marketing dataset",
    ),
    "default_credit": DatasetSpec(
        name="default_credit",
        uci_id=350,
        target="default payment next month",
        description="Default of Credit Card Clients dataset",
    ),
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all").reset_index(drop=True)


def _coerce_question_mark_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace("?", pd.NA)


def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
    return out

# Aquí se aíslan correcciones específicas de ciertos datasets
# para no ensuciar el flujo general de carga.
def _normalize_known_targets(df: pd.DataFrame, dataset_name: str, target_col: str) -> pd.DataFrame:
    out = df.copy()
    if target_col not in out.columns:
        return out

    if dataset_name == "adult":
        out[target_col] = out[target_col].astype(str).str.replace(".", "", regex=False).str.strip()
    return out

# Carga el dataset, une variables y target en una sola tabla
# y aplica una normalización básica antes del preprocesado.
def load_dataset(name: str) -> tuple[pd.DataFrame, str, DatasetSpec]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")

    spec = DATASETS[name]
    uci = fetch_ucirepo(id=spec.uci_id)

    features = _normalize_columns(uci.data.features)
    targets = _normalize_columns(uci.data.targets)

    if isinstance(targets, pd.Series):
        targets = targets.to_frame(name=spec.target)
        
    # El proyecto trabaja con una única tabla donde el target
    # se mantiene como una columna más.
    df = pd.concat([features, targets], axis=1)
    df = _strip_object_columns(df)
    df = _coerce_question_mark_to_nan(df)
    df = _drop_empty_rows(df)

    if spec.target not in df.columns and len(targets.columns) == 1:
        target_col = targets.columns[0]
    else:
        target_col = spec.target

    df = _normalize_known_targets(df, spec.name, target_col)

    return df, target_col, spec


def list_datasets() -> list[str]:
    return sorted(DATASETS)
