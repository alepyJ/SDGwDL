from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Estructura auxiliar para organizar los resultados de utilidad.
@dataclass
class UtilityResult:
    classifier: str
    train_source: str
    accuracy: float
    f1: float
    roc_auc: float | None

# Se usa el mismo preprocesado para datos reales y sintéticos
# para que la comparación de utilidad sea coherente.
def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ])


def _sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({pd.NA: np.nan})

# El target se codifica de forma consistente para comparar
# modelos entrenados con distintas fuentes de datos.
def _encode_target(y_train: pd.Series, y_test: pd.Series) -> tuple[pd.Series, pd.Series, LabelEncoder]:
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train.astype(str))
    y_test_encoded = encoder.transform(y_test.astype(str))
    return y_train_encoded, y_test_encoded, encoder

# Estas métricas resumen si la señal predictiva del dataset
# se conserva tras la síntesis.
def _metrics(y_true, y_pred, y_proba=None) -> tuple[float, float, float | None]:
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="binary" if len(set(y_true)) == 2 else "macro"))
    roc_auc = None
    if y_proba is not None and len(set(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    return accuracy, f1, roc_auc

# TSTR: entrenar en sintético y evaluar en real para medir
# si los datos generados sirven en una tarea downstream.
def run_tstr_baselines(
    real_train_df: pd.DataFrame,
    synth_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    target: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    seed: int = 42,
) -> list[dict]:
    feature_columns = numeric_columns + categorical_columns
    X_test = _sanitize_features(real_test_df[feature_columns])
    y_test = real_test_df[target]

    results = []

    # Se prueban dos familias de clasificadores para no depender
    # de un único tipo de modelo predictivo.
    for classifier_name, estimator in {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }.items():
        for train_source, train_df in {
            "real": real_train_df,
            "synthetic": synth_train_df,
        }.items():
            X_train = _sanitize_features(train_df[feature_columns])
            y_train = train_df[target]
            y_train_enc, y_test_enc, _ = _encode_target(y_train, y_test)

            pipeline = Pipeline([
                ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
                ("classifier", estimator),
            ])
            pipeline.fit(X_train, y_train_enc)
            y_pred = pipeline.predict(X_test)
            y_proba = None
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_test)
                if probs.shape[1] == 2:
                    y_proba = probs[:, 1]
            accuracy, f1, roc_auc = _metrics(y_test_enc, y_pred, y_proba)
            results.append({
                "classifier": classifier_name,
                "train_source": train_source,
                "accuracy": accuracy,
                "f1": f1,
                "roc_auc": roc_auc,
            })
    return results
