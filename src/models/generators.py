from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
except Exception:  # pragma: no cover
    SingleTableMetadata = None
    CTGANSynthesizer = None
    TVAESynthesizer = None


@dataclass
class TrainedGenerator:
    name: str
    synthesizer: object
    metadata: object


def _build_metadata(train_df: pd.DataFrame) -> object:
    if SingleTableMetadata is None:
        raise ImportError("sdv is required to train CTGAN/TVAE. Install dependencies from requirements.txt")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)
    return metadata


def train_generator(model_name: str, train_df: pd.DataFrame, seed: int = 42, epochs: int = 300) -> TrainedGenerator:
    metadata = _build_metadata(train_df)

    model_name = model_name.lower()
    if model_name == "ctgan":
        if CTGANSynthesizer is None:
            raise ImportError("CTGANSynthesizer unavailable. Check sdv installation.")
        synthesizer = CTGANSynthesizer(metadata, enforce_rounding=False, epochs=epochs, verbose=True)
    elif model_name == "tvae":
        if TVAESynthesizer is None:
            raise ImportError("TVAESynthesizer unavailable. Check sdv installation.")
        synthesizer = TVAESynthesizer(metadata, enforce_rounding=False, epochs=epochs, verbose=True)
    else:
        raise ValueError(f"Unsupported model '{model_name}'")

    synthesizer.fit(train_df)
    return TrainedGenerator(name=model_name, synthesizer=synthesizer, metadata=metadata)


def sample_synthetic(trained: TrainedGenerator, num_rows: int) -> pd.DataFrame:
    return trained.synthesizer.sample(num_rows=num_rows)
