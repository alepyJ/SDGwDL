from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
# Permite ejecutar este script directamente desde la raíz del proyecto.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import write_text

# Helper pequeño para simplificar el formateo del markdown.
def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"

# Reorganiza los resultados de utilidad para facilitar
# comparaciones dentro del reporte.
def _utility_by_model(summary: dict) -> dict[str, dict[tuple[str, str], dict]]:
    out: dict[str, dict[tuple[str, str], dict]] = {}
    for model_name, model_payload in summary.get("models", {}).items():
        out[model_name] = {}
        for row in model_payload.get("utility", []):
            out[model_name][(row["classifier"], row["train_source"])] = row
    return out

# Genera un resumen legible a partir de summary.json
# sin depender de inspección manual de métricas crudas.
def build_markdown(summary: dict) -> str:
    dataset = summary["dataset"]
    target = summary["target"]
    lines = [
        f"# Experiment report — {dataset}",
        "",
        f"- Target: `{target}`",
        f"- Train rows: {summary['rows_train']}",
        f"- Test rows: {summary['rows_test']}",
        f"- Models: {', '.join(summary.get('models', {}).keys()) or 'none'}",
        "",
    ]

    utility = _utility_by_model(summary)
    # El reporte prioriza señales interpretables en lugar
    # de mostrar todas las métricas disponibles.
    for model_name, payload in summary.get("models", {}).items():
        fidelity = payload.get("fidelity", {})
        privacy = payload.get("privacy", {})

        lines.extend([
            f"## Model: {model_name}",
            "",
            "### Fidelity",
            f"- Mean absolute correlation gap: {_fmt(fidelity.get('correlation_gap', {}).get('mean_abs_corr_gap'))}",
        ])

        numeric = fidelity.get("numeric", {})
        if numeric:
            # Se destaca la mayor desviación numérica para identificar rápido
            # dónde falla más el generador.
            worst_num = max(
                numeric.items(),
                key=lambda item: float("-inf") if item[1].get("mean_abs_diff") is None else item[1]["mean_abs_diff"],
            )
            lines.append(
                f"- Largest numeric mean shift: `{worst_num[0]}` → {_fmt(worst_num[1].get('mean_abs_diff'))}"
            )

        categorical = fidelity.get("categorical", {})
        if categorical:
            # Lo mismo para variables categóricas: se resume
            # la mayor diferencia de distribución.
            worst_cat = max(categorical.items(), key=lambda item: item[1].get("l1_distance", 0.0))
            lines.append(
                f"- Largest categorical distribution gap: `{worst_cat[0]}` → {_fmt(worst_cat[1].get('l1_distance'))}"
            )

        lines.extend([
            "",
            "### Utility (TSTR)",
        ])

        model_utility = utility.get(model_name, {})
        for clf_name in ["logistic_regression", "random_forest"]:
            real_row = model_utility.get((clf_name, "real"))
            synth_row = model_utility.get((clf_name, "synthetic"))
            if real_row and synth_row:
                lines.append(
                    f"- `{clf_name}`: real ROC-AUC {_fmt(real_row.get('roc_auc'))} vs synthetic ROC-AUC {_fmt(synth_row.get('roc_auc'))}; Δ = {_fmt((real_row.get('roc_auc') or 0.0) - (synth_row.get('roc_auc') or 0.0))}"
                )

        nn = privacy.get("nearest_neighbor", {})
        dup = privacy.get("duplicates", {})
        lines.extend([
            "",
            "### Privacy",
            f"- Nearest-neighbor mean distance: {_fmt(nn.get('min_distance_mean'))}",
            f"- Nearest-neighbor 5th percentile: {_fmt(nn.get('min_distance_p05'))}",
            f"- Exact duplicate rate: {_fmt(dup.get('exact_duplicate_rate'))}",
            "",
        ])

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a short markdown report from an experiment summary.json")
    parser.add_argument("summary", help="Path to summary.json")
    parser.add_argument("--output", help="Optional output markdown path")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # El reporte se construye a partir del resumen guardado,
    # manteniendo separadas ejecución y presentación.
    markdown = build_markdown(summary)

    output_path = Path(args.output) if args.output else summary_path.with_name("report.md")
    write_text(output_path, markdown)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
