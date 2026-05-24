#!/usr/bin/env python3
import html
from pathlib import Path

import pandas as pd


def relpath(base: Path, target: str) -> str:
    if not target:
        return ""
    try:
        return str(Path(target).resolve().relative_to(base.resolve()))
    except Exception:
        try:
            return str(Path(target).resolve())
        except Exception:
            return ""


def main():
    repo_root = Path(__file__).resolve().parents[2]
    input_csv = repo_root / "output" / "ml_results" / "nn_stageA_inventory.csv"
    output_html = repo_root / "output" / "ml_results" / "nn_stageA_report.html"

    df = pd.read_csv(input_csv)

    # Normalize columns from inventory
    if "test_specificity" not in df.columns and "specificity" in df.columns:
        df["test_specificity"] = df["specificity"]
    if "test_recall" not in df.columns and "recall" in df.columns:
        df["test_recall"] = df["recall"]
    if "test_precision" not in df.columns and "precision" in df.columns:
        df["test_precision"] = df["precision"]
    if "test_f1_score" not in df.columns and "f1" in df.columns:
        df["test_f1_score"] = df["f1"]
    if "test_balanced_accuracy" not in df.columns and "balanced_accuracy" in df.columns:
        df["test_balanced_accuracy"] = df["balanced_accuracy"]
    if "test_roc_auc" not in df.columns and "roc_auc" in df.columns:
        df["test_roc_auc"] = df["roc_auc"]
    if "test_pr_auc" not in df.columns and "pr_auc" in df.columns:
        df["test_pr_auc"] = df["pr_auc"]

    # Summary table
    summary = (
        df.groupby(["min_recall", "nn_imbalance_mode"], as_index=False)
        .agg(
            test_precision=("test_precision", "mean"),
            test_recall=("test_recall", "mean"),
            test_specificity=("test_specificity", "mean"),
            test_f1_score=("test_f1_score", "mean"),
            test_balanced_accuracy=("test_balanced_accuracy", "mean"),
            test_roc_auc=("test_roc_auc", "mean"),
            test_pr_auc=("test_pr_auc", "mean"),
        )
        .sort_values(["min_recall", "nn_imbalance_mode"])
    )

    # Stage B shortlist (same rules as aggregate script)
    shortlist = {}
    for min_recall in [0.85, 0.9]:
        subset = summary[summary["min_recall"] == min_recall].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["test_specificity", "test_precision"],
            ascending=[False, False]
        )
        shortlist[min_recall] = subset["nn_imbalance_mode"].head(2).tolist()

    shortlist_modes = []
    if 0.85 in shortlist and 0.9 in shortlist:
        if shortlist[0.85] and shortlist[0.9] and shortlist[0.85][0] == shortlist[0.9][0]:
            shortlist_modes.append(shortlist[0.85][0])
            if len(shortlist[0.85]) > 1:
                shortlist_modes.append(shortlist[0.85][1])
        else:
            shortlist_modes.extend(shortlist.get(0.85, []))
            shortlist_modes.extend(shortlist.get(0.9, []))
    else:
        for modes in shortlist.values():
            shortlist_modes.extend(modes)

    seen = set()
    deduped = []
    for mode in shortlist_modes:
        if mode not in seen:
            seen.add(mode)
            deduped.append(mode)

    winners = {}
    for min_recall in [0.85, 0.9]:
        subset = summary[summary["min_recall"] == min_recall].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["test_specificity", "test_precision"],
            ascending=[False, False]
        )
        winners[min_recall] = subset.iloc[0]["nn_imbalance_mode"]

    # Build HTML
    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html lang='en'>")
    lines.append("<head>")
    lines.append("  <meta charset='UTF-8'>")
    lines.append("  <title>NN Stage A Report</title>")
    lines.append("  <style>")
    lines.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
    lines.append("    table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }")
    lines.append("    th, td { border: 1px solid #ccc; padding: 6px 8px; font-size: 13px; }")
    lines.append("    th { background: #f0f0f0; }")
    lines.append("    h2 { margin-top: 28px; }")
    lines.append("    .small { font-size: 12px; color: #555; }")
    lines.append("  </style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append("<h1>NN Stage A Report</h1>")

    # Summary table
    lines.append("<h2>Summary by Constraint and Mode</h2>")
    lines.append("<table>")
    lines.append("  <tr>")
    for col in summary.columns:
        lines.append(f"    <th>{html.escape(str(col))}</th>")
    lines.append("  </tr>")
    for _, row in summary.iterrows():
        lines.append("  <tr>")
        for col in summary.columns:
            val = row[col]
            if isinstance(val, float):
                cell = f"{val:.4f}"
            else:
                cell = str(val)
            lines.append(f"    <td>{html.escape(cell)}</td>")
        lines.append("  </tr>")
    lines.append("</table>")

    # Links per run
    lines.append("<h2>Run Artifacts</h2>")
    lines.append("<table>")
    cols = [
        "min_recall",
        "nn_imbalance_mode",
        "timestamp",
        "val_roc_plot",
        "val_pr_plot",
        "test_roc_plot",
        "test_pr_plot",
        "val_threshold_sweep_all_csv",
        "val_threshold_sweep_constrained_csv",
        "metrics_json",
    ]
    lines.append("  <tr>")
    for col in cols:
        lines.append(f"    <th>{html.escape(col)}</th>")
    lines.append("  </tr>")

    for _, row in df.iterrows():
        lines.append("  <tr>")
        for col in cols:
            val = row.get(col, "")
            if col.endswith("_plot") or col.endswith("_csv") or col == "metrics_json":
                link = relpath(output_html.parent, val)
                if link:
                    cell = f"<a href='{html.escape(link)}'>{html.escape(Path(link).name)}</a>"
                else:
                    cell = ""
            else:
                cell = html.escape(str(val))
            lines.append(f"    <td>{cell}</td>")
        lines.append("  </tr>")
    lines.append("</table>")

    # Stage B section
    lines.append("<h2>Stage B Recommendation</h2>")
    lines.append("<h3>Winners</h3>")
    if winners:
        lines.append("<ul>")
        for min_recall, mode in winners.items():
            lines.append(f"  <li>min_recall={min_recall}: <strong>{html.escape(mode)}</strong></li>")
        lines.append("</ul>")
    else:
        lines.append("<p class='small'>No winners found.</p>")

    lines.append("<h3>Shortlist</h3>")
    if deduped:
        lines.append("<ul>")
        for mode in deduped:
            lines.append(f"  <li>{html.escape(mode)}</li>")
        lines.append("</ul>")
    else:
        lines.append("<p class='small'>No shortlist candidates found.</p>")

    lines.append("</body>")
    lines.append("</html>")

    output_html.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {output_html}")


if __name__ == "__main__":
    main()
