"""
StormWatch AI - Figure Generation Script
Generates data visualizations from real results for the end-to-end report.

Usage:
    source .venv/bin/activate
    python scripts/generate_figures.py

Output: docs/figures/*.png
"""

from __future__ import annotations

import glob
import sys
import warnings
from pathlib import Path

# Ensure project root is on sys.path for model pickling
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)

# ──────────────────────────────────────────────
#  1. Load weather data from all CSVs
# ──────────────────────────────────────────────

def load_weather_data() -> pd.DataFrame:
    csv_files = sorted(glob.glob(str(DATA_DIR / "weather_*.csv")))
    if not csv_files:
        print("WARNING: No CSV files found in data/raw/")
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, parse_dates=["time"])
        # Ensure consistent column names
        col_rename = {
            "precipitation_sum": "precipitation",
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "temperature_2m_mean": "temp_mean",
        }
        for old, new in col_rename.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} rows from {len(csv_files)} cities")
    return combined


# ──────────────────────────────────────────────
#  2. Load trained models for feature importance
# ──────────────────────────────────────────────

FEATURE_NAMES = {
    "cyclone": ["lat_abs", "lon", "lat", "pressure_min", "dist_to_land",
                 "year", "month", "dayofyear", "wind_kts"],
    "heatwave": ["temp_max", "temp_max_lag_1", "temp_max_lag_3",
                  "temp_max_roll_mean_3", "temp_max_roll_mean_7",
                  "temp_min", "precipitation", "precipitation_lag_1",
                  "relative_humidity_2m_mean", "wind_speed_10m_max",
                  "pressure_msl_mean", "month_sin", "month_cos", "month"],
    "rainfall": ["precipitation", "precipitation_lag_1", "precipitation_lag_3",
                  "precipitation_roll_mean_3", "precipitation_roll_mean_7",
                  "temp_max", "temp_max_roll_mean_3",
                  "relative_humidity_2m_mean", "wind_speed_10m_max",
                  "pressure_msl_mean", "cloud_cover_mean",
                  "month_sin", "month_cos", "month"],
}


def _get_importance(model_path: Path, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importances from a trained model .pkl file."""
    try:
        model = joblib.load(model_path)
        pipeline = getattr(model, "pipeline", None)
        if pipeline is None:
            return {}
        classifier = pipeline.named_steps.get("classifier")
        if classifier is None or not hasattr(classifier, "feature_importances_"):
            return {}
        importances = classifier.feature_importances_
        return dict(zip(feature_names, importances))
    except Exception as e:
        print(f"  Failed to load {model_path.name}: {e}")
        return {}


# ──────────────────────────────────────────────
#  Figures
# ──────────────────────────────────────────────

def figure_1_model_performance():
    """Bar chart: accuracy and ROC-AUC across the 3 models."""
    models = ["Cyclone Intensity", "Heatwave Detection", "Extreme Rainfall"]
    accuracy = [0.989, 0.994, 0.975]
    roc_auc = [0.994, 0.9982, 0.9744]  # cyclone multi-class ovr AUC
    f1_scores = [0.989, 0.987, 0.970]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, accuracy, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x, roc_auc, width, label="ROC-AUC", color="#55A868")
    bars3 = ax.bar(x + width, f1_scores, width, label="F1 Score", color="#C44E52")

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc="lower right", fontsize=11)

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "model_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def figure_2_cyclone_confusion_matrix():
    """Heatmap: cyclone confusion matrix (6 classes) from report data."""
    cm = np.array([
        [262, 0, 0, 0, 0, 0],
        [4, 327, 1, 0, 0, 0],
        [0, 1, 131, 1, 0, 0],
        [0, 0, 3, 54, 0, 0],
        [0, 0, 0, 1, 58, 0],
        [0, 0, 0, 0, 0, 157],
    ])
    labels = ["Cat 0", "Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Count"}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Cyclone Intensity — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "cyclone_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def figure_3_cyclone_class_distribution():
    """Bar chart: test-set class distribution for cyclone categories."""
    labels = ["Cat 0", "Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]
    counts = [262, 332, 133, 57, 59, 157]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xlabel("Saffir-Simpson Category", fontsize=12)
    ax.set_title("Cyclone Class Distribution (Test Set)", fontsize=14, fontweight="bold")

    for bar, count in zip(bars, counts):
        ax.annotate(str(count),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "cyclone_class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def figure_4_feature_importance(imp_dict: dict[str, float],
                                 model_name: str, filename: str):
    """Horizontal bar chart of top-N feature importances."""
    if not imp_dict:
        print(f"  SKIP {filename}: no feature importances available")
        return

    items = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    names, scores = zip(*items) if items else ([], [])

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), scores, color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"{model_name} — Feature Importance", fontsize=14, fontweight="bold")

    for bar, score in zip(bars, scores):
        ax.annotate(f"{score:.3f}",
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def figure_5_extreme_events_by_zone(df: pd.DataFrame):
    """Grouped bar: extreme event counts by climate zone."""
    if df.empty:
        print("  SKIP extreme_events_by_zone.png: no data")
        return

    zone_col = "zone" if "zone" in df.columns else "Zone"
    if zone_col not in df.columns:
        print("  SKIP extreme_events_by_zone.png: no zone column")
        return

    hw_col = "heatwave_flag" if "heatwave_flag" in df.columns else "heatwave_occurred"
    er_col = "extreme_rainfall" if "extreme_rainfall" in df.columns else "extreme_rain"

    zone_data = df.groupby(zone_col).agg(
        heatwaves=(hw_col, "sum"),
        extreme_rainfall=(er_col, "sum"),
    ).reset_index()

    x = np.arange(len(zone_data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, zone_data["heatwaves"], width,
           label="Heatwaves", color="#E24A33")
    ax.bar(x + width / 2, zone_data["extreme_rainfall"], width,
           label="Extreme Rainfall", color="#348ABD")

    ax.set_xticks(x)
    ax.set_xticklabels(zone_data[zone_col].str.capitalize(), fontsize=11)
    ax.set_ylabel("Total Events (2010–2026)", fontsize=12)
    ax.set_title("Extreme Events by Climate Zone", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = FIGURES_DIR / "extreme_events_by_zone.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def figure_6_monthly_patterns(df: pd.DataFrame):
    """Line plot: monthly distribution of extreme events."""
    if df.empty:
        print("  SKIP monthly_event_patterns.png: no data")
        return

    month_col = "month"
    if month_col not in df.columns:
        print("  SKIP monthly_event_patterns.png: no month column")
        return

    hw_col = "heatwave_flag" if "heatwave_flag" in df.columns else "heatwave_occurred"
    er_col = "extreme_rainfall" if "extreme_rainfall" in df.columns else "extreme_rain"

    monthly = df.groupby(month_col).agg(
        heatwaves=(hw_col, "sum"),
        extreme_rainfall=(er_col, "sum"),
    ).reset_index()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_hw = "#E24A33"
    color_er = "#348ABD"

    ax1.plot(month_names, monthly["heatwaves"], "o-", color=color_hw,
             linewidth=2, markersize=6, label="Heatwaves")
    ax1.fill_between(range(12), monthly["heatwaves"], alpha=0.1, color=color_hw)
    ax1.set_ylabel("Heatwave Count", color=color_hw, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_hw)

    ax2 = ax1.twinx()
    ax2.plot(month_names, monthly["extreme_rainfall"], "s--", color=color_er,
             linewidth=2, markersize=6, label="Extreme Rainfall")
    ax2.fill_between(range(12), monthly["extreme_rainfall"], alpha=0.1, color=color_er)
    ax2.set_ylabel("Extreme Rainfall Count", color=color_er, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_er)

    ax1.set_title("Monthly Distribution of Extreme Events", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Month", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    fig.tight_layout()
    path = FIGURES_DIR / "monthly_event_patterns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def figure_7_city_event_counts(df: pd.DataFrame):
    """Horizontal bar: extreme event counts per city."""
    if df.empty:
        print("  SKIP city_event_counts.png: no data")
        return

    city_col = "city" if "city" in df.columns else "City"
    hw_col = "heatwave_flag" if "heatwave_flag" in df.columns else "heatwave_occurred"
    er_col = "extreme_rainfall" if "extreme_rainfall" in df.columns else "extreme_rain"

    if city_col not in df.columns:
        print("  SKIP city_event_counts.png: no city column")
        return

    city_data = df.groupby(city_col).agg(
        heatwaves=(hw_col, "sum"),
        extreme_rainfall=(er_col, "sum"),
    ).reset_index().sort_values("heatwaves", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = range(len(city_data))
    ax.barh([y_i + 0.2 for y_i in y], city_data["heatwaves"], 0.4,
            label="Heatwaves", color="#E24A33")
    ax.barh([y_i - 0.2 for y_i in y], city_data["extreme_rainfall"], 0.4,
            label="Extreme Rainfall", color="#348ABD")
    ax.set_yticks(list(y))
    ax.set_yticklabels(city_data[city_col], fontsize=9)
    ax.set_xlabel("Total Events (2010–2026)", fontsize=12)
    ax.set_title("Extreme Events by City", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    path = FIGURES_DIR / "city_event_counts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def figure_8_temperature_trends(df: pd.DataFrame):
    """Time series: monthly mean max temp for 4 representative cities."""
    if df.empty:
        print("  SKIP temperature_trends.png: no data")
        return

    representative = ["Mumbai", "Delhi", "Chennai", "Kolkata"]
    city_col = "city" if "city" in df.columns else "City"

    if city_col not in df.columns:
        print("  SKIP temperature_trends.png: no city column")
        return

    temp_col = "temp_max" if "temp_max" in df.columns else "temperature_2m_max"
    if temp_col not in df.columns:
        print("  SKIP temperature_trends.png: no temp_max column")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = ["#E24A33", "#348ABD", "#988ED5", "#FDBF6F"]

    for ax, city, color in zip(axes, representative, colors):
        city_df = df[df[city_col] == city].copy()
        if city_df.empty:
            ax.set_title(f"{city} (no data)")
            continue
        city_df["year_month"] = city_df["time"].dt.to_period("M").astype(str)
        monthly = city_df.groupby("year_month")[temp_col].mean().reset_index()
        monthly["year_month"] = pd.to_datetime(monthly["year_month"])
        monthly = monthly.sort_values("year_month")

        ax.plot(monthly["year_month"], monthly[temp_col],
                color=color, linewidth=0.8, alpha=0.7)
        # Rolling annual mean
        monthly["annual_mean"] = monthly[temp_col].rolling(12).mean()
        ax.plot(monthly["year_month"], monthly["annual_mean"],
                color="black", linewidth=1.5, label="12-month avg")
        ax.set_title(city, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylabel("Max Temp (°C)", fontsize=10)

    fig.suptitle("Daily Maximum Temperature Trends (2010–2026)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = FIGURES_DIR / "temperature_trends.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def figure_9_data_coverage(df: pd.DataFrame):
    """Bar chart: available data rows per city."""
    if df.empty:
        print("  SKIP data_coverage.png: no data")
        return

    city_col = "city" if "city" in df.columns else "City"
    if city_col not in df.columns:
        print("  SKIP data_coverage.png: no city column")
        return

    coverage = df.groupby(city_col).size().reset_index(name="rows")
    coverage = coverage.sort_values("rows", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(coverage)), coverage["rows"],
                   color="#55A868", edgecolor="white")
    ax.set_yticks(range(len(coverage)))
    ax.set_yticklabels(coverage[city_col], fontsize=10)
    ax.set_xlabel("Number of Daily Records", fontsize=12)
    ax.set_title("Data Coverage by City (2010–2026)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, row in zip(bars, coverage["rows"]):
        ax.annotate(f"{row:,}",
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "data_coverage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  StormWatch AI — Figure Generation")
    print("=" * 55)

    # ── Load data ──
    print("\n[1/3] Loading weather data...")
    df = load_weather_data()

    # ── Load models ──
    print("\n[2/3] Loading trained models for feature importance...")
    model_files = {
        "cyclone": MODELS_DIR / "cyclone_model.pkl",
        "heatwave": MODELS_DIR / "heatwave_model.pkl",
        "rainfall": MODELS_DIR / "rainfall_model.pkl",
    }
    importances = {}
    for name, path in model_files.items():
        if path.exists():
            imp = _get_importance(path, FEATURE_NAMES[name])
            importances[name] = imp
            print(f"  {name}: {len(imp)} features extracted")
        else:
            print(f"  {name}: model file not found — skipping")

    # ── Generate figures ──
    print("\n[3/3] Generating figures...\n")

    print("  Figure 1: Model Performance Comparison")
    figure_1_model_performance()

    print("  Figure 2: Cyclone Confusion Matrix")
    figure_2_cyclone_confusion_matrix()

    print("  Figure 3: Cyclone Class Distribution")
    figure_3_cyclone_class_distribution()

    print("  Figure 4a: Cyclone Feature Importance")
    figure_4_feature_importance(importances.get("cyclone", {}),
                                 "Cyclone Intensity", "feature_importance_cyclone.png")

    print("  Figure 4b: Heatwave Feature Importance")
    figure_4_feature_importance(importances.get("heatwave", {}),
                                 "Heatwave Detection", "feature_importance_heatwave.png")

    print("  Figure 4c: Rainfall Feature Importance")
    figure_4_feature_importance(importances.get("rainfall", {}),
                                 "Extreme Rainfall", "feature_importance_rainfall.png")

    print("  Figure 5: Extreme Events by Climate Zone")
    figure_5_extreme_events_by_zone(df)

    print("  Figure 6: Monthly Event Patterns")
    figure_6_monthly_patterns(df)

    print("  Figure 7: City Event Counts")
    figure_7_city_event_counts(df)

    print("  Figure 8: Temperature Trends")
    figure_8_temperature_trends(df)

    print("  Figure 9: Data Coverage")
    figure_9_data_coverage(df)

    # ── Summary ──
    generated = list(FIGURES_DIR.glob("*.png"))
    print(f"\n{'=' * 55}")
    print(f"  Generated {len(generated)} figures in docs/figures/")
    print(f"{'=' * 55}")
    for f in sorted(generated):
        size = f.stat().st_size
        print(f"  {f.name:40s} {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
