"""
StormWatch AI - PySpark ETL Module

Reads raw weather CSVs, performs distributed feature engineering via PySpark,
and writes partitioned Parquet files for downstream ML training.

Why PySpark?
  - Handles 84K+ rows across 14 cities with explicit Window-based partitioning
  - Lag features and rolling statistics are expressed naturally via Window
    functions (no fragile groupby-shift chains)
  - Parquet output with automatic partitioning by city enables predicate
    pushdown during model training
  - Scales to thousands of cities without changing a line of ETL code

Usage:
    python -m stormwatch.data.spark_etl                    # run ETL
    python -m stormwatch.data.spark_etl --output parquet   # default
    python -m stormwatch.data.spark_etl --format csv       # CSV output
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

RAW_COLUMNS: List[str] = [
    "time",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "pressure_msl_mean",
    "relative_humidity_2m_mean",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "latitude",
    "longitude",
    "city",
    "state",
    "zone",
]

TEMPERATURE_THRESHOLD: float = 40.0  # °C — heatwave threshold
SEVERE_TEMPERATURE_THRESHOLD: float = 45.0  # °C — severe heatwave
CYCLONE_WIND_THRESHOLD: float = 60.0  # km/h
CYCLONE_PRESSURE_THRESHOLD: float = 1005.0  # hPa
CONSECUTIVE_DAYS: int = 3  # consecutive days for heatwave flag

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "processed"


# ──────────────────────────────────────────────
#  SparkSession management
# ──────────────────────────────────────────────


def create_spark_session(app_name: str = "StormWatch-ETL") -> SparkSession:
    """Create and return a local SparkSession configured for the project.

    Uses JDK 21 for compatibility — Spark 4.x bundled Hadoop does not
    support JDK 24+ (``Subject.getSubject()`` was removed in JDK 24).
    """
    java_home = os.environ.get("JAVA_HOME", "")
    if "java-25" in java_home or "java-24" in java_home:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
        os.environ.pop("PYSPARK_PYTHON", None)
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


# ──────────────────────────────────────────────
#  Ingestion
# ──────────────────────────────────────────────


def read_weather_csvs(
    spark: SparkSession,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> DataFrame:
    """Read all weather_*.csv files into a single PySpark DataFrame.

    Selects only the raw Open-Meteo columns (no pre-engineered features),
    infers schema headers, and filters to valid records.
    """
    csv_files = sorted(raw_dir.glob("weather_*.csv"))
    if not csv_files:
        print(f"[spark_etl] No weather_*.csv files found in {raw_dir}")
        sys.exit(1)

    print(f"[spark_etl] Reading {len(csv_files)} CSV files from {raw_dir}")

    df = spark.read.csv(
        [str(f) for f in csv_files],
        header=True,
        inferSchema=True,
        mode="PERMISSIVE",
    )

    # Keep only the truly raw columns (ignore pre-engineered ones)
    available = [c for c in RAW_COLUMNS if c in df.columns]
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        print(f"[spark_etl] Note: raw columns not in CSV: {missing}")

    df = df.select(*available)
    print(f"[spark_etl] Read {df.count():,} rows across {available} columns")
    return df


# ──────────────────────────────────────────────
#  Feature Engineering (PySpark-native)
# ──────────────────────────────────────────────


def rename_weather_columns(df: DataFrame) -> DataFrame:
    """Rename Open-Meteo column names to shorter model-friendly names."""
    # Keep names matching existing model feature builders:
    #   wind_speed_10m_max, pressure_msl_mean, relative_humidity_2m_mean,
    #   cloud_cover_mean are NOT renamed.
    renames: Dict[str, str] = {
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "temperature_2m_mean": "temp_mean",
        "precipitation_sum": "precipitation",
        "wind_gusts_10m_max": "wind_gust_max",
        "wind_direction_10m_dominant": "wind_direction",
        "shortwave_radiation_sum": "solar_radiation",
        "et0_fao_evapotranspiration": "evapotranspiration",
    }
    for old, new in renames.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    return df


def add_seasonal_features(df: DataFrame, time_col: str = "time") -> DataFrame:
    """Extract month, dayofyear, and cyclic encoding from the time column.

    Uses PySpark's built-in date functions.
    """
    df = df.withColumn("time_parsed", F.to_timestamp(F.col(time_col), "yyyy-MM-dd"))
    df = df.withColumn("month", F.month("time_parsed"))
    df = df.withColumn("dayofyear", F.dayofyear("time_parsed"))
    df = df.withColumn("month_sin", F.sin(2 * math.pi * F.col("month") / 12))
    df = df.withColumn("month_cos", F.cos(2 * math.pi * F.col("month") / 12))
    return df.drop("time_parsed")


def label_extreme_events_pyspark(df: DataFrame) -> DataFrame:
    """Label extreme weather events using PySpark when/otherwise expressions.

    Matches the logic in ``preprocess.label_extreme_events`` but uses
    Spark-native partitioning and Window functions:

    - heatwave_flag: temp_max > 40°C for 3+ consecutive days per city
    - severe_heatwave_flag: temp_max > 45°C for 3+ consecutive days
    - extreme_rainfall: precipitation > 95th percentile per city
    - heavy_rainfall: precipitation > 99th percentile per city
    - cyclonic_flag: wind_gusts > 60 km/h AND pressure < 1005 hPa
    """
    city_window = Window.partitionBy("city").orderBy("time")

    # ── Heatwave streaks via Window ──
    df = df.withColumn(
        "above_heatwave",
        F.when(F.col("temp_max") > TEMPERATURE_THRESHOLD, 1).otherwise(0),
    )
    df = df.withColumn(
        "above_severe",
        F.when(F.col("temp_max") > SEVERE_TEMPERATURE_THRESHOLD, 1).otherwise(0),
    )

    # Consecutive streak: detect change points using lag comparison
    # streak = cumulative sum within each group of consecutive 1s
    df = df.withColumn(
        "hw_change_flag",
        F.when(
            F.col("above_heatwave") != F.lag("above_heatwave", 1).over(city_window),
            1,
        ).otherwise(0),
    )

    df = df.withColumn(
        "hw_group",
        F.sum("hw_change_flag").over(
            city_window.rowsBetween(Window.unboundedPreceding, 0)
        ),
    )

    df = df.withColumn(
        "heatwave_streak",
        F.row_number().over(Window.partitionBy("city", "hw_group").orderBy("time"))
        * F.col("above_heatwave"),
    )

    # Same for severe heatwave
    df = df.withColumn(
        "severe_change_flag",
        F.when(
            F.col("above_severe") != F.lag("above_severe", 1).over(city_window),
            1,
        ).otherwise(0),
    )
    df = df.withColumn(
        "severe_group",
        F.sum("severe_change_flag").over(
            city_window.rowsBetween(Window.unboundedPreceding, 0)
        ),
    )
    df = df.withColumn(
        "severe_heatwave_streak",
        F.row_number().over(Window.partitionBy("city", "severe_group").orderBy("time"))
        * F.col("above_severe"),
    )

    df = df.withColumn(
        "heatwave_flag",
        F.when(F.col("heatwave_streak") >= CONSECUTIVE_DAYS, 1).otherwise(0),
    )
    df = df.withColumn(
        "severe_heatwave_flag",
        F.when(F.col("severe_heatwave_streak") >= CONSECUTIVE_DAYS, 1).otherwise(0),
    )

    # ── Percentile-based rainfall thresholds per city ──
    # PySpark's percentile_approx computes approximate quantiles
    perc_expr = F.expr("percentile_approx(precipitation, 0.95)")
    severe_perc_expr = F.expr("percentile_approx(precipitation, 0.99)")

    city_thresholds = df.groupBy("city").agg(
        perc_expr.alias("p95_threshold"),
        severe_perc_expr.alias("p99_threshold"),
    )

    df = df.join(city_thresholds, on="city", how="left")

    df = df.withColumn(
        "extreme_rainfall",
        F.when(F.col("precipitation") > F.col("p95_threshold"), 1).otherwise(0),
    )
    df = df.withColumn(
        "heavy_rainfall",
        F.when(F.col("precipitation") > F.col("p99_threshold"), 1).otherwise(0),
    )

    # ── Cyclonic conditions ──
    has_wind = "wind_gust_max" in df.columns
    has_pressure = "pressure_msl_mean" in df.columns  # not in CSV, default 0

    if has_wind and has_pressure:
        df = df.withColumn(
            "cyclonic_flag",
            F.when(
                (F.col("wind_gust_max") > CYCLONE_WIND_THRESHOLD)
                & (F.col("pressure_msl_mean") < CYCLONE_PRESSURE_THRESHOLD),
                1,
            ).otherwise(0),
        )
    elif has_wind:
        df = df.withColumn(
            "cyclonic_flag",
            F.when(F.col("wind_gust_max") > CYCLONE_WIND_THRESHOLD, 1).otherwise(0),
        )
    else:
        df = df.withColumn("cyclonic_flag", F.lit(0))

    # Drop intermediate columns
    drop_cols = [
        "above_heatwave",
        "above_severe",
        "heatwave_streak",
        "severe_heatwave_streak",
        "hw_change_flag",
        "hw_group",
        "severe_change_flag",
        "severe_group",
        "p95_threshold",
        "p99_threshold",
    ]
    df = df.drop(*[c for c in drop_cols if c in df.columns])

    return df


def add_lag_features(df: DataFrame, lags: List[int] = None) -> DataFrame:
    """Create lagged weather features using PySpark Window functions.

    Uses ``Window.partitionBy("city").orderBy("time")`` for per-city
    time-series lags — equivalent to ``df.groupby("city")[col].shift(lag)``
    in pandas, but explicit and parallelisable.
    """
    if lags is None:
        lags = [1, 3, 7]

    city_window = Window.partitionBy("city").orderBy("time")

    lag_cols = [
        "temp_max",
        "temp_min",
        "precipitation",
        "wind_speed_max",
    ]
    for col_name in lag_cols:
        if col_name not in df.columns:
            continue
        for lag in lags:
            df = df.withColumn(
                f"{col_name}_lag_{lag}",
                F.lag(col_name, lag).over(city_window),
            )

    return df


def add_rolling_stats(df: DataFrame, windows: List[int] = None) -> DataFrame:
    """Add rolling mean and standard deviation features.

    Uses ``Window.partitionBy("city").orderBy("time").rowsBetween`` for
    per-city rolling windows — equivalent to pandas ``rolling(W).mean()``.
    """
    if windows is None:
        windows = [3, 7]

    city_window = Window.partitionBy("city").orderBy("time")

    roll_cols = ["temp_max", "precipitation"]
    for col_name in roll_cols:
        if col_name not in df.columns:
            continue
        for window in windows:
            w_spec = city_window.rowsBetween(-(window - 1), 0)
            df = df.withColumn(
                f"{col_name}_roll_mean_{window}",
                F.mean(col_name).over(w_spec),
            )
            df = df.withColumn(
                f"{col_name}_roll_std_{window}",
                F.stddev(col_name).over(w_spec),
            )

    # Fill NaN stddev (first row in window) with 0 — matches preprocess.py
    for col_name in roll_cols:
        if col_name not in df.columns:
            continue
        for window in windows:
            std_col = f"{col_name}_roll_std_{window}"
            if std_col in df.columns:
                df = df.withColumn(std_col, F.coalesce(F.col(std_col), F.lit(0.0)))

    return df


# ──────────────────────────────────────────────
#  Full ETL pipeline
# ──────────────────────────────────────────────


def run_etl(
    spark: Optional[SparkSession] = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_format: str = "parquet",
) -> DataFrame:
    """Run the complete PySpark ETL pipeline.

    Steps:
    1. Read raw CSVs (raw Open-Meteo columns only)
    2. Rename columns to model-friendly names
    3. Add seasonal features (month, dayofyear, sin/cos)
    4. Label extreme events (heatwave, rainfall, cyclonic)
    5. Create lag features (1, 3, 7 day)
    6. Add rolling statistics (3, 7 day window)
    7. Write partitioned Parquet output

    Returns:
        The processed DataFrame (cached for inspection)
    """
    close_session = spark is None
    if spark is None:
        spark = create_spark_session()

    try:
        # Step 1-2: Read CSVs and rename columns
        raw_df = read_weather_csvs(spark, raw_dir=raw_dir)
        df = rename_weather_columns(raw_df)
        print(f"[spark_etl] Raw rows: {df.count():,}")

        print("[spark_etl] Adding seasonal features ...")
        df = add_seasonal_features(df)

        # Step 4: Label extreme events
        print("[spark_etl] Labeling extreme events ...")
        df = label_extreme_events_pyspark(df)

        # Step 5: Create lag features (1, 3, 7 day)
        print("[spark_etl] Creating lag features ...")
        df = add_lag_features(df)

        # Step 6: Add rolling statistics (3, 7 day window)
        print("[spark_etl] Adding rolling statistics ...")
        df = add_rolling_stats(df)

        # Drop rows where all lag features are null (first rows per city)
        lag_cols = [c for c in df.columns if "_lag_" in c]
        if lag_cols:
            df = df.dropna(subset=lag_cols, how="all")

        # Sort for consistent output
        df = df.orderBy("city", "time")

        # Step 7: Write output
        output_dir.mkdir(parents=True, exist_ok=True)
        count = df.count()
        print(f"[spark_etl] Writing {count:,} rows to {output_dir} ({output_format})")

        if output_format == "parquet":
            df.write.mode("overwrite").partitionBy("city").parquet(
                str(output_dir / "weather_pyspark.parquet")
            )
        else:
            df.coalesce(1).write.mode("overwrite").option("header", True).csv(
                str(output_dir / "weather_pyspark_csv")
            )

        # Event summary
        event_counts = df.agg(
            F.sum("heatwave_flag").alias("heatwaves"),
            F.sum("severe_heatwave_flag").alias("severe_heatwaves"),
            F.sum("extreme_rainfall").alias("extreme_rain"),
            F.sum("cyclonic_flag").alias("cyclonic_events"),
        ).collect()[0]

        print("[spark_etl] Event summary:")
        print(f"  Heatwaves:       {event_counts['heatwaves'] or 0:,}")
        print(f"  Severe HWs:      {event_counts['severe_heatwaves'] or 0:,}")
        print(f"  Extreme rain:    {event_counts['extreme_rain'] or 0:,}")
        print(f"  Cyclonic events: {event_counts['cyclonic_events'] or 0:,}")
        print(f"[spark_etl] ETL complete → {output_dir / 'weather_pyspark.parquet'}")

        return df

    finally:
        if close_session:
            spark.stop()


# ──────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="StormWatch AI - PySpark ETL Pipeline")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory with weather_*.csv files (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    args = parser.parse_args()

    spark = create_spark_session()
    try:
        run_etl(
            spark=spark,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            output_format=args.format,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
