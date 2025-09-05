"""Polars-centric data processing for Brazil inflation data."""

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl
from rich.console import Console

from .config import Settings, get_settings
from .exceptions import DataProcessingError
from .models import DataSource, InflationDataset, DataQualityReport

console = Console()


class BrazilInflationProcessor:
    """
    Polars-centric processor for Brazil inflation data.
    
    Focuses on efficient data processing using Polars lazy evaluation,
    optimized data types, and native Parquet integration.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize processor with configuration."""
        self.settings = settings or get_settings()
        
    def process_dataset(
        self,
        dataset: InflationDataset,
        apply_transformations: bool = True,
        calculate_derived_metrics: bool = True
    ) -> pl.LazyFrame:
        """
        Process inflation dataset using Polars lazy evaluation.
        
        Args:
            dataset: Input inflation dataset
            apply_transformations: Whether to apply data transformations
            calculate_derived_metrics: Whether to calculate derived metrics
            
        Returns:
            Processed Polars LazyFrame
        """
        console.print(f"[blue]Processing {dataset.source.value} dataset with {len(dataset.data_points)} observations[/blue]")
        
        # Convert to Polars DataFrame with optimized types
        df = self._dataset_to_polars(dataset)
        
        # Start with lazy evaluation
        lf = df.lazy()
        
        # Apply basic transformations
        if apply_transformations:
            lf = self._apply_data_transformations(lf)
        
        # Calculate derived metrics
        if calculate_derived_metrics:
            lf = self._calculate_derived_metrics(lf)
        
        return lf
    
    def _dataset_to_polars(self, dataset: InflationDataset) -> pl.DataFrame:
        """Convert dataset to optimized Polars DataFrame."""
        data = [
            {
                "date": dp.date,
                "inflation_rate": dp.value,
                "source": dp.source.value,
                "period": dp.period.value if dp.period else "modern",
                "year": dp.date.year,
                "month": dp.date.month,
            }
            for dp in dataset.data_points
        ]
        
        return pl.DataFrame(data).with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("inflation_rate").cast(pl.Float32),
            pl.col("year").cast(pl.Int16),
            pl.col("month").cast(pl.Int8),
            pl.col("source").cast(pl.Utf8),
            pl.col("period").cast(pl.Utf8),
        ])
    
    def _apply_data_transformations(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply data cleaning and transformation operations."""
        return lf.with_columns([
            # Sort by date
            pl.col("date"),
            
            # Handle outliers (cap extreme values)
            pl.when(pl.col("inflation_rate") > 5000.0)
            .then(5000.0)
            .when(pl.col("inflation_rate") < -50.0)
            .then(-50.0)
            .otherwise(pl.col("inflation_rate"))
            .alias("inflation_rate_cleaned"),
            
            # Calculate period flags
            (pl.col("year") < self.settings.data_source.plano_real_year).alias("is_hyperinflation"),
            (pl.col("year") >= self.settings.data_source.plano_real_year).alias("is_modern"),
            
            # Quarter information
            ((pl.col("month") - 1) // 3 + 1).cast(pl.Int8).alias("quarter"),
            
            # Decade classification
            (pl.col("year") // 10 * 10).cast(pl.Int16).alias("decade"),
        ]).sort("date")
    
    def _calculate_derived_metrics(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate derived metrics using Polars window functions."""
        return lf.with_columns([
            # Moving averages (window functions)
            pl.col("inflation_rate_cleaned")
            .rolling_mean(window_size=3)
            .alias("ma_3m"),
            
            pl.col("inflation_rate_cleaned")
            .rolling_mean(window_size=6)
            .alias("ma_6m"),
            
            pl.col("inflation_rate_cleaned")
            .rolling_mean(window_size=12)
            .alias("ma_12m"),
            
            # Rolling volatility (standard deviation)
            pl.col("inflation_rate_cleaned")
            .rolling_std(window_size=12)
            .alias("volatility_12m"),
            
            # Year-over-year change
            pl.col("inflation_rate_cleaned")
            .shift(12)
            .alias("inflation_rate_12m_ago"),
            
            # Cumulative inflation within year
            pl.col("inflation_rate_cleaned")
            .cum_sum()
            .over("year")
            .alias("cumulative_inflation_ytd"),
            
            # Rank within year
            pl.col("inflation_rate_cleaned")
            .rank()
            .over("year")
            .alias("rank_within_year"),
        ])
    
    def aggregate_by_period(
        self,
        lf: pl.LazyFrame,
        period: str = "year"
    ) -> pl.LazyFrame:
        """
        Aggregate data by specified time period using Polars groupby.
        
        Args:
            lf: Input LazyFrame
            period: Aggregation period ('year', 'quarter', 'decade')
            
        Returns:
            Aggregated LazyFrame
        """
        if period == "year":
            group_cols = ["year"]
        elif period == "quarter":
            group_cols = ["year", "quarter"]
        elif period == "decade":
            group_cols = ["decade"]
        else:
            raise DataProcessingError(f"Unsupported aggregation period: {period}")
        
        return lf.group_by(group_cols).agg([
            pl.col("inflation_rate_cleaned").mean().alias("avg_inflation"),
            pl.col("inflation_rate_cleaned").median().alias("median_inflation"),
            pl.col("inflation_rate_cleaned").std().alias("std_inflation"),
            pl.col("inflation_rate_cleaned").min().alias("min_inflation"),
            pl.col("inflation_rate_cleaned").max().alias("max_inflation"),
            pl.col("inflation_rate_cleaned").count().alias("observations"),
            pl.col("volatility_12m").mean().alias("avg_volatility"),
            pl.col("date").min().alias("period_start"),
            pl.col("date").max().alias("period_end"),
        ]).sort(group_cols)
    
    def calculate_hyperinflation_metrics(
        self,
        lf: pl.LazyFrame,
        threshold: float = 50.0
    ) -> pl.DataFrame:
        """
        Calculate hyperinflation-specific metrics.
        
        Args:
            lf: Input LazyFrame
            threshold: Monthly inflation threshold for hyperinflation
            
        Returns:
            Hyperinflation metrics DataFrame
        """
        return lf.filter(
            pl.col("is_hyperinflation") == True
        ).select([
            pl.col("inflation_rate_cleaned").count().alias("hyperinflation_months"),
            (pl.col("inflation_rate_cleaned") > threshold).sum().alias("extreme_months"),
            pl.col("inflation_rate_cleaned").max().alias("peak_inflation"),
            pl.col("inflation_rate_cleaned").mean().alias("avg_hyperinflation"),
            pl.col("volatility_12m").mean().alias("avg_volatility"),
            pl.col("date").filter(
                pl.col("inflation_rate_cleaned") == pl.col("inflation_rate_cleaned").max()
            ).first().alias("peak_date"),
        ]).collect()
    
    def calculate_modern_metrics(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """
        Calculate modern period (post-1994) metrics.
        
        Args:
            lf: Input LazyFrame
            
        Returns:
            Modern period metrics DataFrame
        """
        return lf.filter(
            pl.col("is_modern") == True
        ).select([
            pl.col("inflation_rate_cleaned").count().alias("modern_months"),
            pl.col("inflation_rate_cleaned").mean().alias("avg_modern_inflation"),
            pl.col("inflation_rate_cleaned").std().alias("modern_volatility"),
            pl.col("inflation_rate_cleaned").min().alias("min_modern_inflation"),
            pl.col("inflation_rate_cleaned").max().alias("max_modern_inflation"),
            pl.col("ma_12m").mean().alias("avg_12m_trend"),
            (pl.col("inflation_rate_cleaned") < 0).sum().alias("deflationary_months"),
            (pl.col("inflation_rate_cleaned") > 10).sum().alias("high_inflation_months"),
        ]).collect()
    
    def create_dual_period_comparison(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """
        Create comparison between hyperinflation and modern periods.
        
        Args:
            lf: Input LazyFrame
            
        Returns:
            Comparison DataFrame
        """
        return lf.group_by("period").agg([
            pl.col("inflation_rate_cleaned").mean().alias("avg_inflation"),
            pl.col("inflation_rate_cleaned").median().alias("median_inflation"),
            pl.col("inflation_rate_cleaned").std().alias("volatility"),
            pl.col("inflation_rate_cleaned").min().alias("min_inflation"),
            pl.col("inflation_rate_cleaned").max().alias("max_inflation"),
            pl.col("inflation_rate_cleaned").count().alias("observations"),
            pl.col("date").min().alias("period_start"),
            pl.col("date").max().alias("period_end"),
        ]).sort("period").collect()
    
    def assess_data_quality(self, lf: pl.LazyFrame, source: DataSource) -> DataQualityReport:
        """
        Assess data quality using Polars operations.
        
        Args:
            lf: Input LazyFrame
            source: Data source being assessed
            
        Returns:
            Data quality assessment report
        """
        df = lf.collect()
        
        # Calculate quality metrics
        total_points = df.height
        missing_points = df.select(
            pl.col("inflation_rate_cleaned").null_count()
        ).item()
        
        # Detect outliers using IQR method
        q1 = df.select(pl.col("inflation_rate_cleaned").quantile(0.25)).item()
        q3 = df.select(pl.col("inflation_rate_cleaned").quantile(0.75)).item()
        iqr = q3 - q1
        outlier_threshold_low = q1 - 1.5 * iqr
        outlier_threshold_high = q3 + 1.5 * iqr
        
        outliers = df.select(
            ((pl.col("inflation_rate_cleaned") < outlier_threshold_low) |
             (pl.col("inflation_rate_cleaned") > outlier_threshold_high)).sum()
        ).item()
        
        # Hyperinflation anomalies
        hyperinflation_anomalies = df.filter(
            pl.col("is_hyperinflation") == True
        ).select(
            (pl.col("inflation_rate_cleaned") > 3000).sum()  # Extreme hyperinflation
        ).item() or 0
        
        # Generate issues and recommendations
        issues = []
        recommendations = []
        
        if missing_points > 0:
            issues.append(f"{missing_points} missing data points")
            recommendations.append("Consider interpolation or use alternative source")
        
        if outliers > total_points * 0.05:  # More than 5% outliers
            issues.append(f"High number of outliers: {outliers}")
            recommendations.append("Review outlier handling strategy")
        
        if hyperinflation_anomalies > 0:
            issues.append(f"{hyperinflation_anomalies} extreme hyperinflation values")
            recommendations.append("Validate extreme hyperinflation data points")
        
        return DataQualityReport(
            source=source,
            total_points=total_points,
            missing_points=missing_points,
            outliers=outliers,
            hyperinflation_anomalies=hyperinflation_anomalies,
            issues=issues,
            recommendations=recommendations
        )
    
    def save_processed_data(
        self,
        lf: pl.LazyFrame,
        filename: str,
        format: str = "parquet"
    ) -> Path:
        """
        Save processed data using Polars native formats.
        
        Args:
            lf: LazyFrame to save
            filename: Output filename
            format: Output format ('parquet', 'csv', 'json')
            
        Returns:
            Path to saved file
        """
        output_path = self.settings.storage.processed_dir / f"{filename}.{format}"
        
        if format == "parquet":
            lf.sink_parquet(
                output_path,
                compression=self.settings.storage.parquet_compression,
                row_group_size=50000  # Optimize for analytics
            )
        elif format == "csv":
            lf.sink_csv(output_path, separator=",", quote_char='"')
        else:
            # For JSON, we need to collect first
            df = lf.collect()
            df.write_ndjson(output_path)
        
        console.print(f"[green]Saved processed data to {output_path}[/green]")
        return output_path
    
    def load_processed_data(self, filepath: Path) -> pl.LazyFrame:
        """
        Load processed data using Polars lazy evaluation.
        
        Args:
            filepath: Path to processed data file
            
        Returns:
            LazyFrame with loaded data
        """
        if filepath.suffix == ".parquet":
            return pl.scan_parquet(filepath)
        elif filepath.suffix == ".csv":
            return pl.scan_csv(
                filepath,
                try_parse_dates=True,
                infer_schema_length=10000
            )
        else:
            raise DataProcessingError(f"Unsupported file format: {filepath.suffix}")
    
    def create_analytical_views(self, lf: pl.LazyFrame) -> Dict[str, pl.LazyFrame]:
        """
        Create analytical views for different use cases.
        
        Args:
            lf: Input LazyFrame
            
        Returns:
            Dictionary of analytical views
        """
        views = {}
        
        # Hyperinflation period view
        views["hyperinflation"] = lf.filter(
            pl.col("is_hyperinflation") == True
        ).select([
            "date", "year", "month", "inflation_rate_cleaned",
            "ma_3m", "ma_6m", "volatility_12m"
        ])
        
        # Modern period view  
        views["modern"] = lf.filter(
            pl.col("is_modern") == True
        ).select([
            "date", "year", "month", "quarter", "inflation_rate_cleaned",
            "ma_3m", "ma_6m", "ma_12m", "volatility_12m"
        ])
        
        # Annual summary view
        views["annual"] = self.aggregate_by_period(lf, "year")
        
        # Quarterly view (modern period only)
        views["quarterly"] = lf.filter(
            pl.col("is_modern") == True
        ).group_by(["year", "quarter"]).agg([
            pl.col("inflation_rate_cleaned").mean().alias("avg_inflation"),
            pl.col("inflation_rate_cleaned").sum().alias("cumulative_inflation"),
            pl.col("date").min().alias("quarter_start"),
            pl.col("date").max().alias("quarter_end"),
        ])
        
        # Crisis periods (high volatility)
        views["crisis"] = lf.filter(
            pl.col("volatility_12m") > pl.col("volatility_12m").quantile(0.9)
        ).select([
            "date", "year", "inflation_rate_cleaned", "volatility_12m", "period"
        ])
        
        return views