"""Data models for Brazil inflation data processing."""

from datetime import date as Date, datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

import polars as pl
from pydantic import BaseModel, Field, validator


class DataSource(str, Enum):
    """Available data sources for inflation data."""
    
    FRED = "fred"
    WORLD_BANK = "world_bank"
    IBGE_SIDRA = "ibge_sidra"


class InflationPeriod(str, Enum):
    """Economic periods for Brazil inflation analysis."""
    
    HYPERINFLATION = "hyperinflation"  # Pre-1994
    MODERN = "modern"  # Post-1994
    TRANSITION = "transition"  # 1994-1999


class InflationDataPoint(BaseModel):
    """Single inflation data point."""
    
    date: Date = Field(description="Date of the observation")
    value: float = Field(description="Inflation rate as percentage")
    source: DataSource = Field(description="Data source")
    period: Optional[InflationPeriod] = Field(
        default=None, 
        description="Economic period classification"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the data point"
    )
    
    @validator("value")
    def validate_inflation_value(cls, v: float) -> float:
        """Validate inflation rate values."""
        if v < -100:  # Deflation beyond -100% is unrealistic
            raise ValueError(f"Inflation rate {v}% seems unrealistic (too negative)")
        if v > 10000:  # Hyperinflation beyond 10,000% is extreme
            raise ValueError(f"Inflation rate {v}% seems unrealistic (too high)")
        return v
    
    @validator("period", always=True)
    def classify_period(cls, v: Optional[InflationPeriod], values: Dict[str, Any]) -> InflationPeriod:
        """Auto-classify economic period based on date."""
        if v is not None:
            return v
        
        date_val = values.get("date")
        if date_val is None:
            return InflationPeriod.MODERN
        
        year = date_val.year
        if year < 1994:
            return InflationPeriod.HYPERINFLATION
        elif year <= 1999:
            return InflationPeriod.TRANSITION
        else:
            return InflationPeriod.MODERN


class InflationDataset(BaseModel):
    """Complete inflation dataset with metadata."""
    
    data_points: List[InflationDataPoint] = Field(description="List of inflation data points")
    source: DataSource = Field(description="Primary data source")
    retrieved_at: datetime = Field(
        default_factory=datetime.now, 
        description="When the data was retrieved"
    )
    date_range: Optional[Tuple[Date, Date]] = Field(default=None, description="Date range of the dataset")
    total_observations: Optional[int] = Field(default=None, description="Total number of observations")
    hyperinflation_observations: int = Field(
        default=0, 
        description="Number of hyperinflation period observations"
    )
    modern_observations: int = Field(
        default=0,
        description="Number of modern period observations" 
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Dataset-level metadata"
    )
    
    @validator("data_points")
    def validate_data_points(cls, v: List[InflationDataPoint]) -> List[InflationDataPoint]:
        """Validate data points list is not empty and sorted."""
        if not v:
            raise ValueError("Dataset cannot be empty")
        
        # Sort by date
        v.sort(key=lambda x: x.date)
        return v
    
    @validator("date_range", always=True)
    def calculate_date_range(cls, v: Optional[Tuple[Date, Date]], values: Dict[str, Any]) -> Tuple[Date, Date]:
        """Calculate date range from data points."""
        data_points = values.get("data_points", [])
        if not data_points:
            return (Date.today(), Date.today())
        
        dates = [dp.date for dp in data_points]
        return (min(dates), max(dates))
    
    @validator("total_observations", always=True)
    def calculate_total_observations(cls, v: Optional[int], values: Dict[str, Any]) -> int:
        """Calculate total observations."""
        data_points = values.get("data_points", [])
        return len(data_points)
    
    @validator("hyperinflation_observations", always=True)
    def calculate_hyperinflation_observations(cls, v: Optional[int], values: Dict[str, Any]) -> int:
        """Calculate hyperinflation period observations."""
        data_points = values.get("data_points", [])
        return sum(1 for dp in data_points if dp.period == InflationPeriod.HYPERINFLATION)
    
    @validator("modern_observations", always=True) 
    def calculate_modern_observations(cls, v: Optional[int], values: Dict[str, Any]) -> int:
        """Calculate modern period observations."""
        data_points = values.get("data_points", [])
        return sum(1 for dp in data_points if dp.period == InflationPeriod.MODERN)
    
    def to_polars(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        data = [
            {
                "date": dp.date,
                "inflation_rate": dp.value,
                "source": dp.source.value,
                "period": dp.period.value,
                "year": dp.date.year,
                "month": dp.date.month,
            }
            for dp in self.data_points
        ]
        
        return pl.DataFrame(data).with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("inflation_rate").cast(pl.Float32),
            pl.col("year").cast(pl.Int16),
            pl.col("month").cast(pl.Int8),
            # Add cleaned version (for now, same as inflation_rate)
            pl.col("inflation_rate").alias("inflation_rate_cleaned"),
        ])
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the dataset."""
        df = self.to_polars()
        
        stats = df.select([
            pl.col("inflation_rate").mean().alias("mean_inflation"),
            pl.col("inflation_rate").median().alias("median_inflation"),
            pl.col("inflation_rate").std().alias("std_inflation"),
            pl.col("inflation_rate").min().alias("min_inflation"),
            pl.col("inflation_rate").max().alias("max_inflation"),
        ]).to_dict(as_series=False)
        
        # Convert to simple dict
        return {k: v[0] if v else None for k, v in stats.items()}


class DataQualityReport(BaseModel):
    """Data quality assessment report."""
    
    source: DataSource = Field(description="Data source being assessed")
    total_points: int = Field(description="Total data points")
    missing_points: int = Field(description="Missing data points")
    outliers: int = Field(description="Potential outlier points")
    hyperinflation_anomalies: int = Field(
        default=0,
        description="Anomalous values in hyperinflation period"
    )
    data_completeness: float = Field(description="Percentage of data completeness")
    quality_score: float = Field(description="Overall quality score (0-1)")
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    recommendations: List[str] = Field(
        default_factory=list, 
        description="Recommendations for data usage"
    )
    
    @validator("data_completeness", always=True)
    def calculate_completeness(cls, v: Optional[float], values: Dict[str, Any]) -> float:
        """Calculate data completeness percentage."""
        total = values.get("total_points", 0)
        missing = values.get("missing_points", 0)
        if total == 0:
            return 0.0
        return ((total - missing) / total) * 100.0
    
    @validator("quality_score", always=True)
    def calculate_quality_score(cls, v: Optional[float], values: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        completeness = values.get("data_completeness", 0.0) / 100.0
        outliers = values.get("outliers", 0)
        total = values.get("total_points", 1)
        anomalies = values.get("hyperinflation_anomalies", 0)
        
        # Base score from completeness
        score = completeness
        
        # Penalize for outliers and anomalies
        outlier_penalty = (outliers / total) * 0.3
        anomaly_penalty = (anomalies / total) * 0.2
        
        score = max(0.0, score - outlier_penalty - anomaly_penalty)
        return min(1.0, score)