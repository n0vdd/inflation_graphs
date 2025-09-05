"""Monetary policy data integration for Brazil inflation analysis."""

import asyncio
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import httpx
import polars as pl
from rich.console import Console

from .config import Settings, get_settings
from .exceptions import APIError, DataFetchError
from .models import DataSource, InflationDataPoint, InflationDataset

console = Console()


class BrazilMonetaryPolicyFetcher:
    """
    Fetcher for Brazil's monetary policy data including Selic rate.
    
    Integrates with Central Bank of Brazil (BCB) APIs to retrieve:
    - Selic target rate
    - Selic effective rate
    - Real interest rate calculations
    - Monetary policy meeting dates and decisions
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize monetary policy fetcher."""
        self.settings = settings or get_settings()
        self.client: Optional[httpx.AsyncClient] = None
        
        # BCB Time Series API endpoints
        self.bcb_base_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
        self.selic_target_series = "432"  # Selic target rate
        self.selic_effective_series = "11"  # Selic effective rate
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
    
    async def fetch_selic_data(
        self,
        start_year: int,
        end_year: int,
        series_type: str = "target"
    ) -> InflationDataset:
        """
        Fetch Selic rate data from BCB API.
        
        Args:
            start_year: Starting year for data
            end_year: Ending year for data
            series_type: Type of series ('target' or 'effective')
            
        Returns:
            Selic rate dataset
            
        Raises:
            DataFetchError: If data fetching fails
        """
        if not self.client:
            raise DataFetchError("HTTP client not initialized")
        
        series_id = (self.selic_target_series if series_type == "target" 
                    else self.selic_effective_series)
        
        console.print(f"[blue]Fetching Selic {series_type} rate data from BCB API[/blue]")
        
        # BCB API format: /dados/serie/bcdata.sgs/{series_id}/dados
        # Parameters: dataInicial=dd/mm/yyyy&dataFinal=dd/mm/yyyy&formato=json
        url = f"{self.bcb_base_url}/{series_id}/dados"
        
        params = {
            "dataInicial": f"01/01/{start_year}",
            "dataFinal": f"31/12/{end_year}",
            "formato": "json"
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list):
                raise DataFetchError("Invalid BCB API response format")
            
            data_points = []
            
            for record in data:
                try:
                    # BCB format: {"data": "dd/mm/yyyy", "valor": "X.XX"}
                    date_str = record.get("data")
                    value_str = record.get("valor")
                    
                    if not date_str or not value_str:
                        continue
                    
                    # Parse Brazilian date format (dd/mm/yyyy)
                    day, month, year = map(int, date_str.split("/"))
                    observation_date = date(year, month, day)
                    
                    # Parse value (handle decimal comma)
                    value = float(value_str.replace(",", "."))
                    
                    data_points.append(InflationDataPoint(
                        date=observation_date,
                        value=value,
                        source=DataSource.IBGE_SIDRA,  # Using IBGE as closest match
                        metadata={
                            "indicator_type": "selic_rate",
                            "series_type": series_type,
                            "series_id": series_id,
                            "source_api": "bcb"
                        }
                    ))
                    
                except (ValueError, KeyError, TypeError) as e:
                    console.print(f"[yellow]Skipping invalid Selic data point: {e}[/yellow]")
                    continue
            
            if not data_points:
                raise DataFetchError("No valid Selic data points retrieved")
            
            console.print(f"[green]Successfully fetched {len(data_points)} Selic {series_type} rate points[/green]")
            
            return InflationDataset(
                data_points=data_points,
                source=DataSource.IBGE_SIDRA,
                metadata={
                    "dataset_type": "selic_rate",
                    "series_type": series_type,
                    "source_api": "bcb"
                }
            )
            
        except httpx.HTTPStatusError as e:
            raise DataFetchError(f"BCB API HTTP error: {e.response.status_code}")
        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            raise DataFetchError(f"Selic data fetch error: {e}")
    
    async def fetch_both_selic_series(
        self,
        start_year: int,
        end_year: int
    ) -> Dict[str, InflationDataset]:
        """
        Fetch both Selic target and effective rate series.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            Dictionary with both series datasets
        """
        console.print("[blue]Fetching both Selic target and effective rate series[/blue]")
        
        # Fetch both series concurrently
        tasks = {
            "target": self.fetch_selic_data(start_year, end_year, "target"),
            "effective": self.fetch_selic_data(start_year, end_year, "effective")
        }
        
        results = {}
        
        for series_name, coro in tasks.items():
            try:
                results[series_name] = await coro
                console.print(f"[green]Successfully fetched Selic {series_name} series[/green]")
            except Exception as e:
                console.print(f"[red]Failed to fetch Selic {series_name}: {e}[/red]")
                results[series_name] = None
        
        return results
    
    def calculate_real_interest_rates(
        self,
        selic_dataset: InflationDataset,
        inflation_dataset: InflationDataset
    ) -> pl.DataFrame:
        """
        Calculate real interest rates using Fisher equation.
        
        Real Rate ≈ Nominal Rate - Inflation Rate
        
        Args:
            selic_dataset: Selic rate dataset
            inflation_dataset: Inflation rate dataset
            
        Returns:
            DataFrame with real interest rate calculations
        """
        console.print("[blue]Calculating real interest rates[/blue]")
        
        # Convert to Polars DataFrames
        selic_df = pl.DataFrame([
            {
                "date": dp.date,
                "selic_rate": dp.value,
            }
            for dp in selic_dataset.data_points
        ]).with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("selic_rate").cast(pl.Float32)
        ])
        
        inflation_df = pl.DataFrame([
            {
                "date": dp.date,
                "inflation_rate": dp.value,
            }
            for dp in inflation_dataset.data_points
        ]).with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("inflation_rate").cast(pl.Float32)
        ])
        
        # Join datasets on date
        merged_df = selic_df.join(
            inflation_df,
            on="date",
            how="inner"
        ).with_columns([
            # Calculate real interest rate using Fisher equation approximation
            (pl.col("selic_rate") - pl.col("inflation_rate")).alias("real_interest_rate"),
            
            # Calculate exact Fisher equation: (1 + nominal) / (1 + inflation) - 1
            (
                (1 + pl.col("selic_rate") / 100) / 
                (1 + pl.col("inflation_rate") / 100) - 1
            ).mul(100).alias("real_interest_rate_exact"),
            
            # Add period classification
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            
            # Period flags
            (pl.col("date").dt.year() < 1994).alias("is_hyperinflation"),
            (pl.col("date").dt.year() >= 1994).alias("is_modern"),
            
            # Calculate spread between Selic and inflation
            pl.col("selic_rate").alias("nominal_rate"),
            pl.abs(pl.col("selic_rate") - pl.col("inflation_rate")).alias("rate_spread"),
        ]).sort("date")
        
        console.print(f"[green]Calculated real interest rates for {merged_df.height} observations[/green]")
        
        return merged_df
    
    def analyze_monetary_policy_effectiveness(
        self,
        real_rates_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Analyze monetary policy effectiveness metrics.
        
        Args:
            real_rates_df: DataFrame with real interest rate data
            
        Returns:
            DataFrame with policy effectiveness metrics
        """
        console.print("[blue]Analyzing monetary policy effectiveness[/blue]")
        
        # Calculate policy metrics by period
        policy_analysis = real_rates_df.group_by("is_modern").agg([
            # Real interest rate statistics
            pl.col("real_interest_rate").mean().alias("avg_real_rate"),
            pl.col("real_interest_rate").median().alias("median_real_rate"),
            pl.col("real_interest_rate").std().alias("real_rate_volatility"),
            pl.col("real_interest_rate").min().alias("min_real_rate"),
            pl.col("real_interest_rate").max().alias("max_real_rate"),
            
            # Nominal rate statistics
            pl.col("nominal_rate").mean().alias("avg_nominal_rate"),
            pl.col("nominal_rate").std().alias("nominal_rate_volatility"),
            
            # Inflation statistics
            pl.col("inflation_rate").mean().alias("avg_inflation"),
            pl.col("inflation_rate").std().alias("inflation_volatility"),
            
            # Policy effectiveness metrics
            pl.col("rate_spread").mean().alias("avg_rate_spread"),
            (pl.col("real_interest_rate") > 0).sum().alias("positive_real_rate_months"),
            (pl.col("real_interest_rate") < -5).sum().alias("deeply_negative_real_rate_months"),
            
            # Count observations
            pl.col("date").count().alias("total_observations"),
            pl.col("date").min().alias("period_start"),
            pl.col("date").max().alias("period_end"),
        ]).with_columns([
            # Add period labels
            pl.when(pl.col("is_modern"))
            .then(pl.lit("Modern Era (Post-1994)"))
            .otherwise(pl.lit("Hyperinflation Era (Pre-1994)"))
            .alias("period_name"),
            
            # Calculate percentage of positive real rates
            (pl.col("positive_real_rate_months") / pl.col("total_observations") * 100)
            .alias("positive_real_rate_pct"),
        ])
        
        console.print("[green]Completed monetary policy effectiveness analysis[/green]")
        
        return policy_analysis
    
    def identify_policy_regime_changes(
        self,
        real_rates_df: pl.DataFrame,
        window_size: int = 24
    ) -> pl.DataFrame:
        """
        Identify monetary policy regime changes using rolling statistics.
        
        Args:
            real_rates_df: DataFrame with real interest rate data
            window_size: Rolling window size in months
            
        Returns:
            DataFrame with regime change indicators
        """
        console.print(f"[blue]Identifying policy regime changes (window: {window_size} months)[/blue]")
        
        regime_df = real_rates_df.with_columns([
            # Rolling statistics for regime identification
            pl.col("real_interest_rate").rolling_mean(window_size).alias("real_rate_ma"),
            pl.col("real_interest_rate").rolling_std(window_size).alias("real_rate_rolling_std"),
            pl.col("nominal_rate").rolling_mean(window_size).alias("nominal_rate_ma"),
            pl.col("inflation_rate").rolling_mean(window_size).alias("inflation_ma"),
            
            # Calculate rolling correlation between nominal rates and inflation
            pl.col("nominal_rate").rolling_corr(pl.col("inflation_rate"), window_size)
            .alias("policy_inflation_correlation"),
            
            # Regime change indicators (significant shifts in rolling means)
            pl.col("real_interest_rate").rolling_mean(window_size)
            .diff().abs().alias("real_rate_regime_change"),
        ]).filter(
            # Remove initial NaN values from rolling calculations
            pl.col("real_rate_ma").is_not_null()
        ).with_columns([
            # Identify significant regime changes (above 90th percentile of changes)
            (pl.col("real_rate_regime_change") > 
             pl.col("real_rate_regime_change").quantile(0.9))
            .alias("significant_regime_change"),
            
            # Policy stance classification
            pl.when(pl.col("real_rate_ma") > 3)
            .then(pl.lit("Tight"))
            .when(pl.col("real_rate_ma") > 0)
            .then(pl.lit("Neutral"))
            .when(pl.col("real_rate_ma") > -3)
            .then(pl.lit("Accommodative"))
            .otherwise(pl.lit("Very Accommodative"))
            .alias("policy_stance"),
        ])
        
        # Count regime changes
        total_changes = regime_df.filter(
            pl.col("significant_regime_change") == True
        ).height
        
        console.print(f"[green]Identified {total_changes} significant policy regime changes[/green]")
        
        return regime_df
    
    def create_policy_summary_report(
        self,
        real_rates_df: pl.DataFrame,
        policy_analysis_df: pl.DataFrame,
        regime_df: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Create comprehensive monetary policy summary report.
        
        Args:
            real_rates_df: Real interest rate data
            policy_analysis_df: Policy effectiveness analysis
            regime_df: Regime change analysis
            
        Returns:
            Dictionary with comprehensive policy report
        """
        console.print("[blue]Creating monetary policy summary report[/blue]")
        
        # Extract key metrics
        modern_era = policy_analysis_df.filter(
            pl.col("period_name").str.contains("Modern")
        ).to_dicts()
        
        hyperinflation_era = policy_analysis_df.filter(
            pl.col("period_name").str.contains("Hyperinflation")
        ).to_dicts()
        
        # Recent policy stance distribution
        recent_stances = regime_df.filter(
            pl.col("date") >= pl.col("date").max() - pl.duration(years=5)
        ).group_by("policy_stance").agg([
            pl.col("date").count().alias("months")
        ]).with_columns([
            (pl.col("months") / pl.col("months").sum() * 100).alias("percentage")
        ])
        
        report = {
            "summary": {
                "total_observations": int(real_rates_df.height),
                "date_range": {
                    "start": str(real_rates_df.select("date").min().item()),
                    "end": str(real_rates_df.select("date").max().item())
                },
                "analysis_generated": datetime.now().isoformat()
            },
            "period_comparison": {
                "modern_era": modern_era[0] if modern_era else {},
                "hyperinflation_era": hyperinflation_era[0] if hyperinflation_era else {}
            },
            "recent_policy_stance": recent_stances.to_dicts(),
            "key_insights": self._generate_policy_insights(
                real_rates_df, policy_analysis_df, regime_df
            )
        }
        
        console.print("[green]Monetary policy summary report completed[/green]")
        
        return report
    
    def _generate_policy_insights(
        self,
        real_rates_df: pl.DataFrame,
        policy_analysis_df: pl.DataFrame,
        regime_df: pl.DataFrame
    ) -> List[str]:
        """Generate key policy insights from the analysis."""
        insights = []
        
        # Average real rate comparison
        modern_avg = policy_analysis_df.filter(
            pl.col("period_name").str.contains("Modern")
        ).select("avg_real_rate").item()
        
        hyperinflation_avg = policy_analysis_df.filter(
            pl.col("period_name").str.contains("Hyperinflation")
        ).select("avg_real_rate").item()
        
        if modern_avg and hyperinflation_avg:
            if modern_avg > hyperinflation_avg:
                insights.append(
                    f"Real interest rates averaged {modern_avg:.1f}% in the modern era "
                    f"vs {hyperinflation_avg:.1f}% during hyperinflation, showing improved policy credibility"
                )
            else:
                insights.append(
                    f"Real interest rates were negative on average during hyperinflation "
                    f"({hyperinflation_avg:.1f}%) but positive in modern era ({modern_avg:.1f}%)"
                )
        
        # Volatility comparison
        modern_vol = policy_analysis_df.filter(
            pl.col("period_name").str.contains("Modern")
        ).select("real_rate_volatility").item()
        
        if modern_vol:
            if modern_vol < 5:
                insights.append(
                    f"Real rate volatility in modern era is low ({modern_vol:.1f}%), "
                    "indicating stable monetary policy"
                )
            else:
                insights.append(
                    f"Real rate volatility remains high ({modern_vol:.1f}%) even in modern era"
                )
        
        # Recent trend
        recent_avg = real_rates_df.filter(
            pl.col("date") >= pl.col("date").max() - pl.duration(years=2)
        ).select("real_interest_rate").mean().item()
        
        if recent_avg:
            if recent_avg > 3:
                insights.append(f"Recent real rates are tight ({recent_avg:.1f}%), potentially constraining growth")
            elif recent_avg < 0:
                insights.append(f"Recent real rates are negative ({recent_avg:.1f}%), providing monetary stimulus")
            else:
                insights.append(f"Recent real rates are neutral ({recent_avg:.1f}%)")
        
        return insights


# Alias for backward compatibility with enhanced visualizer
MonetaryPolicyAnalyzer = BrazilMonetaryPolicyFetcher