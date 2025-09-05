"""Comparative analysis framework for Brazil inflation vs emerging markets."""

import asyncio
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import httpx
import polars as pl
import numpy as np
from rich.console import Console

from .config import Settings, get_settings
from .exceptions import APIError, DataFetchError
from .models import DataSource, InflationDataPoint, InflationDataset

console = Console()


class EmergingMarketsComparator:
    """
    Comparative analysis framework for Brazil vs other emerging markets.
    
    Analyzes Brazil's inflation performance against peer countries:
    - Mexico (MXN)
    - Argentina (ARG) 
    - Chile (CHL)
    - Colombia (COL)
    - Turkey (TUR)
    - South Africa (ZAF)
    - India (IND)
    - Indonesia (IDN)
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize comparative analysis framework."""
        self.settings = settings or get_settings()
        self.client: Optional[httpx.AsyncClient] = None
        
        # Peer countries for comparison
        self.peer_countries = {
            "MEX": "Mexico",
            "ARG": "Argentina", 
            "CHL": "Chile",
            "COL": "Colombia",
            "TUR": "Turkey",
            "ZAF": "South Africa",
            "IND": "India",
            "IDN": "Indonesia"
        }
        
        # World Bank API settings
        self.world_bank_base_url = "https://api.worldbank.org/v2"
        self.inflation_indicator = "FP.CPI.TOTL.ZG"  # CPI inflation
        
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
    
    async def fetch_peer_country_data(
        self,
        country_codes: Optional[List[str]] = None,
        start_year: int = 1980,
        end_year: int = 2024
    ) -> Dict[str, Optional[InflationDataset]]:
        """
        Fetch inflation data for peer emerging markets.
        
        Args:
            country_codes: List of ISO country codes (defaults to all peers)
            start_year: Starting year for data
            end_year: Ending year for data
            
        Returns:
            Dictionary mapping country codes to their inflation datasets
        """
        country_codes = country_codes or list(self.peer_countries.keys())
        
        console.print(f"[blue]Fetching inflation data for {len(country_codes)} peer countries[/blue]")
        
        # Create concurrent tasks for all countries
        tasks = {
            code: self._fetch_country_inflation(code, start_year, end_year)
            for code in country_codes
        }
        
        results = {}
        
        for country_code, coro in tasks.items():
            try:
                results[country_code] = await coro
                country_name = self.peer_countries.get(country_code, country_code)
                console.print(f"[green]Successfully fetched data for {country_name}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to fetch data for {country_code}: {e}[/red]")
                results[country_code] = None
        
        return results
    
    async def _fetch_country_inflation(
        self,
        country_code: str,
        start_year: int,
        end_year: int
    ) -> InflationDataset:
        """
        Fetch inflation data for a specific country from World Bank.
        
        Args:
            country_code: ISO country code
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            Country's inflation dataset
        """
        if not self.client:
            raise DataFetchError("HTTP client not initialized")
        
        url = f"{self.world_bank_base_url}/country/{country_code}/indicator/{self.inflation_indicator}"
        
        params = {
            "format": "json",
            "date": f"{start_year}:{end_year}",
            "per_page": "1000"
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list) or len(data) < 2:
                raise DataFetchError(f"Invalid World Bank response for {country_code}")
            
            raw_data = data[1]
            if not raw_data:
                raise DataFetchError(f"No data returned for {country_code}")
            
            data_points = []
            
            for item in raw_data:
                if item.get("value") is not None:
                    try:
                        year = int(item["date"])
                        # Use December as representative date
                        observation_date = date(year, 12, 31)
                        value = float(item["value"])
                        
                        data_points.append(InflationDataPoint(
                            date=observation_date,
                            value=value,
                            source=DataSource.WORLD_BANK,
                            metadata={
                                "country_code": country_code,
                                "country_name": self.peer_countries.get(country_code, country_code),
                                "indicator": self.inflation_indicator
                            }
                        ))
                        
                    except (ValueError, KeyError):
                        continue
            
            if not data_points:
                raise DataFetchError(f"No valid data points for {country_code}")
            
            return InflationDataset(
                data_points=data_points,
                source=DataSource.WORLD_BANK,
                metadata={
                    "country_code": country_code,
                    "country_name": self.peer_countries.get(country_code, country_code)
                }
            )
            
        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            raise DataFetchError(f"Error fetching data for {country_code}: {e}")
    
    def create_comparative_dataset(
        self,
        brazil_data: InflationDataset,
        peer_data: Dict[str, Optional[InflationDataset]]
    ) -> pl.DataFrame:
        """
        Create unified comparative dataset.
        
        Args:
            brazil_data: Brazil's inflation dataset
            peer_data: Dictionary of peer country datasets
            
        Returns:
            Comparative DataFrame with all countries
        """
        console.print("[blue]Creating unified comparative dataset[/blue]")
        
        # Start with Brazil data
        all_data = []
        
        # Add Brazil data
        for dp in brazil_data.data_points:
            all_data.append({
                "date": dp.date,
                "inflation_rate": dp.value,
                "country_code": "BRA",
                "country_name": "Brazil",
                "year": dp.date.year,
                "is_brazil": True,
                "period": dp.period.value if dp.period else "modern"
            })
        
        # Add peer country data
        for country_code, dataset in peer_data.items():
            if dataset is None:
                continue
                
            for dp in dataset.data_points:
                all_data.append({
                    "date": dp.date,
                    "inflation_rate": dp.value,
                    "country_code": country_code,
                    "country_name": self.peer_countries.get(country_code, country_code),
                    "year": dp.date.year,
                    "is_brazil": False,
                    "period": "modern" if dp.date.year >= 1994 else "hyperinflation"
                })
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(all_data).with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("inflation_rate").cast(pl.Float32),
            pl.col("year").cast(pl.Int16),
            pl.col("is_brazil").cast(pl.Boolean),
            
            # Add regional classification
            pl.when(pl.col("country_code").is_in(["MEX", "ARG", "CHL", "COL"]))
            .then(pl.lit("Latin America"))
            .when(pl.col("country_code").is_in(["TUR"]))
            .then(pl.lit("Europe/Middle East"))
            .when(pl.col("country_code").is_in(["ZAF"]))
            .then(pl.lit("Africa"))
            .when(pl.col("country_code").is_in(["IND", "IDN"]))
            .then(pl.lit("Asia"))
            .otherwise(pl.lit("Latin America"))  # Brazil
            .alias("region"),
        ]).sort(["country_code", "date"])
        
        console.print(f"[green]Created comparative dataset with {df.height} observations across {df.select('country_code').n_unique()} countries[/green]")
        
        return df
    
    def analyze_inflation_performance(
        self,
        comparative_df: pl.DataFrame,
        analysis_periods: Optional[List[Tuple[int, int]]] = None
    ) -> pl.DataFrame:
        """
        Analyze relative inflation performance across countries and periods.
        
        Args:
            comparative_df: Unified comparative dataset
            analysis_periods: List of (start_year, end_year) tuples for analysis
            
        Returns:
            Performance analysis DataFrame
        """
        if analysis_periods is None:
            analysis_periods = [
                (1980, 1993),  # Pre-Plano Real
                (1994, 1999),  # Transition period
                (2000, 2009),  # 2000s decade
                (2010, 2019),  # 2010s decade
                (2020, 2024),  # Post-COVID era
            ]
        
        console.print(f"[blue]Analyzing inflation performance across {len(analysis_periods)} periods[/blue]")
        
        performance_results = []
        
        for start_year, end_year in analysis_periods:
            period_df = comparative_df.filter(
                (pl.col("year") >= start_year) & (pl.col("year") <= end_year)
            )
            
            if period_df.height == 0:
                continue
            
            # Calculate statistics by country for this period
            period_stats = period_df.group_by(["country_code", "country_name", "region"]).agg([
                pl.col("inflation_rate").mean().alias("avg_inflation"),
                pl.col("inflation_rate").median().alias("median_inflation"),
                pl.col("inflation_rate").std().alias("inflation_volatility"),
                pl.col("inflation_rate").min().alias("min_inflation"),
                pl.col("inflation_rate").max().alias("max_inflation"),
                pl.col("inflation_rate").count().alias("observations"),
                (pl.col("inflation_rate") > 10).sum().alias("high_inflation_years"),
                (pl.col("inflation_rate") < 0).sum().alias("deflation_years"),
            ]).with_columns([
                pl.lit(f"{start_year}-{end_year}").alias("period"),
                pl.lit(start_year).alias("period_start"),
                pl.lit(end_year).alias("period_end"),
                
                # Calculate relative performance metrics
                (pl.col("high_inflation_years") / pl.col("observations") * 100)
                .alias("high_inflation_pct"),
                
                (pl.col("deflation_years") / pl.col("observations") * 100)
                .alias("deflation_pct"),
            ])
            
            performance_results.append(period_stats)
        
        # Combine all periods
        performance_df = pl.concat(performance_results).sort(["period_start", "country_code"])
        
        # Add rankings within each period
        performance_df = performance_df.with_columns([
            pl.col("avg_inflation").rank().over("period").alias("avg_inflation_rank"),
            pl.col("inflation_volatility").rank().over("period").alias("volatility_rank"),
        ])
        
        console.print("[green]Completed inflation performance analysis[/green]")
        
        return performance_df
    
    def identify_inflation_convergence(
        self,
        comparative_df: pl.DataFrame,
        reference_country: str = "BRA",
        window_years: int = 5
    ) -> pl.DataFrame:
        """
        Analyze inflation convergence patterns relative to reference country.
        
        Args:
            comparative_df: Comparative dataset
            reference_country: Reference country code (default Brazil)
            window_years: Rolling window for convergence analysis
            
        Returns:
            Convergence analysis DataFrame
        """
        console.print(f"[blue]Analyzing inflation convergence relative to {reference_country}[/blue]")
        
        # Get reference country data
        reference_df = comparative_df.filter(
            pl.col("country_code") == reference_country
        ).select(["date", "inflation_rate"]).rename({"inflation_rate": "reference_inflation"})
        
        # Join all countries with reference data
        convergence_df = comparative_df.join(
            reference_df,
            on="date",
            how="inner"
        ).with_columns([
            # Calculate inflation differential
            (pl.col("inflation_rate") - pl.col("reference_inflation"))
            .alias("inflation_differential"),
            
            # Calculate absolute differential
            pl.abs(pl.col("inflation_rate") - pl.col("reference_inflation"))
            .alias("abs_inflation_differential"),
        ]).filter(
            # Exclude reference country from analysis
            pl.col("country_code") != reference_country
        )
        
        # Calculate rolling convergence metrics
        convergence_metrics = convergence_df.sort(["country_code", "date"]).with_columns([
            # Rolling average of absolute differential (convergence indicator)
            pl.col("abs_inflation_differential")
            .rolling_mean(window_years)
            .over("country_code")
            .alias("convergence_trend"),
            
            # Rolling correlation with reference country
            pl.col("inflation_rate")
            .rolling_corr(pl.col("reference_inflation"), window_years)
            .over("country_code")
            .alias("inflation_correlation"),
            
            # Convergence classification
            pl.when(pl.col("abs_inflation_differential") < 2)
            .then(pl.lit("Converged"))
            .when(pl.col("abs_inflation_differential") < 5)
            .then(pl.lit("Partially Converged"))
            .otherwise(pl.lit("Divergent"))
            .alias("convergence_status"),
        ])
        
        console.print("[green]Completed inflation convergence analysis[/green]")
        
        return convergence_metrics
    
    def create_peer_rankings(
        self,
        performance_df: pl.DataFrame,
        ranking_criteria: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Create peer country rankings based on inflation performance.
        
        Args:
            performance_df: Performance analysis results
            ranking_criteria: List of criteria for ranking
            
        Returns:
            Peer rankings DataFrame
        """
        if ranking_criteria is None:
            ranking_criteria = ["avg_inflation", "inflation_volatility", "high_inflation_pct"]
        
        console.print(f"[blue]Creating peer rankings based on {len(ranking_criteria)} criteria[/blue]")
        
        # Calculate composite scores for each period
        rankings_df = performance_df.with_columns([
            # Normalize metrics for scoring (lower is better)
            (1 / (pl.col("avg_inflation").abs() + 0.1)).alias("inflation_score"),
            (1 / (pl.col("inflation_volatility") + 0.1)).alias("stability_score"),
            (1 / (pl.col("high_inflation_pct") + 0.1)).alias("consistency_score"),
        ]).with_columns([
            # Calculate composite performance score
            (
                pl.col("inflation_score") * 0.4 +
                pl.col("stability_score") * 0.3 + 
                pl.col("consistency_score") * 0.3
            ).alias("composite_score")
        ])
        
        # Add rankings within each period
        final_rankings = rankings_df.with_columns([
            pl.col("composite_score").rank(descending=True).over("period").alias("overall_rank"),
            pl.col("avg_inflation").rank().over("period").alias("inflation_rank"),
            pl.col("inflation_volatility").rank().over("period").alias("stability_rank"),
        ]).sort(["period_start", "overall_rank"])
        
        console.print("[green]Completed peer rankings calculation[/green]")
        
        return final_rankings
    
    def generate_comparative_insights(
        self,
        comparative_df: pl.DataFrame,
        performance_df: pl.DataFrame,
        convergence_df: pl.DataFrame,
        rankings_df: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparative analysis insights.
        
        Args:
            comparative_df: Unified comparative dataset
            performance_df: Performance analysis results
            convergence_df: Convergence analysis results
            rankings_df: Peer rankings
            
        Returns:
            Dictionary with comparative insights
        """
        console.print("[blue]Generating comparative analysis insights[/blue]")
        
        # Brazil's performance summary
        brazil_performance = performance_df.filter(
            pl.col("country_code") == "BRA"
        ).sort("period_start")
        
        # Recent period analysis (2020-2024)
        recent_rankings = rankings_df.filter(
            pl.col("period") == "2020-2024"
        ).sort("overall_rank")
        
        # Convergence summary
        convergence_summary = convergence_df.group_by("country_code").agg([
            pl.col("convergence_trend").mean().alias("avg_convergence"),
            pl.col("inflation_correlation").mean().alias("avg_correlation"),
            pl.col("convergence_status").mode().first().alias("typical_status")
        ]).sort("avg_convergence")
        
        # Generate insights
        insights = {
            "summary": {
                "countries_analyzed": int(comparative_df.select("country_code").n_unique()),
                "total_observations": int(comparative_df.height),
                "analysis_periods": len(performance_df.select("period").unique()),
                "date_range": {
                    "start": str(comparative_df.select("date").min().item()),
                    "end": str(comparative_df.select("date").max().item())
                }
            },
            "brazil_highlights": self._extract_brazil_insights(brazil_performance),
            "peer_comparison": self._extract_peer_insights(recent_rankings),
            "convergence_analysis": convergence_summary.to_dicts(),
            "regional_patterns": self._analyze_regional_patterns(comparative_df),
            "key_findings": self._generate_key_findings(
                comparative_df, performance_df, convergence_df, rankings_df
            )
        }
        
        console.print("[green]Generated comprehensive comparative insights[/green]")
        
        return insights
    
    def _extract_brazil_insights(self, brazil_performance: pl.DataFrame) -> List[Dict[str, Any]]:
        """Extract key insights about Brazil's performance."""
        insights = []
        
        for row in brazil_performance.to_dicts():
            period = row["period"]
            avg_inflation = row["avg_inflation"]
            volatility = row["inflation_volatility"]
            rank = row.get("overall_rank", "N/A")
            
            insight = {
                "period": period,
                "avg_inflation": round(avg_inflation, 2) if avg_inflation else None,
                "volatility": round(volatility, 2) if volatility else None,
                "rank": int(rank) if isinstance(rank, (int, float)) and not np.isnan(rank) else "N/A",
                "assessment": self._assess_performance(avg_inflation, volatility)
            }
            insights.append(insight)
        
        return insights
    
    def _extract_peer_insights(self, recent_rankings: pl.DataFrame) -> List[Dict[str, Any]]:
        """Extract insights about peer country performance."""
        return [
            {
                "country": row["country_name"],
                "country_code": row["country_code"],
                "rank": int(row["overall_rank"]),
                "avg_inflation": round(row["avg_inflation"], 2),
                "volatility": round(row["inflation_volatility"], 2),
                "region": row["region"]
            }
            for row in recent_rankings.to_dicts()
        ]
    
    def _analyze_regional_patterns(self, comparative_df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze patterns by geographic region."""
        regional_stats = comparative_df.filter(
            pl.col("year") >= 2020
        ).group_by("region").agg([
            pl.col("inflation_rate").mean().alias("avg_inflation"),
            pl.col("inflation_rate").std().alias("volatility"),
            pl.col("country_code").n_unique().alias("countries")
        ]).sort("avg_inflation")
        
        return {
            "regional_rankings": regional_stats.to_dicts(),
            "best_performing_region": regional_stats.row(0)[0] if regional_stats.height > 0 else None
        }
    
    def _generate_key_findings(
        self,
        comparative_df: pl.DataFrame,
        performance_df: pl.DataFrame,
        convergence_df: pl.DataFrame,
        rankings_df: pl.DataFrame
    ) -> List[str]:
        """Generate key comparative findings."""
        findings = []
        
        # Brazil's transformation
        pre_1994_avg = comparative_df.filter(
            (pl.col("country_code") == "BRA") & (pl.col("year") < 1994)
        ).select("inflation_rate").mean().item()
        
        post_1994_avg = comparative_df.filter(
            (pl.col("country_code") == "BRA") & (pl.col("year") >= 1994)
        ).select("inflation_rate").mean().item()
        
        if pre_1994_avg and post_1994_avg:
            improvement = pre_1994_avg - post_1994_avg
            findings.append(
                f"Brazil's inflation averaged {pre_1994_avg:.1f}% before 1994 vs {post_1994_avg:.1f}% after, "
                f"an improvement of {improvement:.1f} percentage points"
            )
        
        # Recent performance
        brazil_recent_rank = rankings_df.filter(
            (pl.col("country_code") == "BRA") & (pl.col("period") == "2020-2024")
        ).select("overall_rank").item()
        
        if brazil_recent_rank:
            total_countries = rankings_df.filter(
                pl.col("period") == "2020-2024"
            ).height
            
            if brazil_recent_rank <= total_countries // 3:
                performance = "top-tier"
            elif brazil_recent_rank <= 2 * total_countries // 3:
                performance = "middle-tier"
            else:
                performance = "bottom-tier"
            
            findings.append(
                f"Brazil ranks #{int(brazil_recent_rank)} out of {total_countries} peer countries "
                f"in recent performance ({performance})"
            )
        
        # Volatility comparison
        brazil_volatility = performance_df.filter(
            (pl.col("country_code") == "BRA") & (pl.col("period") == "2020-2024")
        ).select("inflation_volatility").item()
        
        peer_avg_volatility = performance_df.filter(
            (pl.col("country_code") != "BRA") & (pl.col("period") == "2020-2024")
        ).select("inflation_volatility").mean().item()
        
        if brazil_volatility and peer_avg_volatility:
            if brazil_volatility < peer_avg_volatility:
                findings.append(
                    f"Brazil's inflation volatility ({brazil_volatility:.1f}%) is below peer average "
                    f"({peer_avg_volatility:.1f}%), indicating relatively stable monetary policy"
                )
            else:
                findings.append(
                    f"Brazil's inflation volatility ({brazil_volatility:.1f}%) exceeds peer average "
                    f"({peer_avg_volatility:.1f}%), suggesting room for improvement in policy stability"
                )
        
        return findings
    
    def _assess_performance(self, avg_inflation: float, volatility: float) -> str:
        """Assess inflation performance based on level and volatility."""
        if avg_inflation is None or volatility is None:
            return "Insufficient data"
        
        if avg_inflation < 3 and volatility < 2:
            return "Excellent"
        elif avg_inflation < 5 and volatility < 3:
            return "Good"
        elif avg_inflation < 10 and volatility < 5:
            return "Acceptable"
        else:
            return "Poor"


class ComparativeInflationAnalysis:
    """
    Comprehensive analysis framework for Brazil's inflation data with dual-period support.
    
    This class specializes in analyzing Brazil's dramatic economic transformation,
    particularly the transition from hyperinflation (pre-1994) to modern inflation
    targeting (post-1994). It provides methods for comparative analysis, trend
    identification, and structural break detection.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize comparative inflation analysis framework."""
        self.settings = settings or get_settings()
        self.emerging_markets = EmergingMarketsComparator(settings)
        
        # Key historical breakpoints for Brazil
        self.plano_real_date = date(1994, 7, 1)  # Plano Real implementation
        self.inflation_targeting_date = date(1999, 6, 21)  # Inflation targeting adoption
        
        # Analysis periods
        self.hyperinflation_period = (1980, 1993)
        self.transition_period = (1994, 1999)
        self.modern_period = (2000, 2024)
        
    def classify_inflation_regime(self, inflation_rate: float, date_value: date) -> str:
        """
        Classify inflation observation into regime categories.
        
        Args:
            inflation_rate: Inflation rate value
            date_value: Observation date
            
        Returns:
            Regime classification string
        """
        year = date_value.year
        
        if year < 1994:
            if inflation_rate > 100:
                return "hyperinflation"
            elif inflation_rate > 50:
                return "high_inflation"
            else:
                return "pre_plano_real"
        elif year < 1999:
            return "transition"
        else:
            if inflation_rate > 10:
                return "elevated_modern"
            elif inflation_rate < 0:
                return "deflationary"
            else:
                return "modern_stable"
    
    def calculate_structural_breaks(self, inflation_data: InflationDataset) -> Dict[str, Any]:
        """
        Identify structural breaks in Brazil's inflation series.
        
        Args:
            inflation_data: Brazil's inflation dataset
            
        Returns:
            Dictionary with structural break analysis results
        """
        console.print("[blue]Analyzing structural breaks in inflation data[/blue]")
        
        # Convert to DataFrame for analysis
        data_points = []
        for dp in inflation_data.data_points:
            data_points.append({
                "date": dp.date,
                "inflation_rate": dp.value,
                "year": dp.date.year,
                "month": dp.date.month
            })
        
        df = pl.DataFrame(data_points).sort("date")
        
        # Calculate regime statistics
        pre_1994 = df.filter(pl.col("year") < 1994)
        post_1994 = df.filter(pl.col("year") >= 1994)
        post_1999 = df.filter(pl.col("year") >= 1999)
        
        structural_analysis = {
            "plano_real_impact": {
                "pre_1994_stats": {
                    "mean": float(pre_1994.select("inflation_rate").mean().item()) if pre_1994.height > 0 else None,
                    "std": float(pre_1994.select("inflation_rate").std().item()) if pre_1994.height > 0 else None,
                    "max": float(pre_1994.select("inflation_rate").max().item()) if pre_1994.height > 0 else None,
                    "observations": pre_1994.height
                },
                "post_1994_stats": {
                    "mean": float(post_1994.select("inflation_rate").mean().item()) if post_1994.height > 0 else None,
                    "std": float(post_1994.select("inflation_rate").std().item()) if post_1994.height > 0 else None,
                    "max": float(post_1994.select("inflation_rate").max().item()) if post_1994.height > 0 else None,
                    "observations": post_1994.height
                }
            },
            "inflation_targeting_impact": {
                "pre_targeting_stats": {
                    "mean": float(df.filter((pl.col("year") >= 1994) & (pl.col("year") < 1999)).select("inflation_rate").mean().item()) if df.filter((pl.col("year") >= 1994) & (pl.col("year") < 1999)).height > 0 else None,
                    "std": float(df.filter((pl.col("year") >= 1994) & (pl.col("year") < 1999)).select("inflation_rate").std().item()) if df.filter((pl.col("year") >= 1994) & (pl.col("year") < 1999)).height > 0 else None
                },
                "post_targeting_stats": {
                    "mean": float(post_1999.select("inflation_rate").mean().item()) if post_1999.height > 0 else None,
                    "std": float(post_1999.select("inflation_rate").std().item()) if post_1999.height > 0 else None,
                    "observations": post_1999.height
                }
            },
            "volatility_regimes": self._analyze_volatility_regimes(df),
            "trend_breaks": self._identify_trend_breaks(df)
        }
        
        console.print("[green]Completed structural break analysis[/green]")
        
        return structural_analysis
    
    def analyze_dual_period_performance(self, inflation_data: InflationDataset) -> Dict[str, Any]:
        """
        Comprehensive dual-period analysis comparing hyperinflation vs modern era.
        
        Args:
            inflation_data: Brazil's inflation dataset
            
        Returns:
            Dual-period analysis results
        """
        console.print("[blue]Performing dual-period analysis[/blue]")
        
        # Convert to DataFrame
        data_points = []
        for dp in inflation_data.data_points:
            regime = self.classify_inflation_regime(dp.value, dp.date)
            data_points.append({
                "date": dp.date,
                "inflation_rate": dp.value,
                "year": dp.date.year,
                "regime": regime,
                "log_inflation": np.log(max(dp.value, 0.1)) if dp.value > 0 else None  # Log transform for hyperinflation
            })
        
        df = pl.DataFrame(data_points).sort("date")
        
        # Dual-period comparison
        dual_analysis = {
            "period_comparison": self._compare_inflation_periods(df),
            "regime_transitions": self._analyze_regime_transitions(df),
            "policy_effectiveness": self._assess_policy_effectiveness(df),
            "international_context": self._place_in_international_context(df),
            "volatility_evolution": self._track_volatility_evolution(df)
        }
        
        console.print("[green]Completed dual-period analysis[/green]")
        
        return dual_analysis
    
    def generate_comparative_metrics(
        self, 
        brazil_data: InflationDataset,
        peer_countries: Optional[Dict[str, InflationDataset]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparative metrics for Brazil vs peers.
        
        Args:
            brazil_data: Brazil's inflation dataset
            peer_countries: Optional peer country datasets
            
        Returns:
            Comparative metrics dictionary
        """
        console.print("[blue]Generating comparative metrics[/blue]")
        
        metrics = {
            "brazil_performance": self._calculate_brazil_metrics(brazil_data),
            "structural_analysis": self.calculate_structural_breaks(brazil_data),
            "dual_period_analysis": self.analyze_dual_period_performance(brazil_data)
        }
        
        # Add peer comparison if available
        if peer_countries:
            comparative_df = self.emerging_markets.create_comparative_dataset(brazil_data, peer_countries)
            performance_df = self.emerging_markets.analyze_inflation_performance(comparative_df)
            convergence_df = self.emerging_markets.identify_inflation_convergence(comparative_df)
            rankings_df = self.emerging_markets.create_peer_rankings(performance_df)
            
            metrics["peer_comparison"] = {
                "performance_analysis": performance_df.to_dicts(),
                "convergence_analysis": convergence_df.to_dicts(),
                "rankings": rankings_df.to_dicts()
            }
        
        console.print("[green]Generated comprehensive comparative metrics[/green]")
        
        return metrics
    
    def _compare_inflation_periods(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Compare key statistics across different inflation periods."""
        hyperinflation = df.filter(pl.col("year") < 1994)
        modern = df.filter(pl.col("year") >= 1999)
        transition = df.filter((pl.col("year") >= 1994) & (pl.col("year") < 1999))
        
        comparison = {}
        
        for period_name, period_df in [
            ("hyperinflation", hyperinflation),
            ("transition", transition), 
            ("modern", modern)
        ]:
            if period_df.height > 0:
                comparison[period_name] = {
                    "mean_inflation": float(period_df.select("inflation_rate").mean().item()),
                    "median_inflation": float(period_df.select("inflation_rate").median().item()),
                    "std_inflation": float(period_df.select("inflation_rate").std().item()),
                    "max_inflation": float(period_df.select("inflation_rate").max().item()),
                    "min_inflation": float(period_df.select("inflation_rate").min().item()),
                    "observations": period_df.height,
                    "hyperinflation_months": int(period_df.filter(pl.col("inflation_rate") > 50).height),
                    "deflation_months": int(period_df.filter(pl.col("inflation_rate") < 0).height)
                }
        
        return comparison
    
    def _analyze_regime_transitions(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze transitions between different inflation regimes."""
        regime_transitions = df.with_columns([
            pl.col("regime").shift(1).alias("prev_regime")
        ]).filter(
            pl.col("prev_regime").is_not_null() & (pl.col("regime") != pl.col("prev_regime"))
        )
        
        transitions = {}
        for row in regime_transitions.to_dicts():
            transition_key = f"{row['prev_regime']}_to_{row['regime']}"
            if transition_key not in transitions:
                transitions[transition_key] = []
            transitions[transition_key].append({
                "date": str(row["date"]),
                "inflation_rate": row["inflation_rate"]
            })
        
        return {
            "transition_events": transitions,
            "total_transitions": regime_transitions.height,
            "regime_stability": self._calculate_regime_stability(df)
        }
    
    def _assess_policy_effectiveness(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Assess effectiveness of major policy interventions."""
        policy_windows = [
            ("plano_real", 1994, 1996),
            ("inflation_targeting", 1999, 2001),
            ("crisis_response", 2008, 2010),
            ("post_covid", 2020, 2022)
        ]
        
        policy_assessment = {}
        
        for policy_name, start_year, end_year in policy_windows:
            policy_period = df.filter(
                (pl.col("year") >= start_year) & (pl.col("year") <= end_year)
            )
            
            if policy_period.height > 0:
                before_period = df.filter(
                    (pl.col("year") >= start_year - 2) & (pl.col("year") < start_year)
                )
                
                policy_assessment[policy_name] = {
                    "during_policy": {
                        "mean_inflation": float(policy_period.select("inflation_rate").mean().item()),
                        "volatility": float(policy_period.select("inflation_rate").std().item())
                    },
                    "before_policy": {
                        "mean_inflation": float(before_period.select("inflation_rate").mean().item()) if before_period.height > 0 else None,
                        "volatility": float(before_period.select("inflation_rate").std().item()) if before_period.height > 0 else None
                    },
                    "effectiveness_score": self._calculate_policy_effectiveness_score(before_period, policy_period)
                }
        
        return policy_assessment
    
    def _place_in_international_context(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Place Brazil's performance in international context."""
        # This would typically require peer country data
        # For now, provide Brazil-specific insights
        return {
            "hyperinflation_ranking": "Historical data shows Brazil had one of the most severe hyperinflation episodes globally",
            "stabilization_success": "Post-1994 stabilization ranks among the most successful in emerging markets",
            "modern_performance": "Recent inflation targeting has been generally effective compared to regional peers"
        }
    
    def _track_volatility_evolution(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Track evolution of inflation volatility over time."""
        volatility_windows = df.with_columns([
            pl.col("inflation_rate").rolling_std(window_size=12).alias("rolling_volatility_12m"),
            pl.col("inflation_rate").rolling_std(window_size=24).alias("rolling_volatility_24m")
        ])
        
        volatility_stats = {}
        
        for period_name, start_year, end_year in [
            ("hyperinflation", 1980, 1993),
            ("transition", 1994, 1999),
            ("early_modern", 2000, 2009),
            ("recent", 2010, 2024)
        ]:
            period_data = volatility_windows.filter(
                (pl.col("year") >= start_year) & (pl.col("year") <= end_year)
            )
            
            if period_data.height > 0:
                volatility_stats[period_name] = {
                    "avg_volatility_12m": float(period_data.select("rolling_volatility_12m").mean().item()),
                    "max_volatility_12m": float(period_data.select("rolling_volatility_12m").max().item()),
                    "volatility_trend": self._calculate_volatility_trend(period_data)
                }
        
        return volatility_stats
    
    def _analyze_volatility_regimes(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze different volatility regimes in the data."""
        volatility_analysis = df.with_columns([
            pl.col("inflation_rate").rolling_std(window_size=12).alias("volatility")
        ]).filter(pl.col("volatility").is_not_null())
        
        volatility_percentiles = volatility_analysis.select([
            pl.col("volatility").quantile(0.25).alias("low_vol_threshold"),
            pl.col("volatility").quantile(0.75).alias("high_vol_threshold")
        ])
        
        low_threshold = volatility_percentiles.select("low_vol_threshold").item()
        high_threshold = volatility_percentiles.select("high_vol_threshold").item()
        
        regimes = volatility_analysis.with_columns([
            pl.when(pl.col("volatility") <= low_threshold)
            .then(pl.lit("low_volatility"))
            .when(pl.col("volatility") >= high_threshold)
            .then(pl.lit("high_volatility"))
            .otherwise(pl.lit("medium_volatility"))
            .alias("volatility_regime")
        ])
        
        return {
            "thresholds": {
                "low": float(low_threshold),
                "high": float(high_threshold)
            },
            "regime_distribution": regimes.group_by("volatility_regime").count().to_dicts(),
            "regime_persistence": self._calculate_regime_persistence(regimes, "volatility_regime")
        }
    
    def _identify_trend_breaks(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        """Identify significant trend breaks in the inflation series."""
        # Simple trend break identification using rolling means
        trend_df = df.with_columns([
            pl.col("inflation_rate").rolling_mean(window_size=6).alias("short_trend"),
            pl.col("inflation_rate").rolling_mean(window_size=24).alias("long_trend")
        ]).with_columns([
            (pl.col("short_trend") - pl.col("long_trend")).alias("trend_differential")
        ])
        
        # Identify significant trend changes
        trend_breaks = trend_df.filter(
            pl.col("trend_differential").abs() > trend_df.select("trend_differential").std().item() * 2
        )
        
        return trend_breaks.select(["date", "inflation_rate", "trend_differential"]).to_dicts()
    
    def _calculate_brazil_metrics(self, inflation_data: InflationDataset) -> Dict[str, Any]:
        """Calculate comprehensive metrics for Brazil's inflation performance."""
        data_points = []
        for dp in inflation_data.data_points:
            data_points.append({
                "date": dp.date,
                "inflation_rate": dp.value,
                "year": dp.date.year
            })
        
        df = pl.DataFrame(data_points)
        
        return {
            "overall_statistics": {
                "total_observations": df.height,
                "date_range": {
                    "start": str(df.select("date").min().item()),
                    "end": str(df.select("date").max().item())
                },
                "mean_inflation": float(df.select("inflation_rate").mean().item()),
                "median_inflation": float(df.select("inflation_rate").median().item()),
                "std_inflation": float(df.select("inflation_rate").std().item()),
                "max_inflation": float(df.select("inflation_rate").max().item()),
                "min_inflation": float(df.select("inflation_rate").min().item())
            },
            "regime_analysis": self._analyze_inflation_regimes(df),
            "cyclical_patterns": self._analyze_cyclical_patterns(df)
        }
    
    def _analyze_inflation_regimes(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze different inflation regimes."""
        regime_stats = {}
        
        # Define regime thresholds
        regimes = [
            ("deflationary", lambda x: x < 0),
            ("low_inflation", lambda x: (x >= 0) & (x < 3)),
            ("moderate_inflation", lambda x: (x >= 3) & (x < 10)),
            ("high_inflation", lambda x: (x >= 10) & (x < 50)),
            ("very_high_inflation", lambda x: (x >= 50) & (x < 100)),
            ("hyperinflation", lambda x: x >= 100)
        ]
        
        for regime_name, condition in regimes:
            regime_data = df.filter(condition(pl.col("inflation_rate")))
            if regime_data.height > 0:
                regime_stats[regime_name] = {
                    "observations": regime_data.height,
                    "percentage": round(regime_data.height / df.height * 100, 2),
                    "mean_rate": float(regime_data.select("inflation_rate").mean().item()),
                    "duration_months": regime_data.height
                }
        
        return regime_stats
    
    def _analyze_cyclical_patterns(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze cyclical patterns in inflation data."""
        if df.height == 0:
            return {}
        
        # Add seasonal analysis if we have monthly data
        monthly_df = df.with_columns([
            pl.col("date").dt.month().alias("month")
        ])
        
        seasonal_stats = monthly_df.group_by("month").agg([
            pl.col("inflation_rate").mean().alias("avg_inflation"),
            pl.col("inflation_rate").count().alias("observations")
        ]).sort("month")
        
        return {
            "seasonal_patterns": seasonal_stats.to_dicts(),
            "highest_inflation_month": int(seasonal_stats.sort("avg_inflation", descending=True).select("month").first().item()),
            "lowest_inflation_month": int(seasonal_stats.sort("avg_inflation").select("month").first().item())
        }
    
    def _calculate_policy_effectiveness_score(
        self, 
        before_period: pl.DataFrame, 
        during_period: pl.DataFrame
    ) -> Optional[float]:
        """Calculate effectiveness score for policy interventions."""
        if before_period.height == 0 or during_period.height == 0:
            return None
        
        before_mean = before_period.select("inflation_rate").mean().item()
        during_mean = during_period.select("inflation_rate").mean().item()
        before_std = before_period.select("inflation_rate").std().item()
        during_std = during_period.select("inflation_rate").std().item()
        
        # Simple effectiveness score: reduction in mean + reduction in volatility
        mean_improvement = (before_mean - during_mean) / max(before_mean, 1)
        volatility_improvement = (before_std - during_std) / max(before_std, 1)
        
        return float((mean_improvement + volatility_improvement) / 2)
    
    def _calculate_volatility_trend(self, period_data: pl.DataFrame) -> str:
        """Calculate trend direction for volatility."""
        if period_data.height < 2:
            return "insufficient_data"
        
        first_half_vol = period_data.head(period_data.height // 2).select("rolling_volatility_12m").mean().item()
        second_half_vol = period_data.tail(period_data.height // 2).select("rolling_volatility_12m").mean().item()
        
        if second_half_vol > first_half_vol * 1.1:
            return "increasing"
        elif second_half_vol < first_half_vol * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_regime_stability(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Calculate stability metrics for different regimes."""
        regime_lengths = df.with_columns([
            (pl.col("regime") != pl.col("regime").shift(1)).cumsum().alias("regime_episode")
        ]).group_by(["regime", "regime_episode"]).count().group_by("regime").agg([
            pl.col("count").mean().alias("avg_duration"),
            pl.col("count").max().alias("max_duration"),
            pl.col("count").min().alias("min_duration")
        ])
        
        return regime_lengths.to_dicts()
    
    def _calculate_regime_persistence(self, regimes_df: pl.DataFrame, regime_col: str) -> Dict[str, float]:
        """Calculate persistence metrics for regimes."""
        persistence_scores = {}
        
        for regime in regimes_df.select(regime_col).unique().to_series():
            regime_data = regimes_df.filter(pl.col(regime_col) == regime)
            if regime_data.height > 1:
                # Simple persistence: average duration of consecutive periods
                consecutive_periods = regime_data.with_columns([
                    (pl.col(regime_col) != pl.col(regime_col).shift(1)).cumsum().alias("episode")
                ]).group_by("episode").count()
                
                persistence_scores[regime] = float(consecutive_periods.select("count").mean().item())
        
        return persistence_scores


