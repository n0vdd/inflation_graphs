"""Consolidated Brazil Inflation Data Manager.

Combines the best features from all data fetcher implementations into a single,
efficient data management system using Polars for processing and async for fetching.
"""

import asyncio
import json
import logging
import requests
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import httpx
import polars as pl
from fredapi import Fred

logger = logging.getLogger(__name__)


class BrazilInflationDataManager:
    """
    Consolidated data manager for Brazil inflation data.
    
    Features:
    - Multi-source fetching (FRED → World Bank → IBGE SIDRA)
    - Polars-native data processing 
    - Async/await for efficient API calls
    - Intelligent caching and data validation
    - Graceful fallback between sources
    - Enhanced error handling with exponential backoff
    - Response caching to reduce API calls
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """Initialize the data manager."""
        self.fred_api_key = fred_api_key
        self.fred_client = Fred(api_key=fred_api_key) if fred_api_key else None
        
        # HTTP client for async requests with retry configuration
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers={"User-Agent": "BrazilInflationAnalyzer/1.0"}
        )
        
        # API endpoints
        self.world_bank_url = "https://api.worldbank.org/v2/country/BRA/indicator/FP.CPI.TOTL.ZG"
        self.ibge_sidra_url = "https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/63/p/198001-202412"
        
        # Constants
        self.PLANO_REAL_YEAR = 1994
        self.HYPERINFLATION_THRESHOLD = 50.0
        self.MIN_DATA_POINTS = 100
        
        # Enhanced error handling and caching settings
        self.MAX_RETRIES = 3
        self.INITIAL_DELAY = 1.0  # Initial delay in seconds
        self.BACKOFF_FACTOR = 2.0  # Exponential backoff multiplier
        self.CACHE_TTL_SECONDS = 3600  # 1 hour cache TTL
        self._response_cache = {}  # Simple in-memory cache
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.aclose()
    
    def _generate_cache_key(self, url: str, params: Dict = None) -> str:
        """Generate a cache key for the request."""
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            return f"{url}?{param_str}"
        return url
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        if not cache_entry:
            return False
        cache_time = cache_entry.get("timestamp", 0)
        return (time.time() - cache_time) < self.CACHE_TTL_SECONDS
    
    async def _make_http_request_with_retry(
        self, 
        url: str, 
        params: Dict = None,
        use_cache: bool = True
    ) -> httpx.Response:
        """Make HTTP request with exponential backoff retry and caching."""
        cache_key = self._generate_cache_key(url, params)
        
        # Check cache first
        if use_cache and cache_key in self._response_cache:
            cache_entry = self._response_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.info(f"🚀 Using cached response for {url}")
                # Create mock response object
                response = httpx.Response(
                    status_code=200,
                    content=cache_entry["content"],
                    headers=cache_entry.get("headers", {}),
                    request=httpx.Request("GET", url)
                )
                return response
        
        # Make request with retry logic
        delay = self.INITIAL_DELAY
        last_exception = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(f"🌐 Making HTTP request to {url} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                response = await self.http_client.get(url, params=params)
                response.raise_for_status()
                
                # Cache successful response
                if use_cache:
                    self._response_cache[cache_key] = {
                        "content": response.content,
                        "headers": dict(response.headers),
                        "timestamp": time.time()
                    }
                    logger.info(f"💾 Cached response for {url}")
                
                return response
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code in [429, 502, 503, 504]:  # Retry-able errors
                    logger.warning(f"⚠️ HTTP {e.response.status_code} error, retrying in {delay:.1f}s...")
                    if attempt < self.MAX_RETRIES - 1:
                        await asyncio.sleep(delay)
                        delay *= self.BACKOFF_FACTOR
                        continue
                else:
                    # Non-retryable error
                    logger.error(f"❌ Non-retryable HTTP error {e.response.status_code}: {e}")
                    raise
                    
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
                last_exception = e
                logger.warning(f"⚠️ Network error: {e}, retrying in {delay:.1f}s...")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    delay *= self.BACKOFF_FACTOR
                    continue
                    
        # All retries exhausted
        logger.error(f"❌ All {self.MAX_RETRIES} attempts failed for {url}")
        raise last_exception
    
    async def fetch_best_data(
        self, 
        start_year: int = 1980,
        end_year: int = 2024,
        preferred_source: str = "fred"
    ) -> pl.DataFrame:
        """
        Fetch best available data with intelligent source selection.
        
        Returns Polars DataFrame with columns:
        - date: Date
        - inflation_rate: Float64  
        - source: Utf8
        - is_hyperinflation: Boolean
        - is_modern: Boolean
        """
        logger.info(f"🔍 Fetching Brazil inflation data ({start_year}-{end_year})")
        
        # Try sources in order of preference
        sources = self._get_source_order(preferred_source)
        
        for source_name in sources:
            try:
                logger.info(f"📡 Trying {source_name} source...")
                
                if source_name == "fred":
                    df = await self._fetch_fred_data(start_year, end_year)
                elif source_name == "world_bank":
                    df = await self._fetch_world_bank_data(start_year, end_year)
                elif source_name == "ibge":
                    df = await self._fetch_ibge_data(start_year, end_year)
                else:
                    continue
                
                # Validate data quality
                if self._validate_data_quality(df, start_year, end_year):
                    logger.info(f"✅ Successfully fetched {df.height} observations from {source_name}")
                    return self._process_data(df, source_name)
                else:
                    logger.warning(f"⚠️ {source_name} data quality insufficient")
                    
            except Exception as e:
                logger.warning(f"❌ {source_name} failed: {e}")
                continue
        
        # If all sources fail, create minimal demo data
        logger.error("🚨 All data sources failed - using minimal demo data")
        return self._create_fallback_data(start_year, end_year)
    
    def _get_source_order(self, preferred: str) -> List[str]:
        """Get data source order based on preference - prioritize IBGE for monthly data."""
        # IBGE SIDRA should be first priority for monthly data and structural analysis
        sources = ["ibge", "fred", "world_bank"]
        
        if preferred in sources:
            sources.remove(preferred)
            sources.insert(0, preferred)
        
        return sources
    
    async def _fetch_fred_data(self, start_year: int, end_year: int) -> pl.DataFrame:
        """Fetch data from FRED API."""
        if not self.fred_client:
            raise ValueError("FRED API key not configured")
        
        # FRED series for Brazil inflation (annual % change in CPI)
        series_id = "FPCPITOTLZGBRA"
        
        try:
            # Fetch data
            data = self.fred_client.get_series(
                series_id,
                start=f"{start_year}-01-01",
                end=f"{end_year}-12-31",
                frequency='a'  # Annual data
            )
            
            # Convert to Polars DataFrame
            df = pl.DataFrame({
                "date": [date(idx.year, 12, 31) for idx in data.index],
                "inflation_rate": data.values.astype(float)
            })
            
            return df.filter(pl.col("inflation_rate").is_not_null())
            
        except Exception as e:
            raise Exception(f"FRED API error: {e}")
    
    async def _fetch_world_bank_data(self, start_year: int, end_year: int) -> pl.DataFrame:
        """Fetch data from World Bank API with retry and caching."""
        params = {
            "format": "json",
            "date": f"{start_year}:{end_year}"
        }
        
        try:
            response = await self._make_http_request_with_retry(self.world_bank_url, params)
            
            data = response.json()
            if not isinstance(data, list) or len(data) < 2:
                raise ValueError("Invalid World Bank response format")
            
            records = data[1]  # Data is in second element
            
            inflation_data = []
            for record in records:
                if record.get('value') is not None:
                    year = int(record['date'])
                    value = float(record['value'])
                    inflation_data.append({
                        'date': date(year, 12, 31),
                        'inflation_rate': value
                    })
            
            return pl.DataFrame(inflation_data).sort("date")
            
        except Exception as e:
            raise Exception(f"World Bank API error: {e}")
    
    async def _fetch_ibge_data(self, start_year: int, end_year: int) -> pl.DataFrame:
        """Fetch monthly data from IBGE SIDRA with improved parsing and retry logic."""
        try:
            # Use improved IBGE SIDRA URL format
            table = "1737"  # IPCA table
            variable = "63"  # IPCA - Variação mensal (%)
            period = f"{start_year}01-{end_year}12"  # Period range YYYYMM-YYYYMM
            
            url = f"https://apisidra.ibge.gov.br/values/t/{table}/n1/all/v/{variable}/p/{period}"
            
            response = await self._make_http_request_with_retry(url)
            
            data = response.json()
            logger.debug(f"IBGE parsed data type: {type(data)}, length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
            
            if not isinstance(data, list) or len(data) < 2:
                raise ValueError(f"Invalid IBGE response format: {type(data)}")
            
            records = data[1:]  # Skip header
            monthly_data = []
            
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                
                value_str = record.get('V')
                period_code = record.get('D3C')  # YYYYMM format
                
                # Skip empty values
                if not value_str or value_str in ['-', '...', '']:
                    continue
                
                try:
                    # Handle Brazilian decimal format
                    value = float(value_str.replace(',', '.'))
                    
                    if not period_code or len(period_code) != 6:
                        continue
                    
                    year = int(period_code[:4])
                    month = int(period_code[4:6])
                    
                    # Validate year/month ranges
                    if not (1900 <= year <= 2100 and 1 <= month <= 12):
                        continue
                    
                    if start_year <= year <= end_year:
                        monthly_data.append({
                            'date': date(year, month, 1),
                            'inflation_rate': value
                        })
                        
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse record {i}: {e}")
                    continue
            
            logger.info(f"IBGE parsed {len(monthly_data)} valid monthly data points")
            
            if not monthly_data:
                raise ValueError("No valid IBGE data found")
            
            # Return monthly data for structural analysis (don't aggregate to annual)
            df = pl.DataFrame(monthly_data)
            return df.sort("date")
            
        except Exception as e:
            logger.error(f"IBGE API detailed error: {e}")
            raise Exception(f"IBGE API error: {e}")
    
    def _validate_data_quality(self, df: pl.DataFrame, start_year: int, end_year: int) -> bool:
        """Validate data quality and completeness."""
        logger.info(f"Validating data: {df.height} records, expected range {start_year}-{end_year}")
        
        # Calculate expected points based on data frequency
        years_span = end_year - start_year + 1
        
        # Check if this looks like monthly data (more points than years)
        if df.height > years_span * 1.5:
            # Monthly data: expect around 12 points per year
            min_expected_points = years_span * 12 * 0.7  # 70% completeness for monthly
            logger.info(f"Detected monthly data frequency, expecting ~{years_span * 12} points")
        else:
            # Annual data: expect 1 point per year
            min_expected_points = years_span * 0.3  # 30% completeness for annual
            logger.info(f"Detected annual data frequency, expecting ~{years_span} points")
        
        if df.height < min_expected_points:
            logger.warning(f"Insufficient data points: {df.height} < {min_expected_points:.0f}")
            return False
        
        # Check for reasonable data range
        stats = df.select([
            pl.col("inflation_rate").min().alias("min_rate"),
            pl.col("inflation_rate").max().alias("max_rate"),
            pl.col("inflation_rate").null_count().alias("null_count")
        ]).row(0)
        
        min_rate, max_rate, null_count = stats
        logger.info(f"Data stats: min={min_rate:.2f}%, max={max_rate:.2f}%, nulls={null_count}/{df.height}")
        
        # Basic sanity checks
        if null_count > df.height * 0.5:  # More than 50% missing
            logger.warning(f"Too many nulls: {null_count}/{df.height} > 50%")
            return False
        
        if min_rate < -50 or max_rate > 10000:  # Allow for historical hyperinflation
            logger.warning(f"Data out of range: min={min_rate}, max={max_rate}")
            return False
        
        logger.info("✅ Data quality validation passed")
        return True
    
    def _process_data(self, df: pl.DataFrame, source: str) -> pl.DataFrame:
        """Process and enrich the data with additional columns."""
        return (df
                .with_columns([
                    pl.lit(source).alias("source"),
                    pl.col("date").dt.year().alias("year"),
                    (pl.col("date").dt.year() < self.PLANO_REAL_YEAR).alias("is_hyperinflation"),
                    (pl.col("date").dt.year() >= self.PLANO_REAL_YEAR).alias("is_modern"),
                    (pl.col("inflation_rate") >= self.HYPERINFLATION_THRESHOLD).alias("is_extreme_inflation")
                ])
                .sort("date"))
    
    def _create_fallback_data(self, start_year: int, end_year: int) -> pl.DataFrame:
        """Create minimal fallback data if all sources fail."""
        import random
        
        logger.warning("🎭 Creating fallback demo data")
        
        data_points = []
        for year in range(start_year, end_year + 1):
            if year < self.PLANO_REAL_YEAR:
                # Hyperinflation period: high volatile rates
                rate = random.uniform(20, 200) if year < 1990 else random.uniform(50, 2000)
            else:
                # Modern period: low stable rates  
                rate = random.uniform(2, 15)
            
            data_points.append({
                'date': date(year, 12, 31),
                'inflation_rate': rate
            })
        
        df = pl.DataFrame(data_points)
        return self._process_data(df, "demo")
    
    def save_data(self, df: pl.DataFrame, filename: str, format: str = "parquet") -> Path:
        """Save processed data to file."""
        data_dir = Path("data/processed")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            file_path = data_dir / f"{filename}.parquet"
            df.write_parquet(file_path)
        elif format == "csv":
            file_path = data_dir / f"{filename}.csv"
            df.write_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"💾 Data saved to {file_path}")
        return file_path
    
    def load_data(self, file_path: Union[str, Path]) -> pl.DataFrame:
        """Load data from file."""
        file_path = Path(file_path)
        
        if file_path.suffix == ".parquet":
            return pl.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            return pl.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    async def fetch_all_sources_for_validation(
        self, 
        start_year: int = 1980, 
        end_year: int = 2024
    ) -> Dict[str, Optional[pl.DataFrame]]:
        """Fetch data from all sources for cross-validation."""
        sources_data = {}
        
        for source in ["ibge", "fred", "world_bank"]:
            try:
                logger.info(f"🔄 Fetching {source} data for validation...")
                if source == "ibge":
                    df = await self._fetch_ibge_data(start_year, end_year)
                elif source == "fred":
                    df = await self._fetch_fred_data(start_year, end_year)
                elif source == "world_bank":
                    df = await self._fetch_world_bank_data(start_year, end_year)
                
                if self._validate_data_quality(df, start_year, end_year):
                    sources_data[source] = df
                    logger.info(f"✅ {source}: {df.height} points validated")
                else:
                    sources_data[source] = None
                    logger.warning(f"⚠️ {source}: validation failed")
                    
            except Exception as e:
                logger.warning(f"❌ {source} failed: {e}")
                sources_data[source] = None
        
        return sources_data
    
    def cross_validate_sources(self, sources_data: Dict[str, Optional[pl.DataFrame]]) -> Dict[str, Any]:
        """Cross-validate data between different sources."""
        available_sources = {k: v for k, v in sources_data.items() if v is not None}
        
        if len(available_sources) < 2:
            return {"validation_possible": False, "reason": "Need at least 2 sources for cross-validation"}
        
        validation_report = {
            "validation_possible": True,
            "sources_count": len(available_sources),
            "consistency_scores": {},
            "discrepancies": [],
            "recommendations": []
        }
        
        # Compare IBGE (monthly) with FRED/World Bank (annual) if available
        if "ibge" in available_sources and ("fred" in available_sources or "world_bank" in available_sources):
            ibge_df = available_sources["ibge"]
            
            # Aggregate IBGE monthly to annual for comparison
            ibge_annual = (ibge_df
                          .with_columns(pl.col("date").dt.year().alias("year"))
                          .group_by("year")
                          .agg([
                              # Compound monthly inflation to annual
                              ((pl.col("inflation_rate") / 100 + 1).product() - 1).alias("annual_rate") * 100,
                              pl.col("date").min().alias("first_date")
                          ])
                          .sort("year"))
            
            # Compare with FRED if available
            if "fred" in available_sources:
                fred_df = available_sources["fred"]
                comparison = self._compare_annual_data(ibge_annual, fred_df, "IBGE", "FRED")
                validation_report["consistency_scores"]["ibge_vs_fred"] = comparison
            
            # Compare with World Bank if available
            if "world_bank" in available_sources:
                wb_df = available_sources["world_bank"]
                comparison = self._compare_annual_data(ibge_annual, wb_df, "IBGE", "World Bank")
                validation_report["consistency_scores"]["ibge_vs_worldbank"] = comparison
        
        # Generate recommendations based on validation
        best_source = max(available_sources.keys(), key=lambda k: available_sources[k].height)
        validation_report["recommended_primary_source"] = best_source
        validation_report["data_points_by_source"] = {k: v.height for k, v in available_sources.items()}
        
        return validation_report
    
    def _compare_annual_data(self, df1: pl.DataFrame, df2: pl.DataFrame, name1: str, name2: str) -> Dict[str, Any]:
        """Compare two annual datasets."""
        # Join on year for comparison
        df1_renamed = df1.select([
            pl.col("year"),
            pl.col("annual_rate").alias("rate_1") if "annual_rate" in df1.columns else pl.col("inflation_rate").alias("rate_1")
        ])
        
        df2_renamed = df2.select([
            pl.col("date").dt.year().alias("year"),
            pl.col("inflation_rate").alias("rate_2")
        ])
        
        joined = df1_renamed.join(df2_renamed, on="year", how="inner")
        
        if joined.height == 0:
            return {"comparison_possible": False, "reason": "No overlapping years"}
        
        # Calculate differences and correlations
        comparison_stats = joined.select([
            pl.count().alias("overlapping_years"),
            ((pl.col("rate_1") - pl.col("rate_2")).abs()).mean().alias("mean_abs_difference"),
            ((pl.col("rate_1") - pl.col("rate_2")).abs()).max().alias("max_difference"),
            pl.corr("rate_1", "rate_2").alias("correlation")
        ]).row(0, named=True)
        
        # Identify large discrepancies (>2% difference)
        discrepancies = joined.filter(
            (pl.col("rate_1") - pl.col("rate_2")).abs() > 2.0
        ).select(["year", "rate_1", "rate_2", ((pl.col("rate_1") - pl.col("rate_2")).abs()).alias("difference")])
        
        return {
            "comparison_possible": True,
            "source_1": name1,
            "source_2": name2,
            "overlapping_years": comparison_stats["overlapping_years"],
            "mean_abs_difference": comparison_stats["mean_abs_difference"],
            "max_difference": comparison_stats["max_difference"],
            "correlation": comparison_stats["correlation"],
            "large_discrepancies": discrepancies.to_dicts() if discrepancies.height > 0 else [],
            "consistency_score": max(0, 1 - (comparison_stats["mean_abs_difference"] / 10))  # Normalize to 0-1
        }

    def get_data_summary(self, df: pl.DataFrame) -> Dict:
        """Generate comprehensive data summary."""
        summary = (df
                  .select([
                      pl.count().alias("total_observations"),
                      pl.col("date").min().alias("start_date"),
                      pl.col("date").max().alias("end_date"),
                      pl.col("inflation_rate").mean().alias("mean_inflation"),
                      pl.col("inflation_rate").median().alias("median_inflation"),
                      pl.col("inflation_rate").std().alias("std_inflation"),
                      pl.col("inflation_rate").min().alias("min_inflation"),
                      pl.col("inflation_rate").max().alias("max_inflation"),
                      pl.col("is_hyperinflation").sum().alias("hyperinflation_count"),
                      pl.col("is_modern").sum().alias("modern_count"),
                      pl.col("is_extreme_inflation").sum().alias("extreme_count")
                  ])
                  .row(0, named=True))
        
        return {
            "total_observations": summary["total_observations"],
            "date_range": (summary["start_date"], summary["end_date"]),
            "hyperinflation_observations": summary["hyperinflation_count"], 
            "modern_observations": summary["modern_count"],
            "extreme_observations": summary["extreme_count"],
            "statistics": {
                "mean": summary["mean_inflation"],
                "median": summary["median_inflation"], 
                "std": summary["std_inflation"],
                "min": summary["min_inflation"],
                "max": summary["max_inflation"]
            }
        }


# Convenience functions for backward compatibility
async def fetch_brazil_inflation_data(
    start_year: int = 1980,
    end_year: int = 2024, 
    fred_api_key: Optional[str] = None,
    preferred_source: str = "fred"
) -> pl.DataFrame:
    """Convenience function to fetch Brazil inflation data."""
    async with BrazilInflationDataManager(fred_api_key) as manager:
        return await manager.fetch_best_data(start_year, end_year, preferred_source)


def create_demo_data(start_year: int = 1980, end_year: int = 2024) -> pl.DataFrame:
    """Create demo data for testing purposes.""" 
    manager = BrazilInflationDataManager()
    return manager._create_fallback_data(start_year, end_year)