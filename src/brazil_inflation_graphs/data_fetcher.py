"""Brazil inflation data fetcher with multiple data sources and async support."""

import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx
import polars as pl
from fredapi import Fred
from rich.console import Console
from rich.progress import Progress, TaskID

from .config import Settings, get_settings
from .exceptions import (
    APIError,
    DataFetchError, 
    FREDAPIError,
    IBGESIDRAError,
    WorldBankAPIError,
)
from .models import DataSource, InflationDataPoint, InflationDataset


console = Console()


class BrazilInflationFetcher:
    """
    Comprehensive Brazil inflation data fetcher with multiple data sources.
    
    Supports FRED API, World Bank API, and IBGE SIDRA with graceful fallback.
    Uses async/await patterns for efficient data retrieval.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the fetcher with configuration."""
        self.settings = settings or get_settings()
        self.client: Optional[httpx.AsyncClient] = None
        self.fred_client: Optional[Fred] = None
        
        # Initialize FRED client if API key is available
        if self.settings.api.fred_api_key:
            try:
                self.fred_client = Fred(api_key=self.settings.api.fred_api_key)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to initialize FRED client: {e}[/yellow]")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.api.timeout_seconds),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
    
    async def fetch_all_sources(
        self, 
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        include_extended_data: bool = True
    ) -> Dict[str, Optional[InflationDataset]]:
        """
        Fetch data from all available sources concurrently.
        
        Args:
            start_year: Starting year for data (defaults to config)
            end_year: Ending year for data (defaults to config)
            include_extended_data: Whether to fetch core inflation and sectoral data
        
        Returns:
            Dictionary mapping data sources to their datasets
        """
        start_year = start_year or self.settings.data_source.start_year
        end_year = end_year or self.settings.data_source.end_year
        
        # Create tasks for all data sources
        tasks = {
            "FRED": self._fetch_fred_data(start_year, end_year),
            "WORLD_BANK": self._fetch_world_bank_data(start_year, end_year),
            "IBGE_SIDRA": self._fetch_ibge_data(start_year, end_year),
        }
        
        # Add extended IBGE data sources if requested
        if include_extended_data:
            tasks.update({
                "IBGE_CORE": self._fetch_ibge_core_inflation(start_year, end_year),
                "IBGE_SECTORAL": self._fetch_ibge_sectoral_data(start_year, end_year),
                "IBGE_REGIONAL": self._fetch_ibge_regional_data(start_year, end_year),
            })
        
        results = {}
        
        with Progress() as progress:
            task_id = progress.add_task("[cyan]Fetching inflation data...", total=len(tasks))
            
            # Execute tasks concurrently
            for source_key, coro in tasks.items():
                try:
                    results[source_key] = await coro
                    console.print(f"[green]SUCCESS[/green] Successfully fetched data from {source_key}")
                except Exception as e:
                    console.print(f"[red]ERROR[/red] Failed to fetch data from {source_key}: {e}")
                    results[source_key] = None
                
                progress.update(task_id, advance=1)
        
        return results
    
    async def fetch_best_data(
        self,
        start_year: Optional[int] = None, 
        end_year: Optional[int] = None,
        preferred_source: Optional[DataSource] = None
    ) -> InflationDataset:
        """
        Fetch data with fallback strategy, returning the best available dataset.
        
        Args:
            start_year: Starting year for data
            end_year: Ending year for data  
            preferred_source: Preferred data source (defaults to FRED)
        
        Returns:
            Best available inflation dataset
            
        Raises:
            DataFetchError: If no data sources are available
        """
        # Prioritize IBGE_SIDRA for monthly data, then FRED for fallback
        preferred_source = preferred_source or DataSource.IBGE_SIDRA
        
        # Define source priority order (IBGE_SIDRA for monthly, FRED for fallback, World Bank last)
        if preferred_source == DataSource.IBGE_SIDRA:
            source_priority = [DataSource.IBGE_SIDRA, DataSource.FRED, DataSource.WORLD_BANK]
        elif preferred_source == DataSource.FRED:
            source_priority = [DataSource.FRED, DataSource.IBGE_SIDRA, DataSource.WORLD_BANK]
        else:
            source_priority = [preferred_source, DataSource.IBGE_SIDRA, DataSource.FRED, DataSource.WORLD_BANK]
        
        # Remove duplicates while preserving order
        seen = set()
        source_priority = [x for x in source_priority if not (x in seen or seen.add(x))]
        
        # Try each source in order
        for source in source_priority:
            try:
                console.print(f"[blue]Trying {source.value}...[/blue]")
                
                if source == DataSource.FRED:
                    dataset = await self._fetch_fred_data(start_year, end_year)
                elif source == DataSource.WORLD_BANK:
                    dataset = await self._fetch_world_bank_data(start_year, end_year)
                elif source == DataSource.IBGE_SIDRA:
                    dataset = await self._fetch_ibge_data(start_year, end_year)
                else:
                    continue
                
                if dataset and dataset.data_points:
                    console.print(f"[green]SUCCESS[/green] Successfully retrieved data from {source.value}")
                    return dataset
                    
            except Exception as e:
                console.print(f"[yellow]Warning: {source.value} failed: {e}[/yellow]")
                continue
        
        raise DataFetchError("All data sources failed to retrieve data")
    
    async def _fetch_fred_data(
        self, 
        start_year: int,
        end_year: int
    ) -> Optional[InflationDataset]:
        """
        Fetch data from FRED API.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            FRED inflation dataset or None if failed
        """
        if not self.fred_client:
            raise FREDAPIError("FRED API key not available", source="FRED")
        
        try:
            # FRED API is synchronous, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            
            data = await loop.run_in_executor(
                None,
                lambda: self.fred_client.get_series(
                    self.settings.api.fred_series_id,
                    start=f"{start_year}-01-01",
                    end=f"{end_year}-12-31"
                )
            )
            
            if data is None or data.empty:
                raise FREDAPIError("No data returned from FRED", source="FRED")
            
            # Convert pandas Series to Polars DataFrame directly
            df = pl.DataFrame({
                "date": data.index.to_list(),
                "inflation_rate": data.values.tolist()
            }).with_columns([
                pl.col("date").cast(pl.Date),
                pl.col("inflation_rate").cast(pl.Float32)
            ]).filter(
                pl.col("inflation_rate").is_not_null()
            )
            
            # Convert to data model for compatibility
            data_points = [
                InflationDataPoint(
                    date=row["date"],
                    value=float(row["inflation_rate"]),
                    source=DataSource.FRED
                )
                for row in df.to_dicts()
            ]
            
            if not data_points:
                raise FREDAPIError("No valid data points from FRED", source="FRED")
            
            return InflationDataset(
                data_points=data_points,
                source=DataSource.FRED
            )
            
        except Exception as e:
            if isinstance(e, FREDAPIError):
                raise
            raise FREDAPIError(f"FRED API error: {e}", source="FRED")
    
    async def _fetch_world_bank_data(
        self,
        start_year: int, 
        end_year: int
    ) -> Optional[InflationDataset]:
        """
        Fetch data from World Bank API.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            World Bank inflation dataset or None if failed
        """
        if not self.client:
            raise WorldBankAPIError("HTTP client not initialized", source="World Bank")
        
        url = (
            f"{self.settings.api.world_bank_base_url}/country/BRA/indicator/"
            f"{self.settings.api.world_bank_indicator}"
        )
        
        params = {
            "format": "json",
            "date": f"{start_year}:{end_year}",
            "per_page": "1000"
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # World Bank returns [metadata, data] array
            if not isinstance(data, list) or len(data) < 2:
                raise WorldBankAPIError("Invalid World Bank API response format", source="World Bank")
            
            raw_data = data[1]  # Data is in second element
            if not raw_data:
                raise WorldBankAPIError("No data returned from World Bank", source="World Bank")
            
            data_points = []
            for item in raw_data:
                if item.get("value") is not None:
                    try:
                        year = int(item["date"])
                        # World Bank provides annual data, use December as representative date
                        data_points.append(InflationDataPoint(
                            date=date(year, 12, 31),
                            value=float(item["value"]),
                            source=DataSource.WORLD_BANK
                        ))
                    except (ValueError, KeyError) as e:
                        console.print(f"[yellow]Skipping invalid World Bank data point: {e}[/yellow]")
                        continue
            
            if not data_points:
                raise WorldBankAPIError("No valid data points from World Bank", source="World Bank")
            
            return InflationDataset(
                data_points=data_points,
                source=DataSource.WORLD_BANK
            )
            
        except httpx.HTTPStatusError as e:
            raise WorldBankAPIError(
                f"World Bank API HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                source="World Bank"
            )
        except Exception as e:
            if isinstance(e, WorldBankAPIError):
                raise
            raise WorldBankAPIError(f"World Bank API error: {e}", source="World Bank")
    
    async def _fetch_ibge_data(
        self,
        start_year: int,
        end_year: int
    ) -> Optional[InflationDataset]:
        """
        Fetch data from IBGE SIDRA API.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            IBGE inflation dataset or None if failed
        """
        if not self.client:
            raise IBGESIDRAError("HTTP client not initialized", source="IBGE SIDRA")
        
        # IBGE SIDRA API URL - using REST-style format that works
        # Format: /values/t/TABLE/n1/all/v/VARIABLE/p/PERIOD
        table = str(self.settings.api.ibge_sidra_table)  # 1737 = IPCA table
        variable = "63"  # IPCA - Variação mensal (%)
        period = f"{start_year}01-{end_year}12"  # Period range YYYYMM-YYYYMM
        
        url = f"https://apisidra.ibge.gov.br/values/t/{table}/n1/all/v/{variable}/p/{period}"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list) or len(data) < 2:
                raise IBGESIDRAError("Invalid IBGE SIDRA API response format", source="IBGE SIDRA")
            
            # First record contains headers, skip it
            raw_data = data[1:]
            
            data_points = []
            for record in raw_data:
                try:
                    # IBGE JSON format has consistent field names:
                    # V = value, D3C = period code (YYYYMM), D3N = period name
                    if not isinstance(record, dict):
                        console.print(f"[yellow]Skipping non-dict record: {record}[/yellow]")
                        continue
                        
                    value_str = record.get('V')
                    period_code = record.get('D3C')  # Format: "202312"
                    period_name = record.get('D3N')  # Format: "dezembro 2023"
                    
                    # Skip empty or invalid values
                    if value_str in [None, "", "-", "..."]:
                        console.print(f"[yellow]Skipping empty value for period {period_name}[/yellow]")
                        continue
                    
                    # Parse value (handle Brazilian decimal format)
                    value = float(value_str.replace(",", "."))
                    
                    # Parse period code (YYYYMM format)
                    if not period_code or len(period_code) != 6:
                        console.print(f"[yellow]Invalid period code {period_code} for {period_name}[/yellow]")
                        continue
                        
                    year = int(period_code[:4])
                    month = int(period_code[4:6])
                    
                    # Validate year/month ranges
                    if not (1900 <= year <= 2100 and 1 <= month <= 12):
                        console.print(f"[yellow]Invalid date {year}-{month} for {period_name}[/yellow]")
                        continue
                    
                    data_points.append(InflationDataPoint(
                        date=date(year, month, 1),  # First day of month
                        value=value,
                        source=DataSource.IBGE_SIDRA
                    ))
                    
                except (ValueError, KeyError, TypeError) as e:
                    console.print(f"[yellow]Skipping invalid IBGE data point: {e} - Record: {record}[/yellow]")
                    continue
            
            if not data_points:
                raise IBGESIDRAError("No valid data points from IBGE SIDRA", source="IBGE SIDRA")
            
            console.print(f"[green]Successfully parsed {len(data_points)} IBGE data points[/green]")
            
            return InflationDataset(
                data_points=data_points,
                source=DataSource.IBGE_SIDRA
            )
            
        except httpx.HTTPStatusError as e:
            raise IBGESIDRAError(
                f"IBGE SIDRA API HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                source="IBGE SIDRA"
            )
        except Exception as e:
            if isinstance(e, IBGESIDRAError):
                raise
            raise IBGESIDRAError(f"IBGE SIDRA API error: {e}", source="IBGE SIDRA")
    
    async def _fetch_ibge_core_inflation(
        self,
        start_year: int,
        end_year: int
    ) -> Optional[InflationDataset]:
        """
        Fetch core inflation (IPCA-EX) from IBGE SIDRA.
        
        Core inflation excludes volatile items like food and energy.
        Uses IBGE Table 11428 - IPCA-EX (ex-alimentos não processados e combustíveis).
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            IBGE core inflation dataset or None if failed
        """
        if not self.client:
            raise IBGESIDRAError("HTTP client not initialized", source="IBGE SIDRA Core")
        
        # IBGE Table 11428 - IPCA-EX (core inflation)
        table = "11428"
        variable = "11427"  # IPCA-EX - Variação mensal (%)
        period = f"{start_year}01-{end_year}12"
        
        url = f"https://apisidra.ibge.gov.br/values/t/{table}/n1/all/v/{variable}/p/{period}"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list) or len(data) < 2:
                raise IBGESIDRAError("Invalid IBGE core inflation response", source="IBGE SIDRA Core")
            
            raw_data = data[1:]  # Skip header
            data_points = []
            
            for record in raw_data:
                try:
                    if not isinstance(record, dict):
                        continue
                        
                    value_str = record.get('V')
                    period_code = record.get('D3C')
                    
                    if value_str in [None, "", "-", "..."]:
                        continue
                    
                    value = float(value_str.replace(",", "."))
                    
                    if not period_code or len(period_code) != 6:
                        continue
                    
                    year = int(period_code[:4])
                    month = int(period_code[4:6])
                    
                    if not (1900 <= year <= 2100 and 1 <= month <= 12):
                        continue
                    
                    data_points.append(InflationDataPoint(
                        date=date(year, month, 1),
                        value=value,
                        source=DataSource.IBGE_SIDRA,
                        metadata={"indicator_type": "core_inflation"}
                    ))
                    
                except (ValueError, KeyError, TypeError):
                    continue
            
            if not data_points:
                raise IBGESIDRAError("No valid core inflation data points", source="IBGE SIDRA Core")
            
            console.print(f"[green]Successfully parsed {len(data_points)} IBGE core inflation points[/green]")
            
            return InflationDataset(
                data_points=data_points,
                source=DataSource.IBGE_SIDRA,
                metadata={"dataset_type": "core_inflation", "table": table}
            )
            
        except Exception as e:
            if isinstance(e, IBGESIDRAError):
                raise
            raise IBGESIDRAError(f"IBGE core inflation error: {e}", source="IBGE SIDRA Core")
    
    async def _fetch_ibge_sectoral_data(
        self,
        start_year: int,
        end_year: int
    ) -> Optional[InflationDataset]:
        """
        Fetch sectoral breakdown of IPCA from IBGE SIDRA.
        
        Retrieves inflation data by major categories:
        - Food and beverages
        - Housing  
        - Transportation
        - Health and personal care
        - Education
        - Communication
        - Recreation
        - Clothing
        - Personal expenses
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            IBGE sectoral inflation dataset or None if failed
        """
        if not self.client:
            raise IBGESIDRAError("HTTP client not initialized", source="IBGE SIDRA Sectoral")
        
        # IBGE Table 7060 - IPCA by groups (reliable sectoral data from 2020 onwards)
        # Only use table 7060 as it has proper structure, adjust date range as needed
        table = "7060"
        if start_year < 2020:
            start_year = 2020  # Adjust to table availability
        variable = "63"  # Variação mensal (%)  
        period = f"{start_year}01-{end_year}12"
        
        # Use the correct URL format for IBGE SIDRA API
        # Table 7060 uses c315 classification for groups - only use this table
        url = f"https://apisidra.ibge.gov.br/values/t/{table}/n1/all/v/{variable}/p/{period}/c315/7169,7170,7445,7486,7558,7625,7660,7712,7766"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list) or len(data) < 2:
                raise IBGESIDRAError("Invalid IBGE sectoral response", source="IBGE SIDRA Sectoral")
            
            raw_data = data[1:]  # Skip header
            data_points = []
            
            for i, record in enumerate(raw_data):
                try:
                    if not isinstance(record, dict):
                        continue
                        
                    value_str = record.get('V')
                    period_code = record.get('D3C')
                    # Group information is in D4C/D4N for table 7060
                    
                    # Use table 7060 field names (c315 classification data is in D4)
                    group_code = record.get('D4C')  # Group code
                    group_name = record.get('D4N')  # Group name
                    
                    if value_str in [None, "", "-", "..."]:
                        continue
                    
                    value = float(value_str.replace(",", "."))
                    
                    if not period_code or len(period_code) != 6:
                        continue
                    
                    year = int(period_code[:4])
                    month = int(period_code[4:6])
                    
                    if not (1900 <= year <= 2100 and 1 <= month <= 12):
                        continue
                    
                    data_points.append(InflationDataPoint(
                        date=date(year, month, 1),
                        value=value,
                        source=DataSource.IBGE_SIDRA,
                        metadata={
                            "indicator_type": "sectoral",
                            "group_code": group_code,
                            "group_name": group_name
                        }
                    ))
                    
                except (ValueError, KeyError, TypeError):
                    continue
            
            if not data_points:
                raise IBGESIDRAError("No valid sectoral data points", source="IBGE SIDRA Sectoral")
            
            console.print(f"[green]Successfully parsed {len(data_points)} IBGE sectoral data points[/green]")
            
            return InflationDataset(
                data_points=data_points,
                source=DataSource.IBGE_SIDRA,
                metadata={"dataset_type": "sectoral", "table": table}
            )
            
        except Exception as e:
            if isinstance(e, IBGESIDRAError):
                raise
            raise IBGESIDRAError(f"IBGE sectoral data error: {e}", source="IBGE SIDRA Sectoral")
    
    async def _fetch_ibge_regional_data(
        self,
        start_year: int,
        end_year: int
    ) -> Optional[InflationDataset]:
        """
        Fetch regional IPCA data from IBGE SIDRA.
        
        Retrieves inflation data for Brazil's 15 metropolitan areas
        covered by IPCA: Belém, Fortaleza, Recife, Salvador, Belo Horizonte,
        Vitória, Rio de Janeiro, São Paulo, Curitiba, Florianópolis, 
        Porto Alegre, Campo Grande, Cuiabá, Goiânia, and Brasília.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            IBGE regional inflation dataset or None if failed
        """
        if not self.client:
            raise IBGESIDRAError("HTTP client not initialized", source="IBGE SIDRA Regional")
        
        # IBGE Table 1737 - IPCA by metropolitan areas
        table = "1737"
        variable = "63"  # Variação mensal (%)
        period = f"{start_year}01-{end_year}12"
        
        # Use the correct URL format for IBGE SIDRA API - table 1737 uses n7 classification for metropolitan areas
        url = f"https://apisidra.ibge.gov.br/values/t/{table}/n7/all/v/{variable}/p/{period}"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list) or len(data) < 2:
                raise IBGESIDRAError("Invalid IBGE regional response", source="IBGE SIDRA Regional")
            
            raw_data = data[1:]  # Skip header
            data_points = []
            
            for record in raw_data:
                try:
                    if not isinstance(record, dict):
                        continue
                        
                    value_str = record.get('V')
                    period_code = record.get('D3C')
                    region_code = record.get('D7C')  # Metropolitan area code
                    region_name = record.get('D7N')  # Metropolitan area name
                    
                    if value_str in [None, "", "-", "..."]:
                        continue
                    
                    value = float(value_str.replace(",", "."))
                    
                    if not period_code or len(period_code) != 6:
                        continue
                    
                    year = int(period_code[:4])
                    month = int(period_code[4:6])
                    
                    if not (1900 <= year <= 2100 and 1 <= month <= 12):
                        continue
                    
                    data_points.append(InflationDataPoint(
                        date=date(year, month, 1),
                        value=value,
                        source=DataSource.IBGE_SIDRA,
                        metadata={
                            "indicator_type": "regional",
                            "region_code": region_code,
                            "region_name": region_name
                        }
                    ))
                    
                except (ValueError, KeyError, TypeError):
                    continue
            
            if not data_points:
                raise IBGESIDRAError("No valid regional data points", source="IBGE SIDRA Regional")
            
            console.print(f"[green]Successfully parsed {len(data_points)} IBGE regional data points[/green]")
            
            return InflationDataset(
                data_points=data_points,
                source=DataSource.IBGE_SIDRA,
                metadata={"dataset_type": "regional", "table": table}
            )
            
        except Exception as e:
            if isinstance(e, IBGESIDRAError):
                raise
            raise IBGESIDRAError(f"IBGE regional data error: {e}", source="IBGE SIDRA Regional")
    
    async def fetch_sectoral_data_as_dataframe(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> Optional[pl.DataFrame]:
        """
        Fetch sectoral IPCA data and return as Polars DataFrame.
        
        Converts the sectoral data from InflationDataset to the format expected
        by the sectoral visualizer: DataFrame with date column and sector columns.
        
        Args:
            start_year: Starting year for data
            end_year: Ending year for data
            
        Returns:
            Polars DataFrame with date and sectoral breakdown columns, or None if failed
        """
        try:
            # Fetch raw sectoral data
            sectoral_dataset = await self._fetch_ibge_sectoral_data(start_year, end_year)
            
            if not sectoral_dataset or not sectoral_dataset.data_points:
                console.print("[yellow]No sectoral data available[/yellow]")
                return None
            
            console.print(f"[blue]Converting {len(sectoral_dataset.data_points)} sectoral data points to DataFrame[/blue]")
            
            # Group data by date and sector
            data_by_date = {}
            
            for point in sectoral_dataset.data_points:
                date_key = point.date
                group_name = point.metadata.get('group_name', 'Unknown') if point.metadata else 'Unknown'
                
                # Skip if group_name is None
                if group_name is None:
                    continue
                
                # Map IBGE group names to standardized sector keys
                sector_key = self._map_ibge_group_to_sector(group_name)
                
                if date_key not in data_by_date:
                    data_by_date[date_key] = {'date': date_key}
                
                data_by_date[date_key][sector_key] = float(point.value)
            
            # Convert to list of records
            records = list(data_by_date.values())
            
            if not records:
                console.print("[yellow]No sectoral records to convert[/yellow]")
                return None
            
            # Create DataFrame
            df = pl.DataFrame(records)
            
            # Ensure date column is properly typed
            df = df.with_columns(pl.col("date").cast(pl.Date))
            
            # Sort by date
            df = df.sort("date")
            
            console.print(f"[green]Successfully created sectoral DataFrame with {df.shape[0]} rows and {df.shape[1]} columns[/green]")
            console.print(f"[blue]Sectoral columns: {[col for col in df.columns if col != 'date']}[/blue]")
            
            return df
            
        except Exception as e:
            console.print(f"[red]Error creating sectoral DataFrame: {e}[/red]")
            return None
    
    async def fetch_regional_data_as_dataframe(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> Optional[pl.DataFrame]:
        """
        Fetch regional IPCA data and return as Polars DataFrame.
        
        Note: Regional IPCA data fetching is currently under development.
        The IBGE SIDRA API structure for regional metropolitan area data
        needs further investigation.
        
        Args:
            start_year: Starting year for data
            end_year: Ending year for data
            
        Returns:
            None (regional data fetching temporarily disabled)
        """
        console.print("[yellow]Regional data fetching is temporarily disabled while debugging IBGE API endpoints[/yellow]")
        return None
    
    def _map_ibge_group_to_sector(self, group_name: str) -> str:
        """Map IBGE sectoral group names to standardized sector keys."""
        # Handle None or empty group_name
        if not group_name:
            return 'unknown_sector'
            
        # Normalize the group name for matching
        normalized = group_name.lower().strip()
        
        # IBGE sectoral group mappings to SectoralRegionalVisualizer keys
        mappings = {
            # Food and Beverages
            'alimentação e bebidas': 'food_beverages',
            'alimentos e bebidas': 'food_beverages',
            'alimentação': 'food_beverages',
            
            # Housing
            'habitação': 'housing',
            'moradia': 'housing',
            'casa': 'housing',
            
            # Household Articles
            'artigos de residência': 'household_articles',
            'residência': 'household_articles',
            'mobiliário': 'household_articles',
            
            # Clothing
            'vestuário': 'clothing',
            'roupas': 'clothing',
            
            # Transportation
            'transportes': 'transportation',
            'transporte': 'transportation',
            
            # Health and Personal Care
            'saúde e cuidados pessoais': 'health_personal_care',
            'cuidados pessoais': 'health_personal_care',
            'saúde': 'health_personal_care',
            
            # Personal Expenses
            'despesas pessoais': 'personal_expenses',
            'gastos pessoais': 'personal_expenses',
            
            # Education
            'educação': 'education',
            'ensino': 'education',
            
            # Communication
            'comunicação': 'communication',
            'comunicações': 'communication',
        }
        
        # Try exact matches first
        for ibge_name, sector_key in mappings.items():
            if ibge_name in normalized:
                return sector_key
        
        # If no match found, create a key from the group name
        # Remove special characters and convert to snake_case
        sanitized = ''.join(c.lower() if c.isalnum() else '_' for c in group_name)
        # Remove multiple underscores
        sanitized = '_'.join(filter(None, sanitized.split('_')))
        
        return sanitized if sanitized else 'unknown_sector'
    
    def _map_ibge_region_to_key(self, region_name: str) -> str:
        """Map IBGE regional names to standardized region keys."""
        # Normalize the region name for matching
        normalized = region_name.lower().strip()
        
        # IBGE regional mappings to SectoralRegionalVisualizer keys
        mappings = {
            'belém': 'belem',
            'fortaleza': 'fortaleza', 
            'recife': 'recife',
            'salvador': 'salvador',
            'belo horizonte': 'belo_horizonte',
            'vitória': 'vitoria',
            'rio de janeiro': 'rio_de_janeiro',
            'são paulo': 'sao_paulo',
            'curitiba': 'curitiba',
            'porto alegre': 'porto_alegre',
            'campo grande': 'campo_grande',
            'cuiabá': 'cuiaba',
            'brasília': 'brasilia',
            'goiânia': 'goiania',
            'aracaju': 'aracaju',
            
            # Alternative names
            'região metropolitana de belém': 'belem',
            'região metropolitana de fortaleza': 'fortaleza',
            'região metropolitana de recife': 'recife',
            'região metropolitana de salvador': 'salvador',
            'região metropolitana de belo horizonte': 'belo_horizonte',
            'região metropolitana de vitória': 'vitoria',
            'região metropolitana do rio de janeiro': 'rio_de_janeiro',
            'região metropolitana de são paulo': 'sao_paulo',
            'região metropolitana de curitiba': 'curitiba',
            'região metropolitana de porto alegre': 'porto_alegre',
            'região metropolitana de campo grande': 'campo_grande',
            'região metropolitana de cuiabá': 'cuiaba',
            'região metropolitana de brasília': 'brasilia',
            'região metropolitana de goiânia': 'goiania',
            'região metropolitana de aracaju': 'aracaju',
        }
        
        # Try exact matches first
        for ibge_name, region_key in mappings.items():
            if ibge_name in normalized:
                return region_key
        
        # If no match found, create a key from the region name
        # Remove special characters and convert to snake_case
        sanitized = ''.join(c.lower() if c.isalnum() else '_' for c in region_name)
        # Remove multiple underscores
        sanitized = '_'.join(filter(None, sanitized.split('_')))
        
        return sanitized if sanitized else 'unknown_region'

    async def save_raw_data(
        self,
        dataset: InflationDataset, 
        filename: Optional[str] = None
    ) -> Path:
        """
        Save raw data to disk.
        
        Args:
            dataset: Dataset to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"brazil_inflation_{dataset.source.value}_{timestamp}.json"
        
        filepath = self.settings.storage.raw_dir / filename
        
        # Convert dataset to dict for JSON serialization
        data_dict = {
            "metadata": {
                "source": dataset.source.value,
                "retrieved_at": dataset.retrieved_at.isoformat(),
                "total_observations": dataset.total_observations,
                "date_range": [d.isoformat() for d in dataset.date_range],
            },
            "data": [
                {
                    "date": dp.date.isoformat(),
                    "value": dp.value,
                    "source": dp.source.value,
                    "period": dp.period.value,
                }
                for dp in dataset.data_points
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]Saved raw data to {filepath}[/green]")
        return filepath


# Alias for backward compatibility
BrazilInflationDataFetcher = BrazilInflationFetcher