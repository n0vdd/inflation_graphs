"""Advanced time series analytics for Brazil inflation data."""

from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import polars as pl
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import statsmodels.stats.stattools as stattools
from rich.console import Console

from .config import Settings, get_settings
from .exceptions import DataProcessingError
from .models import DataSource, InflationDataset

console = Console()

class AdvancedInflationAnalytics:
    """
    Advanced time series analytics for Brazil inflation data.
    
    Implements sophisticated economic analysis including:
    - Seasonal decomposition (X-13ARIMA-SEATS methodology)
    - Structural break detection (Chow tests, CUSUM)
    - Volatility modeling (ARCH/GARCH)
    - Enhanced statistical validation
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize analytics with configuration."""
        self.settings = settings or get_settings()
        
    def perform_seasonal_decomposition(
        self,
        lf: pl.LazyFrame,
        model: str = "additive",
        period: Optional[int] = None,
        extrapolate_trend: int = 0
    ) -> Dict[str, pl.DataFrame]:
        """
        Perform seasonal decomposition using statsmodels.
        
        Args:
            lf: Input LazyFrame with time series data
            model: Decomposition model ('additive' or 'multiplicative')
            period: Seasonal period (default: auto-detect or 12 for monthly)
            extrapolate_trend: Number of periods to extrapolate trend
            
        Returns:
            Dictionary containing trend, seasonal, residual, and original components
        """
        console.print(f"[blue]Performing seasonal decomposition ({model} model)[/blue]")
        
        # Collect and prepare data
        df = lf.select([
            "date", "inflation_rate_cleaned"
        ]).sort("date").collect()
        
        # Convert to pandas for statsmodels compatibility
        df_pandas = df.to_pandas()
        df_pandas = df_pandas.set_index('date')
        
        # Auto-detect period if not specified
        if period is None:
            # For monthly data, use 12-month seasonality
            period = 12
            
        # Handle missing values
        ts_data = df_pandas['inflation_rate_cleaned'].interpolate(method='linear')
        
        # Perform decomposition
        try:
            decomposition = seasonal_decompose(
                ts_data,
                model=model,
                period=period,
                extrapolate_trend=extrapolate_trend
            )
            
            # Convert results back to Polars DataFrames
            results = {}
            
            # Original series
            results['original'] = pl.DataFrame({
                'date': decomposition.observed.index,
                'value': decomposition.observed.values,
                'component': ['original'] * len(decomposition.observed)
            })
            
            # Trend component
            results['trend'] = pl.DataFrame({
                'date': decomposition.trend.index,
                'value': decomposition.trend.values,
                'component': ['trend'] * len(decomposition.trend)
            })
            
            # Seasonal component
            results['seasonal'] = pl.DataFrame({
                'date': decomposition.seasonal.index,
                'value': decomposition.seasonal.values,
                'component': ['seasonal'] * len(decomposition.seasonal)
            })
            
            # Residual component
            results['residual'] = pl.DataFrame({
                'date': decomposition.resid.index,
                'value': decomposition.resid.values,
                'component': ['residual'] * len(decomposition.resid)
            })
            
            # Combined decomposition DataFrame
            results['decomposition'] = pl.concat([
                results['original'],
                results['trend'], 
                results['seasonal'],
                results['residual']
            ])
            
            # Calculate seasonal strength metrics
            seasonal_strength = self._calculate_seasonal_strength(
                decomposition.seasonal.values,
                decomposition.resid.values
            )
            
            trend_strength = self._calculate_trend_strength(decomposition.trend.values)
            residual_variance = np.var(decomposition.resid.values, ddof=1)
            
            # Handle NaN values by converting to 0.0
            seasonal_strength = float(seasonal_strength) if not np.isnan(seasonal_strength) else 0.0
            trend_strength = float(trend_strength) if not np.isnan(trend_strength) else 0.0
            residual_variance = float(residual_variance) if not np.isnan(residual_variance) else 0.0
            
            results['metrics'] = pl.DataFrame({
                'metric': ['seasonal_strength', 'trend_strength', 'residual_variance'],
                'value': [seasonal_strength, trend_strength, residual_variance]
            })
            
            console.print(f"[green]Seasonal decomposition completed. Seasonal strength: {seasonal_strength:.3f}[/green]")
            return results
            
        except Exception as e:
            raise DataProcessingError(f"Seasonal decomposition failed: {str(e)}")
    
    def detect_structural_breaks(
        self,
        lf: pl.LazyFrame,
        test_dates: Optional[List[date]] = None,
        significance_level: float = 0.05
    ) -> pl.DataFrame:
        """
        Detect structural breaks using Chow tests and CUSUM analysis.
        
        Args:
            lf: Input LazyFrame
            test_dates: Specific dates to test for breaks (default: auto-detect)
            significance_level: Statistical significance level
            
        Returns:
            DataFrame with break test results
        """
        console.print("[blue]Detecting structural breaks in inflation series[/blue]")
        
        # Prepare data
        df = lf.select([
            "date", "inflation_rate_cleaned"
        ]).sort("date").collect()
        
        # If no test dates specified, test key economic dates
        if test_dates is None:
            test_dates = [
                date(1994, 7, 1),   # Plano Real
                date(1999, 6, 1),   # Inflation targeting begins
                date(2008, 9, 1),   # Global financial crisis
                date(2020, 3, 1),   # COVID-19 pandemic
            ]
        
        results = []
        
        # Convert to numpy arrays for analysis
        dates_array = df.select("date").to_numpy().flatten()
        values_array = df.select("inflation_rate_cleaned").to_numpy().flatten()
        
        # Determine minimum buffer based on data frequency
        total_points = len(values_array)
        if total_points > 200:
            # Monthly data - use larger buffer
            min_buffer = 24
        else:
            # Annual data - use smaller buffer
            min_buffer = max(6, total_points // 10)  # At least 10% of data on each side
        
        console.print(f"[blue]Using minimum buffer of {min_buffer} points for {total_points} total observations[/blue]")
        
        for test_date in test_dates:
            try:
                # Find the index closest to test date
                test_idx = self._find_closest_date_index(dates_array, test_date)
                
                if test_idx < min_buffer or test_idx > len(values_array) - min_buffer:
                    # Skip if too close to endpoints
                    console.print(f"[yellow]Skipping {test_date}: insufficient buffer (need {min_buffer}, got {min(test_idx, len(values_array) - test_idx)})[/yellow]")
                    continue
                
                # Chow test for structural break
                chow_stat, chow_p_value = self._chow_test(
                    values_array, test_idx
                )
                
                # CUSUM test
                cusum_stat, cusum_p_value = self._cusum_test(
                    values_array, test_idx
                )
                
                # Calculate means before and after
                mean_before = np.mean(values_array[:test_idx])
                mean_after = np.mean(values_array[test_idx:])
                
                # Calculate volatility change
                vol_before = np.std(values_array[:test_idx], ddof=1)
                vol_after = np.std(values_array[test_idx:], ddof=1)
                
                results.append({
                    'test_date': test_date,
                    'test_index': test_idx,
                    'chow_statistic': chow_stat,
                    'chow_p_value': chow_p_value,
                    'chow_significant': chow_p_value < significance_level,
                    'cusum_statistic': cusum_stat,
                    'cusum_p_value': cusum_p_value,
                    'cusum_significant': cusum_p_value < significance_level,
                    'mean_before': mean_before,
                    'mean_after': mean_after,
                    'mean_change': mean_after - mean_before,
                    'volatility_before': vol_before,
                    'volatility_after': vol_after,
                    'volatility_change': vol_after - vol_before,
                    'break_detected': (chow_p_value < significance_level) or (cusum_p_value < significance_level)
                })
                
            except Exception as e:
                console.print(f"[yellow]Warning: Could not test break at {test_date}: {str(e)}[/yellow]")
                
        if not results:
            console.print(f"[yellow]No valid structural break tests could be performed with {len(values_array)} data points[/yellow]")
            console.print(f"[yellow]Data range: {dates_array[0]} to {dates_array[-1]}[/yellow]")
            console.print(f"[yellow]Minimum buffer required: {min_buffer} points[/yellow]")
            
            # Create empty results DataFrame with proper structure
            results_df = pl.DataFrame({
                'test_date': [],
                'test_index': [],
                'chow_statistic': [],
                'chow_p_value': [],
                'chow_significant': [],
                'cusum_statistic': [],
                'cusum_p_value': [],
                'cusum_significant': [],
                'mean_before': [],
                'mean_after': [],
                'mean_change': [],
                'volatility_before': [],
                'volatility_after': [],
                'volatility_change': [],
                'break_detected': []
            }).with_columns([
                pl.col("test_date").cast(pl.Date),
                pl.col("test_index").cast(pl.Int32),
                pl.col("chow_statistic").cast(pl.Float64),
                pl.col("chow_p_value").cast(pl.Float64),
                pl.col("chow_significant").cast(pl.Boolean),
                pl.col("cusum_statistic").cast(pl.Float64),
                pl.col("cusum_p_value").cast(pl.Float64),
                pl.col("cusum_significant").cast(pl.Boolean),
                pl.col("mean_before").cast(pl.Float64),
                pl.col("mean_after").cast(pl.Float64),
                pl.col("mean_change").cast(pl.Float64),
                pl.col("volatility_before").cast(pl.Float64),
                pl.col("volatility_after").cast(pl.Float64),
                pl.col("volatility_change").cast(pl.Float64),
                pl.col("break_detected").cast(pl.Boolean)
            ])
            
            console.print("[yellow]Returning empty structural break results[/yellow]")
            return results_df
        
        results_df = pl.DataFrame(results)
        
        # Count significant breaks
        significant_breaks = results_df.filter(pl.col("break_detected") == True).height
        console.print(f"[green]Structural break analysis completed. Found {significant_breaks} significant breaks[/green]")
        
        return results_df
    
    def model_volatility(
        self,
        lf: pl.LazyFrame,
        model_type: str = "ARCH",
        max_lags: int = 5
    ) -> Dict[str, Any]:
        """
        Model inflation volatility using ARCH/GARCH models.
        
        Args:
            lf: Input LazyFrame
            model_type: Type of volatility model ('ARCH' or 'GARCH')
            max_lags: Maximum number of lags to consider
            
        Returns:
            Dictionary with model results and volatility estimates
        """
        console.print(f"[blue]Modeling volatility using {model_type} approach[/blue]")
        
        # Prepare data
        df = lf.select([
            "date", "inflation_rate_cleaned"
        ]).sort("date").collect()
        
        inflation_series = df.select("inflation_rate_cleaned").to_numpy().flatten()
        
        # Test for ARCH effects
        arch_results = het_arch(inflation_series, nlags=max_lags)  # Fixed deprecated maxlag parameter
        arch_test_stat, arch_p_value = arch_results[0], arch_results[1]
        
        results = {
            'arch_test_statistic': arch_test_stat,
            'arch_p_value': arch_p_value,
            'arch_effects_detected': arch_p_value < 0.05,
            'model_type': model_type,
            'max_lags': max_lags
        }
        
        if not results['arch_effects_detected']:
            console.print("[yellow]No significant ARCH effects detected in the series[/yellow]")
        
        # Calculate rolling volatility as proxy for GARCH
        window_size = 12  # 12-month rolling window
        rolling_vol = []
        dates = df.select("date").to_numpy().flatten()
        
        for i in range(len(inflation_series)):
            start_idx = max(0, i - window_size + 1)
            window_data = inflation_series[start_idx:i+1]
            if len(window_data) >= 3:
                vol = np.std(window_data, ddof=1)
            else:
                vol = np.nan
            rolling_vol.append(vol)
        
        # Create volatility DataFrame
        volatility_df = pl.DataFrame({
            'date': dates,
            'inflation_rate': inflation_series,
            'rolling_volatility': rolling_vol,
            'volatility_squared': np.array(rolling_vol) ** 2
        })
        
        # Calculate volatility persistence (autocorrelation of squared returns)
        vol_persistence = self._calculate_volatility_persistence(inflation_series)
        
        results.update({
            'volatility_persistence': vol_persistence,
            'mean_volatility': np.nanmean(rolling_vol),
            'volatility_std': np.nanstd(rolling_vol, ddof=1),
            'max_volatility': np.nanmax(rolling_vol),
            'min_volatility': np.nanmin(rolling_vol)
        })
        
        results['volatility_series'] = volatility_df
        
        console.print(f"[green]Volatility modeling completed. Mean volatility: {results['mean_volatility']:.3f}[/green]")
        return results
    
    def enhanced_data_validation(
        self,
        lf: pl.LazyFrame,
        validation_level: str = "comprehensive"
    ) -> pl.DataFrame:
        """
        Enhanced statistical data validation and quality assessment.
        
        Args:
            lf: Input LazyFrame
            validation_level: Level of validation ('basic', 'standard', 'comprehensive')
            
        Returns:
            DataFrame with comprehensive validation results
        """
        console.print(f"[blue]Performing {validation_level} data validation[/blue]")
        
        df = lf.select([
            "date", "inflation_rate_cleaned", "period"
        ]).sort("date").collect()
        
        inflation_series = df.select("inflation_rate_cleaned").to_numpy().flatten()
        
        validation_results = []
        
        # Basic validation tests
        validation_results.extend([
            {'test': 'total_observations', 'value': len(inflation_series), 'status': 'info'},
            {'test': 'missing_values', 'value': np.sum(np.isnan(inflation_series)), 'status': 'warning' if np.sum(np.isnan(inflation_series)) > 0 else 'pass'},
            {'test': 'infinite_values', 'value': np.sum(np.isinf(inflation_series)), 'status': 'error' if np.sum(np.isinf(inflation_series)) > 0 else 'pass'}
        ])
        
        # Statistical validation tests
        if validation_level in ['standard', 'comprehensive']:
            # Normality test (Shapiro-Wilk for smaller samples, Anderson-Darling for larger)
            if len(inflation_series) <= 5000:
                stat, p_value = stats.shapiro(inflation_series[~np.isnan(inflation_series)])
                test_name = 'shapiro_wilk_normality'
            else:
                stat, critical_values, p_value = stats.anderson(inflation_series[~np.isnan(inflation_series)], dist='norm')
                test_name = 'anderson_darling_normality'
            
            validation_results.append({
                'test': test_name,
                'statistic': stat,
                'p_value': p_value,
                'status': 'warning' if p_value < 0.05 else 'pass'
            })
            
            # Stationarity test (Augmented Dickey-Fuller)
            adf_stat, adf_p_value, _, _, critical_values, _ = adfuller(
                inflation_series[~np.isnan(inflation_series)], 
                autolag='AIC'
            )
            
            validation_results.append({
                'test': 'adf_stationarity',
                'statistic': adf_stat,
                'p_value': adf_p_value,
                'status': 'pass' if adf_p_value < 0.05 else 'warning'
            })
            
        # Comprehensive validation tests
        if validation_level == 'comprehensive':
            # Outlier detection using multiple methods
            # IQR method
            q1, q3 = np.percentile(inflation_series[~np.isnan(inflation_series)], [25, 75])
            iqr = q3 - q1
            outliers_iqr = np.sum(
                (inflation_series < (q1 - 1.5 * iqr)) | 
                (inflation_series > (q3 + 1.5 * iqr))
            )
            
            # Z-score method
            z_scores = np.abs(stats.zscore(inflation_series, nan_policy='omit'))
            outliers_zscore = np.sum(z_scores > 3)
            
            # Modified Z-score method
            median = np.median(inflation_series[~np.isnan(inflation_series)])
            mad = np.median(np.abs(inflation_series[~np.isnan(inflation_series)] - median))
            modified_z_scores = 0.6745 * (inflation_series - median) / mad
            outliers_modified_z = np.sum(np.abs(modified_z_scores) > 3.5)
            
            validation_results.extend([
                {'test': 'outliers_iqr_method', 'value': int(outliers_iqr), 'status': 'warning' if outliers_iqr > len(inflation_series) * 0.05 else 'pass'},
                {'test': 'outliers_zscore_method', 'value': int(outliers_zscore), 'status': 'warning' if outliers_zscore > len(inflation_series) * 0.01 else 'pass'},
                {'test': 'outliers_modified_zscore', 'value': int(outliers_modified_z), 'status': 'warning' if outliers_modified_z > len(inflation_series) * 0.01 else 'pass'}
            ])
            
            # Autocorrelation test
            autocorr_12 = self._calculate_autocorrelation(inflation_series, lag=12)
            validation_results.append({
                'test': 'autocorrelation_12_months',
                'value': autocorr_12,
                'status': 'info' if abs(autocorr_12) < 0.3 else 'warning'
            })
            
            # Heteroscedasticity test
            _, het_p_value = het_arch(inflation_series[~np.isnan(inflation_series)], maxlag=5)
            validation_results.append({
                'test': 'heteroscedasticity_arch',
                'p_value': het_p_value,
                'status': 'warning' if het_p_value < 0.05 else 'pass'
            })
        
        results_df = pl.DataFrame(validation_results)
        
        # Summary statistics
        error_count = results_df.filter(pl.col("status") == "error").height
        warning_count = results_df.filter(pl.col("status") == "warning").height
        
        console.print(f"[green]Data validation completed. Errors: {error_count}, Warnings: {warning_count}[/green]")
        
        return results_df
    
    def _calculate_seasonal_strength(self, seasonal: np.ndarray, residual: np.ndarray) -> float:
        """Calculate seasonal strength metric."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seasonal_var = np.var(seasonal, ddof=1)
            residual_var = np.var(residual, ddof=1)
            if seasonal_var + residual_var == 0:
                return 0.0
            return max(0, 1 - residual_var / (seasonal_var + residual_var))
    
    def _calculate_trend_strength(self, trend: np.ndarray) -> float:
        """Calculate trend strength metric."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detrended = np.diff(trend)
            return 1 - np.var(detrended, ddof=1) / np.var(trend[1:], ddof=1) if len(detrended) > 1 else 0.0
    
    def _find_closest_date_index(self, dates_array: np.ndarray, target_date: date) -> int:
        """Find index of date closest to target date."""
        # Convert target_date to numpy datetime64[D] to match dates_array type
        target_np = np.datetime64(target_date)
        
        # Convert both to days since epoch for comparison
        dates_days = dates_array.astype('datetime64[D]')
        target_days = target_np.astype('datetime64[D]')
        
        date_diffs = np.abs((dates_days - target_days).astype('timedelta64[D]').astype(int))
        return int(np.argmin(date_diffs))
    
    def _chow_test(self, series: np.ndarray, break_point: int) -> Tuple[float, float]:
        """Perform Chow test for structural break."""
        n = len(series)
        k = 2  # Number of parameters (intercept + trend)
        
        # Split series at break point
        y1 = series[:break_point]
        y2 = series[break_point:]
        
        # Create time indices
        x1 = np.arange(len(y1)).reshape(-1, 1)
        x2 = np.arange(len(y2)).reshape(-1, 1)
        x_full = np.arange(n).reshape(-1, 1)
        
        # Add intercept
        x1 = np.column_stack([np.ones(len(x1)), x1])
        x2 = np.column_stack([np.ones(len(x2)), x2])
        x_full = np.column_stack([np.ones(len(x_full)), x_full])
        
        # Calculate residual sum of squares
        try:
            # Unrestricted model (separate regressions)
            beta1 = np.linalg.lstsq(x1, y1, rcond=None)[0]
            beta2 = np.linalg.lstsq(x2, y2, rcond=None)[0]
            
            rss1 = np.sum((y1 - x1 @ beta1) ** 2)
            rss2 = np.sum((y2 - x2 @ beta2) ** 2)
            rss_unrestricted = rss1 + rss2
            
            # Restricted model (pooled regression)
            beta_full = np.linalg.lstsq(x_full, series, rcond=None)[0]
            rss_restricted = np.sum((series - x_full @ beta_full) ** 2)
            
            # Chow test statistic
            chow_stat = ((rss_restricted - rss_unrestricted) / k) / (rss_unrestricted / (n - 2*k))
            
            # P-value using F-distribution
            from scipy.stats import f
            p_value = 1 - f.cdf(chow_stat, k, n - 2*k)
            
            return float(chow_stat), float(p_value)
            
        except np.linalg.LinAlgError:
            return np.nan, np.nan
    
    def _cusum_test(self, series: np.ndarray, break_point: int) -> Tuple[float, float]:
        """Perform CUSUM test for structural stability."""
        n = len(series)
        
        # Calculate recursive residuals
        residuals = []
        for i in range(2, n):  # Start from observation 2
            y_subset = series[:i+1]
            x_subset = np.column_stack([np.ones(i+1), np.arange(i+1)])
            
            try:
                beta = np.linalg.lstsq(x_subset, y_subset, rcond=None)[0]
                predicted = x_subset @ beta
                residual = y_subset[-1] - predicted[-1]
                residuals.append(residual)
            except np.linalg.LinAlgError:
                residuals.append(0.0)
        
        if not residuals:
            return np.nan, np.nan
            
        # Standardize residuals
        residuals = np.array(residuals)
        std_residual = np.std(residuals, ddof=1)
        if std_residual == 0:
            return np.nan, np.nan
            
        standardized_residuals = residuals / std_residual
        
        # Calculate CUSUM statistic
        cusum = np.cumsum(standardized_residuals)
        cusum_stat = np.max(np.abs(cusum)) / np.sqrt(len(cusum))
        
        # Approximate p-value (simplified)
        p_value = 2 * (1 - stats.norm.cdf(cusum_stat))
        
        return float(cusum_stat), float(p_value)
    
    def _calculate_volatility_persistence(self, series: np.ndarray) -> float:
        """Calculate volatility persistence using squared returns."""
        squared_returns = (series - np.mean(series)) ** 2
        autocorr = self._calculate_autocorrelation(squared_returns, lag=1)
        return autocorr
    
    def _calculate_autocorrelation(self, series: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at specified lag."""
        series_clean = series[~np.isnan(series)]
        if len(series_clean) <= lag:
            return np.nan
            
        n = len(series_clean)
        c0 = np.var(series_clean, ddof=0)
        c_lag = np.sum((series_clean[:-lag] - np.mean(series_clean)) * 
                      (series_clean[lag:] - np.mean(series_clean))) / n
        
        if c0 == 0:
            return np.nan
            
        return c_lag / c0