"""Configuration management for Brazil Inflation Data Fetcher.

Provides comprehensive environment-based configuration with validation,
optimized for Polars-based data processing and dual-period analysis.
"""

import os
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentType(str, Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class DataSourceConfig(BaseModel):
    """Configuration for data source priorities and settings."""
    
    start_year: int = Field(default=1980, description="Start year for data collection")
    end_year: int = Field(default=2024, description="End year for data collection")
    hyperinflation_threshold: float = Field(
        default=50.0, 
        description="Monthly inflation rate threshold for hyperinflation validation"
    )
    plano_real_year: int = Field(
        default=1994, 
        description="Year of Plano Real implementation (major breakpoint)"
    )
    
    @validator("start_year")
    def validate_start_year(cls, v: int) -> int:
        """Validate start year is reasonable."""
        if v < 1900 or v > 2030:
            raise ValueError(f"Start year {v} must be between 1900 and 2030")
        return v
    
    @validator("end_year")
    def validate_end_year(cls, v: int) -> int:
        """Validate end year is reasonable."""
        if v < 1900 or v > 2030:
            raise ValueError(f"End year {v} must be between 1900 and 2030")
        return v
    
    @validator("hyperinflation_threshold")
    def validate_hyperinflation_threshold(cls, v: float) -> float:
        """Validate hyperinflation threshold."""
        if v <= 0 or v > 1000:
            raise ValueError(f"Hyperinflation threshold {v}% must be between 0 and 1000")
        return v


class APIConfig(BaseModel):
    """API configuration for external data sources."""
    
    fred_api_key: Optional[str] = Field(default=None, description="FRED API key")
    fred_series_id: str = Field(
        default="FPCPITOTLZGBRA", 
        description="FRED series ID for Brazil inflation"
    )
    
    world_bank_base_url: str = Field(
        default="https://api.worldbank.org/v2",
        description="World Bank API base URL"
    )
    world_bank_indicator: str = Field(
        default="FP.CPI.TOTL.ZG",
        description="World Bank indicator for Brazil inflation"
    )
    
    ibge_sidra_table: int = Field(
        default=1737,
        description="IBGE SIDRA table number for IPCA data"
    )
    
    # Request settings
    timeout_seconds: int = Field(default=30, description="HTTP request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    
    @validator("timeout_seconds")
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is reasonable."""
        if v <= 0 or v > 300:
            raise ValueError(f"Timeout {v} seconds must be between 1 and 300")
        return v
    
    @validator("max_retries")
    def validate_max_retries(cls, v: int) -> int:
        """Validate max retries is reasonable."""
        if v < 0 or v > 10:
            raise ValueError(f"Max retries {v} must be between 0 and 10")
        return v


class StorageConfig(BaseModel):
    """Configuration for data storage and processing."""
    
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory") 
    processed_dir: Path = Field(
        default=Path("data/processed"), 
        description="Processed data directory"
    )
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    
    # File formats
    raw_format: str = Field(default="json", description="Raw data storage format")
    processed_format: str = Field(default="parquet", description="Processed data format")
    parquet_compression: str = Field(
        default="snappy", 
        description="Parquet compression algorithm"
    )
    
    # Data processing
    use_lazy_evaluation: bool = Field(
        default=True, 
        description="Use Polars lazy evaluation"
    )
    optimize_dtypes: bool = Field(
        default=True, 
        description="Optimize data types for memory efficiency"
    )
    
    def __post_init__(self):
        """Create directories after initialization."""
        for dir_attr in ["data_dir", "raw_dir", "processed_dir", "output_dir"]:
            dir_path = getattr(self, dir_attr)
            if isinstance(dir_path, Path):
                dir_path.mkdir(parents=True, exist_ok=True)


class VisualizationConfig(BaseModel):
    """Configuration for visualization and charting."""
    
    # Chart dimensions and output
    default_width: int = Field(default=1200, description="Default chart width in pixels")
    default_height: int = Field(default=800, description="Default chart height in pixels")
    dpi: int = Field(default=300, description="Chart resolution (DPI)")
    output_formats: List[str] = Field(default=["png", "svg", "html"], description="Output formats")
    
    # Dual-period visualization
    use_log_scale_hyperinflation: bool = Field(
        default=True, 
        description="Use logarithmic scale for hyperinflation period"
    )
    hyperinflation_color_scheme: str = Field(default="reds", description="Color scheme for hyperinflation period")
    modern_color_scheme: str = Field(default="blues", description="Color scheme for modern period")
    
    # Styling
    style_theme: str = Field(default="seaborn", description="Chart styling theme")
    font_family: str = Field(default="Arial", description="Font family for charts")
    title_size: int = Field(default=16, description="Chart title font size")
    axis_label_size: int = Field(default=12, description="Axis label font size")
    
    @validator("output_formats")
    def validate_output_formats(cls, v: List[str]) -> List[str]:
        """Validate output formats are supported."""
        valid_formats = {"png", "svg", "pdf", "html", "jpg", "jpeg", "webp"}
        for fmt in v:
            if fmt.lower() not in valid_formats:
                raise ValueError(f"Unsupported output format: {fmt}")
        return [fmt.lower() for fmt in v]


class HTTPConfig(BaseModel):
    """HTTP client configuration."""
    
    max_connections: int = Field(default=10, description="Maximum HTTP connections")
    max_keepalive: int = Field(default=5, description="Maximum keep-alive connections")
    user_agent: str = Field(default="BrazilInflationAnalyzer/1.0", description="HTTP user agent")
    
    @validator("max_connections")
    def validate_max_connections(cls, v: int) -> int:
        """Validate connection limits."""
        if v <= 0 or v > 100:
            raise ValueError(f"Max connections {v} must be between 1 and 100")
        return v


class PolarsConfig(BaseModel):
    """Polars-specific performance configuration."""
    
    thread_pool_size: int = Field(default=0, description="Thread pool size (0 = auto-detect)")
    streaming_chunk_size: int = Field(default=1000000, description="Streaming chunk size")
    
    @validator("streaming_chunk_size")
    def validate_chunk_size(cls, v: int) -> int:
        """Validate streaming chunk size."""
        if v <= 0 or v > 10_000_000:
            raise ValueError(f"Chunk size {v} must be between 1 and 10,000,000")
        return v


class CacheConfig(BaseModel):
    """Caching configuration."""
    
    enable_cache: bool = Field(default=True, description="Enable data caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    @validator("cache_ttl_seconds")
    def validate_cache_ttl(cls, v: int) -> int:
        """Validate cache TTL is reasonable."""
        if v <= 0 or v > 86400 * 7:  # Max 1 week
            raise ValueError(f"Cache TTL {v} seconds must be between 1 and 604800 (1 week)")
        return v


class Settings(BaseSettings):
    """Main application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=[
            ".env",  # Try current directory first
            Path(__file__).parent.parent.parent / ".env",  # Try project root
        ],
        env_file_encoding="utf-8", 
        case_sensitive=False,
        env_prefix="BRAZIL_INFLATION_",
        env_nested_delimiter="__",
        validate_default=True,
        extra="ignore",  # Ignore extra fields to handle FRED_API_KEY
    )
    
    # API configuration
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Data source configuration  
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    
    # Storage configuration
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Visualization configuration
    viz: VisualizationConfig = Field(default_factory=VisualizationConfig)
    
    # HTTP configuration
    http: HTTPConfig = Field(default_factory=HTTPConfig)
    
    # Polars configuration
    polars: PolarsConfig = Field(default_factory=PolarsConfig)
    
    # Cache configuration
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    # Environment settings
    environment: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT, description="Environment type")
    debug: bool = Field(default=False, description="Enable debug mode")
    profile_performance: bool = Field(default=False, description="Enable performance profiling")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(
        default=None, 
        description="Log file path (None for console only)"
    )
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is supported."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level '{v}' must be one of {valid_levels}")
        return v.upper()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_directories()
        self._load_api_keys()
        self._configure_polars()
        self._setup_logging()
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [
            self.storage.data_dir,
            self.storage.raw_dir, 
            self.storage.processed_dir,
            self.storage.output_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_api_keys(self) -> None:
        """Load API keys from environment variables."""
        # Load .env file manually to ensure it's available
        from dotenv import load_dotenv
        
        # Try to load from multiple locations
        env_files = [
            Path.cwd() / ".env",
            Path(__file__).parent.parent.parent / ".env",
        ]
        
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(env_file)
                break
        
        # Support both prefixed and non-prefixed FRED API key
        fred_key = (
            os.getenv("FRED_API_KEY") or 
            os.getenv("BRAZIL_INFLATION_API__FRED_API_KEY") or
            os.getenv("BRAZIL_INFLATION_FRED_API_KEY")
        )
        if fred_key:
            self.api.fred_api_key = fred_key
    
    def _configure_polars(self) -> None:
        """Configure Polars with performance settings."""
        try:
            import polars as pl
            
            # Set thread pool size if specified
            if self.polars.thread_pool_size > 0:
                pl.Config.set_tbl_rows(20)  # Default table display rows
                
        except ImportError:
            pass  # Polars not available
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        level = getattr(logging, self.log_level)
        
        # Configure basic logging
        handlers = [logging.StreamHandler()]
        
        if self.log_file:
            handlers.append(logging.FileHandler(self.log_file))
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # Override existing configuration
        )
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate complete configuration and return validation report."""
        issues = []
        warnings = []
        
        # API validation
        if not self.api.fred_api_key:
            warnings.append("FRED API key not configured - will use fallback sources")
        
        # Environment validation
        if self.environment == EnvironmentType.PRODUCTION and self.debug:
            warnings.append("Debug mode enabled in production environment")
        
        # Performance validation
        if self.polars.thread_pool_size > os.cpu_count():
            warnings.append(f"Polars thread pool size ({self.polars.thread_pool_size}) exceeds CPU count ({os.cpu_count()})")
        
        # Directory validation
        for dir_name, dir_path in [
            ("data_dir", self.storage.data_dir),
            ("raw_dir", self.storage.raw_dir),
            ("processed_dir", self.storage.processed_dir),
            ("output_dir", self.storage.output_dir),
        ]:
            if not dir_path.exists():
                issues.append(f"{dir_name} does not exist: {dir_path}")
            elif not os.access(dir_path, os.W_OK):
                issues.append(f"{dir_name} is not writable: {dir_path}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "environment": self.environment.value,
            "debug_mode": self.debug,
            "api_configured": bool(self.api.fred_api_key),
        }
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective configuration as a dictionary."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "data_source": self.data_source.dict(),
            "api": {
                **self.api.dict(),
                "fred_api_key": "***" if self.api.fred_api_key else None,  # Mask sensitive data
            },
            "storage": self.storage.dict(),
            "viz": self.viz.dict(),
            "http": self.http.dict(),
            "polars": self.polars.dict(),
            "cache": self.cache.dict(),
            "log_level": self.log_level,
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def load_settings_from_file(config_file: Path) -> Settings:
    """Load settings from a specific configuration file."""
    return Settings(_env_file=config_file)


def validate_environment_setup() -> Dict[str, Any]:
    """Validate the complete environment setup."""
    settings = get_settings()
    validation_report = settings.validate_configuration()
    
    # Additional system checks
    try:
        import polars as pl
        validation_report["polars_version"] = pl.__version__
        validation_report["polars_available"] = True
    except ImportError:
        validation_report["polars_available"] = False
        validation_report["issues"].append("Polars not available - install with: pip install polars")
    
    try:
        import matplotlib
        validation_report["matplotlib_available"] = True
    except ImportError:
        validation_report["matplotlib_available"] = False
        validation_report["warnings"].append("Matplotlib not available - visualization features limited")
    
    return validation_report


def create_env_template(output_path: Path) -> None:
    """Create an .env.example template file."""
    template_content = '''# ====================================================================
# BRAZIL INFLATION DATA VISUALIZATION PROJECT - CONFIGURATION TEMPLATE
# ====================================================================
# 
# Copy this file to .env and customize the values for your environment.
# All settings are optional and have reasonable defaults.
#

# ====================================================================
# API KEYS & CREDENTIALS
# ====================================================================

# FRED (Federal Reserve Economic Data) API Key
# Get your key from: https://fred.stlouisfed.org/docs/api/api_key.html
# FRED_API_KEY=your_fred_api_key_here

# ====================================================================
# ENVIRONMENT CONFIGURATION
# ====================================================================

# Environment mode (development, production, testing)
BRAZIL_INFLATION_ENVIRONMENT=development

# Debug settings
BRAZIL_INFLATION_DEBUG=true

# ====================================================================
# DATA COLLECTION SETTINGS
# ====================================================================

# Data collection period
# BRAZIL_INFLATION_DATA_SOURCE__START_YEAR=1980
# BRAZIL_INFLATION_DATA_SOURCE__END_YEAR=2024

# Hyperinflation analysis parameters  
# BRAZIL_INFLATION_DATA_SOURCE__HYPERINFLATION_THRESHOLD=50.0
# BRAZIL_INFLATION_DATA_SOURCE__PLANO_REAL_YEAR=1994

# ====================================================================
# PERFORMANCE OPTIMIZATION
# ====================================================================

# HTTP settings
# BRAZIL_INFLATION_HTTP__MAX_CONNECTIONS=10
# BRAZIL_INFLATION_HTTP__TIMEOUT_SECONDS=30

# Polars settings
# BRAZIL_INFLATION_POLARS__THREAD_POOL_SIZE=0  # 0 = auto-detect

# Cache settings
# BRAZIL_INFLATION_CACHE__ENABLE_CACHE=true
# BRAZIL_INFLATION_CACHE__CACHE_TTL_SECONDS=3600

# ====================================================================
# LOGGING & OUTPUT
# ====================================================================

# Logging configuration
# BRAZIL_INFLATION_LOG_LEVEL=INFO
# BRAZIL_INFLATION_LOG_FILE=brazil_inflation.log

# Visualization settings
# BRAZIL_INFLATION_VIZ__DEFAULT_WIDTH=1200
# BRAZIL_INFLATION_VIZ__DEFAULT_HEIGHT=800
# BRAZIL_INFLATION_VIZ__OUTPUT_FORMATS=png,svg,html
'''
    
    output_path.write_text(template_content)


def print_configuration_help() -> None:
    """Print help information about configuration options."""
    help_text = """
Brazil Inflation Data - Configuration Help

ENVIRONMENT VARIABLES:

  Required:
  ---------
  None! All settings have defaults.

  Optional API Keys:
  -----------------
  FRED_API_KEY                 - Federal Reserve Economic Data API key
                                 Get from: https://fred.stlouisfed.org/docs/api/api_key.html

  Common Settings:
  ---------------
  BRAZIL_INFLATION_ENVIRONMENT - development|production|testing (default: development)
  BRAZIL_INFLATION_DEBUG       - true|false (default: true in development)
  BRAZIL_INFLATION_LOG_LEVEL   - DEBUG|INFO|WARNING|ERROR|CRITICAL (default: INFO)

  Data Collection:
  ---------------
  BRAZIL_INFLATION_DATA_SOURCE__START_YEAR            - Start year (default: 1980)
  BRAZIL_INFLATION_DATA_SOURCE__END_YEAR              - End year (default: 2024)
  BRAZIL_INFLATION_DATA_SOURCE__HYPERINFLATION_THRESHOLD - Threshold % (default: 50.0)

  Performance:
  -----------
  BRAZIL_INFLATION_POLARS__THREAD_POOL_SIZE          - Thread count (default: auto)
  BRAZIL_INFLATION_HTTP__MAX_CONNECTIONS             - Max HTTP connections (default: 10)
  BRAZIL_INFLATION_CACHE__ENABLE_CACHE               - Enable caching (default: true)

  Visualization:
  -------------
  BRAZIL_INFLATION_VIZ__DEFAULT_WIDTH                - Chart width (default: 1200)
  BRAZIL_INFLATION_VIZ__DEFAULT_HEIGHT               - Chart height (default: 800)
  BRAZIL_INFLATION_VIZ__OUTPUT_FORMATS               - Formats: png,svg,html (default)

TIP: Use nested delimiter '__' for hierarchical settings
TIP: Run 'brazil-inflation info' to see current configuration
TIP: All directory paths are created automatically if they don't exist

Create .env.example template with: brazil-inflation config create-template
"""
    print(help_text)