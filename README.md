# Brazil Inflation Data Analytics

A **Polars-first** data fetching and processing system for analyzing Brazil's inflation data, focusing on the dramatic economic transformation from hyperinflation (pre-1994) to modern monetary stability (post-1994).

## 🎯 Key Features

- **Polars-Centric Architecture**: Leverages Polars' lazy evaluation, optimized data types, and native Parquet integration
- **Dual-Period Analysis**: Specialized handling of hyperinflation era vs. modern economic periods
- **Multi-Source Data Fetching**: FRED API, World Bank API, and IBGE SIDRA with intelligent fallback
- **Efficient Storage**: Parquet-first with compression optimized for analytical workloads
- **Async Data Fetching**: High-performance async/await patterns for data retrieval
- **Type-Optimized**: Int16 for years, Float32 for rates, maximizing memory efficiency

## 🚀 Quick Start

### Installation with uv (Recommended)

```bash
# Install with minimal dependencies (Polars-focused)
uv add brazil-inflation-graphs

# Or install with visualization capabilities
uv add "brazil-inflation-graphs[viz]"

# Full installation with all features
uv add "brazil-inflation-graphs[full]"
```

### Traditional pip Installation

```bash
pip install -e .
# or with extras: pip install -e ".[viz,full]"
```

### Environment Setup

```bash
# Optional: Set FRED API key for primary data source
export FRED_API_KEY="your_fred_api_key_here"

# Or create .env file
echo "FRED_API_KEY=your_key_here" > .env
```

## 📊 Usage Examples

### Command Line Interface

```bash
# Fetch and process data with automatic fallback
brazil-inflation fetch --source fred --start-year 1990

# Process existing data with analytics
brazil-inflation process --views --quality

# System information
brazil-inflation info
```

### Python API - Polars Showcase

```python
import asyncio
from src.data_fetcher import BrazilInflationFetcher
from src.data_processor import BrazilInflationProcessor
from src.config import get_settings

async def analyze_brazil_inflation():
    settings = get_settings()
    
    # 1. Fetch data with async patterns
    async with BrazilInflationFetcher(settings) as fetcher:
        dataset = await fetcher.fetch_best_data(
            start_year=1980, 
            end_year=2024
        )
    
    # 2. Process with Polars lazy evaluation
    processor = BrazilInflationProcessor(settings)
    lf = processor.process_dataset(dataset)
    
    # 3. Dual-period comparison (key insight)
    comparison = processor.create_dual_period_comparison(lf)
    print("Hyperinflation vs Modern Era:")
    print(comparison)
    
    # 4. Advanced Polars operations
    recent_trends = lf.filter(
        pl.col("year") >= 2020
    ).with_columns([
        pl.col("inflation_rate_cleaned").rolling_mean(12).alias("annual_trend")
    ]).select([
        "date", "inflation_rate_cleaned", "annual_trend", "volatility_12m"
    ]).collect()
    
    # 5. Save as optimized Parquet
    processor.save_processed_data(lf, "brazil_inflation_analysis", "parquet")

# Run analysis
asyncio.run(analyze_brazil_inflation())
```

## 🏗️ Architecture

### Polars-First Design Philosophy

This project prioritizes **Polars** over pandas for several key advantages:

- **Lazy Evaluation**: Complex transformations are optimized automatically
- **Memory Efficiency**: Optimized data types (Int16, Float32) reduce memory usage
- **Native Parquet**: Direct integration without external dependencies
- **Performance**: Parallel processing and columnar operations
- **API Consistency**: Expression-based API for readable, maintainable code

### Project Structure

```
brazil-inflation-graphs/
├── src/
│   ├── config.py           # Configuration management
│   ├── data_fetcher.py     # Multi-source async data fetching
│   ├── data_processor.py   # Polars-centric data processing
│   ├── main.py            # CLI application with Typer
│   ├── models.py          # Pydantic data models
│   └── exceptions.py      # Custom exception hierarchy
├── data/
│   ├── raw/              # Original JSON data
│   └── processed/        # Optimized Parquet files
├── output/               # Generated reports and views
├── example_usage.py      # Comprehensive usage examples
└── pyproject.toml        # uv-based dependency management
```

### Data Sources (Priority Order)

1. **FRED API** (Primary): Series `FPCPITOTLZGBRA` - High quality, monthly data
2. **World Bank API** (Fallback): Indicator `FP.CPI.TOTL.ZG` - Annual data
3. **IBGE SIDRA API** (Optional): Table 1737 - Official Brazilian data, monthly

## 📈 Analytical Capabilities

### Dual-Period Analysis

The core architectural decision treats Brazil's economic history as two distinct eras:

- **Hyperinflation Period** (Pre-1994): Requires logarithmic scaling, extreme value handling
- **Modern Era** (Post-1994): Linear analysis, inflation targeting regime

```python
# Example: Compare economic periods
comparison = processor.create_dual_period_comparison(lf)

# Typical results:
# Hyperinflation: Avg 500%+ monthly, extreme volatility
# Modern Era: Avg 0.5% monthly, controlled volatility
```

### Advanced Time Series Features

```python
# Polars window functions for moving averages
lf = lf.with_columns([
    pl.col("inflation_rate").rolling_mean(3).alias("ma_3m"),
    pl.col("inflation_rate").rolling_mean(12).alias("ma_12m"),
    pl.col("inflation_rate").rolling_std(12).alias("volatility_12m"),
])

# Identify crisis periods
crisis_periods = lf.filter(
    pl.col("volatility_12m") > pl.col("volatility_12m").quantile(0.9)
)
```

### Analytical Views

The system creates specialized views for different analytical needs:

- **hyperinflation**: Pre-1994 data with extreme value handling
- **modern**: Post-1994 data with detailed trend analysis
- **annual**: Yearly aggregations for long-term trends
- **quarterly**: Modern era quarterly analysis
- **crisis**: High-volatility periods across all eras

## ⚡ Performance Optimizations

### Polars-Specific Optimizations

```python
# Optimized data types for memory efficiency
df = df.with_columns([
    pl.col("year").cast(pl.Int16),        # Years fit in 16 bits
    pl.col("month").cast(pl.Int8),        # Months fit in 8 bits  
    pl.col("inflation_rate").cast(pl.Float32),  # 32-bit precision sufficient
])

# Lazy evaluation for complex pipelines
result = df.lazy().filter(
    pl.col("year") > 1994
).with_columns([
    pl.col("inflation_rate").rolling_mean(12).alias("ma_12")
]).group_by("year").agg([
    pl.col("inflation_rate").mean().alias("avg_inflation")
]).collect()  # Only execute when needed
```

### Parquet Storage Strategy

```python
# Optimized Parquet settings
lf.sink_parquet(
    output_path,
    compression="snappy",      # Fast compression/decompression
    row_group_size=50000,     # Optimized for analytics
    use_pyarrow=False         # Polars native engine
)
```

## 📊 Data Quality & Validation

### Polars-Native Quality Assessment

```python
# Automated outlier detection using IQR method
q1 = df.select(pl.col("inflation_rate").quantile(0.25)).item()
q3 = df.select(pl.col("inflation_rate").quantile(0.75)).item()
iqr = q3 - q1

outliers = df.filter(
    (pl.col("inflation_rate") < q1 - 1.5 * iqr) |
    (pl.col("inflation_rate") > q3 + 1.5 * iqr)
)
```

### Historical Event Annotations

Key events automatically identified and flagged:

- **1994**: Plano Real implementation (major breakpoint)
- **1999**: Inflation targeting regime begins
- **2008**: Global financial crisis impact
- **2020-2021**: COVID-19 pandemic effects

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
FRED_API_KEY=your_fred_api_key
BRAZIL_INFLATION_START_YEAR=1980
BRAZIL_INFLATION_END_YEAR=2024

# Storage Configuration
BRAZIL_INFLATION_DATA_DIR=./data
BRAZIL_INFLATION_USE_LAZY_EVALUATION=true
BRAZIL_INFLATION_PARQUET_COMPRESSION=snappy
```

### Programmatic Configuration

```python
from src.config import Settings

settings = Settings(
    data_source=DataSourceConfig(
        start_year=1990,
        end_year=2024,
        plano_real_year=1994
    ),
    storage=StorageConfig(
        processed_format="parquet",
        use_lazy_evaluation=True,
        parquet_compression="snappy"
    )
)
```

## 🧪 Development

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd brazil-inflation-graphs

# Install with uv (recommended)
uv sync --dev

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Run Tests

```bash
# Run test suite
pytest

# With coverage
pytest --cov=src --cov-report=html

# Type checking
mypy src/

# Linting
ruff check src/
black --check src/
```

### Run Example

```bash
# Comprehensive example showcasing all features
python example_usage.py
```

## 📈 Expected Results

When working with Brazil inflation data, you'll typically see:

### Hyperinflation Period (Pre-1994)
- Monthly rates: 10-2000%+ 
- Extreme volatility
- Exponential growth patterns
- Requires logarithmic visualization

### Modern Era (Post-1994)
- Monthly rates: -1% to 2% typically
- Controlled volatility
- Inflation targeting effects visible
- Linear analysis appropriate

### Key Economic Events
- **1994 Breakpoint**: Dramatic structural change visible in all metrics
- **2002-2003**: Political transition volatility
- **2008**: Global crisis impact
- **2015-2016**: Political/economic crisis
- **2020-2021**: Pandemic effects

## 🤝 Contributing

Contributions welcome! Focus areas:

- Additional data sources integration
- Enhanced Polars operations
- Performance optimizations
- Analytical methods
- Visualization components

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Polars Development Team** for the exceptional DataFrame library
- **FRED (Federal Reserve Economic Data)** for reliable economic data
- **World Bank** for comprehensive international statistics
- **IBGE** for authoritative Brazilian economic data