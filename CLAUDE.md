# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Python project for visualizing Brazil's inflation data over 30-40 years, focusing on the hyperinflation period (pre-1994) and modern era (post-1994). The project emphasizes the use of Polars for efficient data processing and creates dual-period visualizations to show the dramatic economic transformation after Plano Real in 1994.

## Project Structure
The planned structure follows this pattern:
```
brazil-inflation-viz/
├── data/
│   ├── raw/           # Original downloaded data
│   └── processed/     # Cleaned data files (Parquet format)
├── src/
│   ├── data_fetcher.py    # Download inflation data
│   ├── data_processor.py  # Clean and prepare data with Polars
│   └── visualizer.py      # Create graphs
├── output/
│   └── (generated graphs)
├── main.py            # Main script to run everything
├── requirements.txt   # Dependencies
└── README.md
```

## Key Dependencies
- `polars >= 0.20.0` - Fast dataframe library (primary data processing)
- `matplotlib >= 3.5.0` - Static plots
- `plotly >= 5.0.0` - Interactive plots
- `requests >= 2.25.0` - API calls
- `seaborn >= 0.12.0` - Styling
- `python-dateutil` - Date handling

## Development Commands
Since this is a new project, standard Python commands apply:
```bash
# Install dependencies
pip install -r requirements.txt

# Run main visualization script
python main.py
```

## Architecture and Technical Approach

### Data Processing with Polars
This project prioritizes Polars over pandas for performance:
- Use lazy evaluation for complex data transformations
- Store processed data as Parquet files for fast loading
- Leverage window functions for moving averages and volatility calculations
- Optimize data types (Int16 for years, Float32 for inflation rates)

### Dual-Period Visualization Strategy
The core architectural decision is treating pre-1994 and post-1994 as fundamentally different datasets:
- **Pre-1994**: Logarithmic scale due to hyperinflation (>2000% rates)
- **Post-1994**: Linear scale for modern inflation patterns
- Clear visual and analytical separation at the Plano Real implementation

### Data Sources Priority
1. World Bank API (primary): `https://api.worldbank.org/v2/country/BRA/indicator/FP.CPI.TOTL.ZG`
2. FRED API (backup): Series ID `FPCPITOTLZGBRA`
3. IBGE SIDRA API for IPCA (manual backup)

## Key Implementation Patterns

### Polars Lazy Evaluation
```python
df_processed = (
    df.lazy()
    .filter(pl.col("year") >= 1980)
    .with_columns([
        pl.col("inflation_rate").fill_null(strategy="forward"),
        pl.col("inflation_rate").rolling_mean(window_size=12).alias("ma_12")
    ])
    .group_by("year")
    .agg([pl.col("inflation_rate").mean().alias("annual_inflation")])
    .collect()
)
```

### Critical Historical Events to Annotate
- 1994: Plano Real (major breakpoint)
- 1999: Inflation targeting begins
- 2008: Global financial crisis
- 2020-2021: COVID-19 pandemic

## Output Requirements
- Static PNG/SVG files for publication
- Interactive HTML using Plotly
- Data export as Parquet (primary) and CSV (backup)
- Publication-ready quality with proper attribution

## Performance Considerations
- Use Parquet format for all intermediate data storage
- Implement lazy loading for large historical datasets  
- Leverage Polars' parallel operations for data processing
- Optimize memory usage with appropriate data types