# Brazil Inflation Visualization Project Plan

## Overview
A Python project to fetch, process, and visualize Brazil's inflation data over the last 30-40 years, with special focus on the hyperinflation period (pre-1994) and modern era (post-1994).

## Project Structure
```
brazil-inflation-viz/
├── data/
│   ├── raw/           # Original downloaded data
│   └── processed/     # Cleaned data files (Parquet format)
├── src/
│   ├── data_fetcher.py    # Download inflation data
│   ├── data_processor.py  # Clean and prepare data with Polars
│   └── visualizer.py       # Create graphs
├── output/
│   └── (generated graphs)
├── main.py            # Main script to run everything
├── requirements.txt   # Dependencies
└── README.md
```

## Technical Specifications

### Dependencies
```python
polars >= 0.20.0      # Fast dataframe library
matplotlib >= 3.5.0   # Static plots
plotly >= 5.0.0       # Interactive plots
requests >= 2.25.0    # API calls
seaborn >= 0.12.0     # Styling
python-dateutil       # Date handling
```

## Core Requirements

### Data Requirements
- Fetch IPCA inflation data from 1980-2024 (or available range)
- Handle missing data appropriately
- Calculate annual inflation rates if only monthly data available
- Create clean CSV/Parquet for reproducibility

### Visualization Requirements

#### 1. Pre-1994 Graph (Hyperinflation Period)
- Years: 1980-1994
- Use logarithmic Y-axis (due to values >2000%)
- Highlight key economic plans (Cruzado, Bresser, Collor)
- Color scheme: Red gradient showing intensity

#### 2. Post-1994 Graph (Modern Era)
- Years: 1995-2024
- Linear Y-axis
- Add reference line for inflation target (when it started in 1999)
- Highlight major events (2008 crisis, COVID-19, etc.)
- Color scheme: Blue/green for stable periods

#### 3. Combined Overview (Optional)
- Split panel or dual-axis design
- Clear visual break at 1994
- Annotation for Plano Real implementation

### Output Formats
- Static PNG/SVG files for reports
- Interactive HTML using Plotly
- Data export as Parquet (primary) and CSV (backup)

## Implementation Phases

### Phase 1 (Core)
- Data fetching from World Bank API or FRED API
- Basic dual graph creation
- Export as PNG

### Phase 2 (Enhancements)
- Add economic event annotations
- Implement moving averages
- Add inflation target bands (post-1999)
- Interactive tooltips with Plotly

### Phase 3 (Advanced)
- Comparison with other Latin American countries
- Real vs. nominal analysis
- Forecast projections
- Dashboard combining multiple views

## Data Source Strategy

### Primary Source: World Bank API
```python
# Example endpoint
https://api.worldbank.org/v2/country/BRA/indicator/FP.CPI.TOTL.ZG
```

### Backup Source: FRED API
```python
# FRED Series ID for Brazil CPI
series_id = "FPCPITOTLZGBRA"
```

### Manual Backup: IBGE SIDRA API for IPCA

## Polars Implementation Details

### Data Processing Example
```python
import polars as pl

# Reading data - faster than pandas
df = pl.read_csv("inflation_data.csv")

# Efficient operations
df_processed = (
    df.lazy()  # Lazy evaluation for optimization
    .filter(pl.col("year") >= 1980)
    .with_columns([
        pl.col("inflation_rate").fill_null(strategy="forward"),
        pl.col("inflation_rate").rolling_mean(window_size=12).alias("ma_12")
    ])
    .group_by("year")
    .agg([
        pl.col("inflation_rate").mean().alias("annual_inflation"),
        pl.col("inflation_rate").std().alias("volatility")
    ])
    .sort("year")
    .collect()  # Execute the query
)

# Save as Parquet for better performance
df_processed.write_parquet("data/processed/brazil_inflation.parquet")
```

### Key Polars Features to Leverage
1. **Lazy Evaluation** - Build query plans before execution
2. **Parquet Format** - Use for intermediate data storage (much faster than CSV)
3. **Expression API** - Chainable, readable transformations
4. **Native datetime handling** - Better performance with time series

### Memory Efficiency
```python
# Use lazy frames for large datasets
lazy_df = pl.scan_csv("large_file.csv")
result = lazy_df.filter(...).group_by(...).collect()
```

### Data Type Optimization
```python
# Specify dtypes for better performance
schema = {
    "year": pl.Int16,  # Years fit in Int16
    "month": pl.Int8,   # Months 1-12
    "inflation_rate": pl.Float32  # Sufficient precision
}
```

### Time Series Operations
```python
# Polars excels at time series
df.with_columns([
    pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
]).set_sorted("date")  # Tell Polars it's sorted for optimization
```

## Sample Main Script Structure

```python
# main.py
import polars as pl
from src.data_fetcher import fetch_inflation_data
from src.data_processor import process_with_polars
from src.visualizer import create_dual_graphs

def main():
    # Fetch raw data
    raw_data = fetch_inflation_data()
    
    # Process with Polars
    df = pl.DataFrame(raw_data)
    df_processed = process_with_polars(df)
    
    # Split periods
    df_pre_1994 = df_processed.filter(pl.col("year") < 1994)
    df_post_1994 = df_processed.filter(pl.col("year") >= 1994)
    
    # Create visualizations
    create_dual_graphs(df_pre_1994, df_post_1994)
    
    # Save processed data
    df_processed.write_parquet("data/processed/final_data.parquet")

if __name__ == "__main__":
    main()
```

## Claude Code Commands

### Initial Setup
```bash
"Create a Python project structure for Brazil inflation visualization using Polars"
```

### Data Phase
```bash
"Fetch Brazil inflation data from World Bank API from 1980-2024"
"Use Polars lazy evaluation to clean data and calculate annual rates"
"Save processed data as Parquet file for fast loading"
```

### Visualization Phase
```bash
"Create two separate graphs: pre-1994 with log scale, post-1994 with linear scale"
"Add annotations for major economic events like Plano Real"
"Make the graphs publication-ready with proper labels and styling"
```

### Enhancement Phase
```bash
"Add an interactive Plotly version with hover details"
"Include moving averages and inflation target bands"
"Calculate and visualize volatility metrics using Polars window functions"
```

## Performance Benefits

Using Polars provides:
- ⚡ 5-10x faster data loading
- ⚡ More efficient memory usage
- ⚡ Better handling of large historical datasets
- ⚡ Cleaner API for complex transformations
- ⚡ Native support for parallel operations

## Success Metrics
- ✅ Accurate data from reliable source
- ✅ Clear visual distinction between hyperinflation and modern era
- ✅ Professional, publication-ready graphics
- ✅ Reproducible with saved data and clear documentation
- ✅ Both static and interactive versions available
- ✅ Efficient processing with Polars
- ✅ Fast data loading with Parquet format

## Key Considerations for Claude Code

1. **Start simple**: First get basic data fetching working
2. **Iterate on visuals**: Start with basic plots, then enhance
3. **Handle edge cases**: Especially data gaps in the 1980s
4. **Document assumptions**: About data transformations
5. **Make it reproducible**: Save intermediate Parquet files
6. **Leverage Polars**: Use lazy evaluation for complex queries
7. **Optimize memory**: Use appropriate data types and lazy loading

## Additional Notes

### Historical Context to Include
- **1986**: Plano Cruzado
- **1987**: Plano Bresser
- **1989**: Plano Verão
- **1990**: Plano Collor I
- **1991**: Plano Collor II
- **1994**: Plano Real (critical turning point)
- **1999**: Inflation targeting regime begins
- **2008**: Global financial crisis
- **2015-2016**: Brazilian recession
- **2020-2021**: COVID-19 pandemic
- **2022-2023**: Post-pandemic inflation surge

### Visualization Best Practices
- Use consistent color schemes across related charts
- Include data source attribution
- Add confidence intervals where appropriate
- Ensure accessibility (colorblind-friendly palettes)
- Export at high DPI for publication quality
- Include both Portuguese and English labels (optional)