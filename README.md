# Brazil Inflation Visualization

A Python project for visualizing Brazil's inflation data over the past 30-40 years, highlighting the dramatic economic transformation from hyperinflation to modern monetary stability.

## Overview

This project creates dual-period visualizations that showcase:
- **Pre-1994**: Hyperinflation period (logarithmic scale)
- **Post-1994**: Modern inflation era after Plano Real (linear scale)

## Key Features

- Fast data processing with Polars
- Interactive and static visualizations
- Historical economic event annotations
- Publication-ready output formats

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Clone the repository
git clone https://github.com/n0vdd/inflation_graphs.git
cd inflation_graphs

# Install dependencies
uv sync
```

## Usage

```bash
# Run the main visualization script
uv run python main.py
```

## Data Sources

- World Bank API (primary)
- FRED API (backup)  
- IBGE SIDRA API (manual backup)

## Output

Generated visualizations are saved to the `output/` directory in PNG, SVG, and interactive HTML formats.