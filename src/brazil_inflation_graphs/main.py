"""Main execution script for Brazil inflation data fetching and processing."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import typer

from .config import Settings, get_settings
from .data_fetcher import BrazilInflationFetcher
from .data_processor import BrazilInflationProcessor
from .visualizer import BrazilInflationVisualizer
from .exceptions import BrazilInflationError
from .models import DataSource, InflationDataset, InflationDataPoint

console = Console()
app = typer.Typer(help="Brazil Inflation Data Fetching and Processing Tool")


def display_banner():
    """Display application banner."""
    banner = """
[bold blue]Brazil Inflation Data Analytics[/bold blue]
[dim]Polars-powered data fetching and processing for Brazil's inflation history[/dim]
[dim]Focusing on hyperinflation period (pre-1994) and modern era (post-1994)[/dim]
    """
    console.print(Panel(banner, title="Brazil Economic Data Analysis", border_style="blue"))


@app.command()
def fetch(
    source: Optional[str] = typer.Option(
        None, 
        "--source", 
        "-s",
        help="Preferred data source (fred, world_bank, ibge_sidra)"
    ),
    start_year: Optional[int] = typer.Option(
        None,
        "--start-year",
        help="Starting year for data collection"
    ),
    end_year: Optional[int] = typer.Option(
        None,
        "--end-year", 
        help="Ending year for data collection"
    ),
    save_raw: bool = typer.Option(
        True,
        "--save-raw/--no-save-raw",
        help="Save raw data to disk"
    ),
    process_data: bool = typer.Option(
        True,
        "--process/--no-process",
        help="Process data after fetching"
    )
):
    """Fetch Brazil inflation data from external sources."""
    display_banner()
    
    settings = get_settings()
    
    # Parse source parameter
    preferred_source = None
    if source:
        try:
            preferred_source = DataSource(source.lower())
        except ValueError:
            console.print(f"[red]Invalid source '{source}'. Valid options: fred, world_bank, ibge_sidra[/red]")
            raise typer.Exit(1)
    
    # Run async fetch operation
    asyncio.run(fetch_data_async(
        settings=settings,
        preferred_source=preferred_source,
        start_year=start_year,
        end_year=end_year,
        save_raw=save_raw,
        process_data=process_data
    ))


async def fetch_data_async(
    settings: Settings,
    preferred_source: Optional[DataSource] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    save_raw: bool = True,
    process_data: bool = True
):
    """Async data fetching operation."""
    try:
        console.print(f"[blue]Starting data fetch operation...[/blue]")
        
        async with BrazilInflationFetcher(settings) as fetcher:
            # Fetch best available data
            dataset = await fetcher.fetch_best_data(
                start_year=start_year,
                end_year=end_year,
                preferred_source=preferred_source
            )
            
            # Display dataset summary
            display_dataset_summary(dataset)
            
            # Save raw data if requested
            if save_raw:
                raw_path = await fetcher.save_raw_data(dataset)
                console.print(f"[green]Raw data saved to: {raw_path}[/green]")
            
            # Process data if requested
            if process_data:
                await process_dataset(dataset, settings)
                
    except BrazilInflationError as e:
        console.print(f"[red]Data fetch error: {e}[/red]")
        if e.source:
            console.print(f"[red]Source: {e.source}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def process(
    input_file: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input file path (parquet/csv)"
    ),
    output_format: str = typer.Option(
        "parquet",
        "--format",
        "-f",
        help="Output format (parquet, csv, json)"
    ),
    create_views: bool = typer.Option(
        True,
        "--views/--no-views",
        help="Create analytical views"
    ),
    quality_assessment: bool = typer.Option(
        True,
        "--quality/--no-quality",
        help="Perform data quality assessment"
    )
):
    """Process Brazil inflation data using Polars."""
    display_banner()
    
    settings = get_settings()
    processor = BrazilInflationProcessor(settings)
    
    try:
        if input_file:
            # Load from file
            console.print(f"[blue]Loading data from {input_file}[/blue]")
            lf = processor.load_processed_data(input_file)
        else:
            # Look for latest processed data
            processed_files = list(settings.storage.processed_dir.glob("brazil_inflation_*.parquet"))
            if not processed_files:
                console.print("[red]No processed data files found. Run 'fetch' command first.[/red]")
                raise typer.Exit(1)
            
            latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
            console.print(f"[blue]Loading latest processed data: {latest_file}[/blue]")
            lf = processor.load_processed_data(latest_file)
        
        # Process data
        console.print("[blue]Processing data with Polars...[/blue]")
        
        # Generate analytics
        generate_analytics_report(lf, processor, settings)
        
        # Create analytical views
        if create_views:
            create_analytical_views(lf, processor, output_format)
        
        # Quality assessment
        if quality_assessment:
            perform_quality_assessment(lf, processor)
        
        console.print("[green]Processing completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Processing error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def visualize(
    input_file: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input data file path (parquet/csv/json)"
    ),
    output_format: str = typer.Option(
        "both",
        "--format",
        "-f",
        help="Output format (matplotlib, plotly, both)"
    ),
    chart_type: str = typer.Option(
        "separate-periods",
        "--type",
        "-t",
        help="Chart type (separate-periods, statistics, both)"
    ),
    title: Optional[str] = typer.Option(
        None,
        "--title",
        help="Custom chart title"
    )
):
    """Create visualizations of Brazil inflation data."""
    display_banner()
    
    settings = get_settings()
    
    try:
        # Load dataset
        dataset = None
        
        if input_file:
            console.print(f"[blue]Loading data from {input_file}[/blue]")
            # Load data from file and convert to dataset
            if input_file.suffix == '.parquet':
                df = pl.read_parquet(input_file)
            elif input_file.suffix == '.csv':
                df = pl.read_csv(input_file)
            elif input_file.suffix == '.json':
                import json
                with open(input_file, 'r') as f:
                    data = json.load(f)
                # Assuming it's a saved InflationDataset
                dataset = InflationDataset(**data)
            else:
                console.print(f"[red]Unsupported file format: {input_file.suffix}[/red]")
                raise typer.Exit(1)
                
            if dataset is None:
                # Convert DataFrame back to InflationDataset format
                data_points = []
                for row in df.to_dicts():
                    data_points.append(InflationDataPoint(
                        date=row['date'],
                        value=row['inflation_rate'],
                        source=DataSource(row.get('source', 'unknown')),
                    ))
                dataset = InflationDataset(
                    data_points=data_points,
                    source=DataSource(data_points[0].source if data_points else DataSource.FRED)
                )
        else:
            # Look for latest processed data
            processed_files = list(settings.storage.processed_dir.glob("brazil_inflation_*.parquet"))
            if not processed_files:
                console.print("[red]No processed data found. Run 'fetch' or 'process' command first.[/red]")
                raise typer.Exit(1)
            
            latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
            console.print(f"[blue]Loading latest processed data: {latest_file}[/blue]")
            
            df = pl.read_parquet(latest_file)
            # Convert to InflationDataset
            data_points = []
            for row in df.to_dicts():
                data_points.append(InflationDataPoint(
                    date=row['date'],
                    value=row['inflation_rate'],
                    source=DataSource(row.get('source', 'fred')),
                ))
            dataset = InflationDataset(
                data_points=data_points,
                source=DataSource(data_points[0].source if data_points else DataSource.FRED)
            )
        
        # Create visualizer
        visualizer = BrazilInflationVisualizer(settings)
        
        console.print(f"[blue]Creating {chart_type} visualizations in {output_format} format...[/blue]")
        
        # Create visualizations based on chart_type
        output_files = {}
        
        if chart_type in ("dual-period", "separate-periods", "both"):
            separate_files = visualizer.create_separate_period_charts(
                dataset=dataset,
                output_format=output_format
            )
            output_files.update(separate_files)
        
        if chart_type in ("statistics", "both"):
            stats_file = visualizer.create_summary_statistics_chart(dataset)
            output_files["statistics"] = stats_file
        
        # Export chart data
        data_file = visualizer.export_chart_data(dataset)
        output_files["data"] = data_file
        
        # Display results
        console.print("\n[green]Visualization completed successfully![/green]")
        
        results_table = Table(title="Generated Files")
        results_table.add_column("Type", style="cyan")
        results_table.add_column("File Path", style="white")
        
        for file_type, file_path in output_files.items():
            results_table.add_row(file_type.title(), str(file_path))
        
        console.print(results_table)
        
        # Display dataset info
        display_dataset_summary(dataset)
        
    except Exception as e:
        import traceback
        console.print(f"[red]Visualization error: {e}[/red]")
        console.print(f"[red]Full traceback:[/red]")
        traceback.print_exc()
        raise typer.Exit(1)


async def process_dataset(dataset, settings: Settings):
    """Process a fetched dataset."""
    processor = BrazilInflationProcessor(settings)
    
    console.print("[blue]Processing dataset with Polars...[/blue]")
    
    # Process the dataset
    lf = processor.process_dataset(dataset)
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"brazil_inflation_{dataset.source.value}_{timestamp}"
    
    processed_path = processor.save_processed_data(lf, filename)
    
    # Generate initial analytics
    generate_analytics_report(lf, processor, settings)
    
    # Create analytical views
    create_analytical_views(lf, processor, "parquet")


def display_dataset_summary(dataset):
    """Display dataset summary information."""
    table = Table(title="Dataset Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Source", dataset.source.value)
    table.add_row("Total Observations", str(dataset.total_observations))
    table.add_row("Date Range", f"{dataset.date_range[0]} to {dataset.date_range[1]}")
    table.add_row("Hyperinflation Obs", str(dataset.hyperinflation_observations))
    table.add_row("Modern Era Obs", str(dataset.modern_observations))
    table.add_row("Retrieved At", dataset.retrieved_at.strftime("%Y-%m-%d %H:%M:%S"))
    
    console.print(table)


def generate_analytics_report(lf: pl.LazyFrame, processor: BrazilInflationProcessor, settings: Settings):
    """Generate comprehensive analytics report."""
    console.print("[blue]Generating analytics report...[/blue]")
    
    # Dual-period comparison
    comparison = processor.create_dual_period_comparison(lf)
    
    # Display comparison table
    table = Table(title="Hyperinflation vs Modern Era Comparison")
    table.add_column("Period", style="cyan")
    table.add_column("Avg Inflation (%)", style="red")
    table.add_column("Median (%)", style="yellow")
    table.add_column("Volatility", style="magenta")
    table.add_column("Min/Max (%)", style="green")
    table.add_column("Observations", style="white")
    
    for row in comparison.to_dicts():
        period = row["period"].title()
        avg_inf = f"{row['avg_inflation']:.2f}" if row['avg_inflation'] else "N/A"
        median_inf = f"{row['median_inflation']:.2f}" if row['median_inflation'] else "N/A"
        volatility = f"{row['volatility']:.2f}" if row['volatility'] else "N/A"
        min_max = f"{row['min_inflation']:.1f} / {row['max_inflation']:.1f}" if row['min_inflation'] else "N/A"
        obs = str(row['observations'])
        
        table.add_row(period, avg_inf, median_inf, volatility, min_max, obs)
    
    console.print(table)
    
    # Hyperinflation metrics
    try:
        hyper_metrics = processor.calculate_hyperinflation_metrics(lf)
        if hyper_metrics.height > 0:
            display_hyperinflation_metrics(hyper_metrics)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not calculate hyperinflation metrics: {e}[/yellow]")
    
    # Modern metrics
    try:
        modern_metrics = processor.calculate_modern_metrics(lf)
        if modern_metrics.height > 0:
            display_modern_metrics(modern_metrics)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not calculate modern metrics: {e}[/yellow]")


def display_hyperinflation_metrics(metrics: pl.DataFrame):
    """Display hyperinflation period metrics."""
    table = Table(title="Hyperinflation Period (Pre-1994) Metrics")
    table.add_column("Metric", style="red")
    table.add_column("Value", style="white")
    
    row = metrics.to_dicts()[0]
    table.add_row("Total Months", str(row.get('hyperinflation_months', 'N/A')))
    table.add_row("Extreme Months (>50%)", str(row.get('extreme_months', 'N/A')))
    table.add_row("Peak Inflation (%)", f"{row.get('peak_inflation', 0):.1f}")
    table.add_row("Average Inflation (%)", f"{row.get('avg_hyperinflation', 0):.1f}")
    table.add_row("Average Volatility", f"{row.get('avg_volatility', 0):.1f}")
    
    if row.get('peak_date'):
        table.add_row("Peak Date", str(row['peak_date']))
    
    console.print(table)


def display_modern_metrics(metrics: pl.DataFrame):
    """Display modern period metrics."""
    table = Table(title="Modern Era (Post-1994) Metrics")
    table.add_column("Metric", style="green")
    table.add_column("Value", style="white")
    
    row = metrics.to_dicts()[0]
    table.add_row("Total Months", str(row.get('modern_months', 'N/A')))
    table.add_row("Average Inflation (%)", f"{row.get('avg_modern_inflation', 0):.2f}")
    table.add_row("Volatility", f"{row.get('modern_volatility', 0):.2f}")
    table.add_row("Min/Max (%)", f"{row.get('min_modern_inflation', 0):.1f} / {row.get('max_modern_inflation', 0):.1f}")
    table.add_row("12M Trend Average", f"{row.get('avg_12m_trend', 0):.2f}")
    table.add_row("Deflationary Months", str(row.get('deflationary_months', 'N/A')))
    table.add_row("High Inflation Months (>10%)", str(row.get('high_inflation_months', 'N/A')))
    
    console.print(table)


def create_analytical_views(lf: pl.LazyFrame, processor: BrazilInflationProcessor, format: str):
    """Create and save analytical views."""
    console.print("[blue]Creating analytical views...[/blue]")
    
    views = processor.create_analytical_views(lf)
    
    for view_name, view_lf in views.items():
        filename = f"view_{view_name}"
        try:
            saved_path = processor.save_processed_data(view_lf, filename, format)
            console.print(f"[green]Created {view_name} view: {saved_path.name}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to create {view_name} view: {e}[/red]")


def perform_quality_assessment(lf: pl.LazyFrame, processor: BrazilInflationProcessor):
    """Perform data quality assessment."""
    console.print("[blue]Performing data quality assessment...[/blue]")
    
    # Assuming we can extract source from the LazyFrame
    try:
        # Get first source from data
        source_str = lf.select(pl.col("source").first()).collect().item()
        source = DataSource(source_str)
        
        quality_report = processor.assess_data_quality(lf, source)
        
        # Display quality report
        table = Table(title="Data Quality Assessment")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Source", quality_report.source.value)
        table.add_row("Total Points", str(quality_report.total_points))
        table.add_row("Missing Points", str(quality_report.missing_points))
        table.add_row("Outliers", str(quality_report.outliers))
        table.add_row("Hyperinflation Anomalies", str(quality_report.hyperinflation_anomalies))
        table.add_row("Data Completeness", f"{quality_report.data_completeness:.1f}%")
        table.add_row("Quality Score", f"{quality_report.quality_score:.2f}")
        
        console.print(table)
        
        # Display issues and recommendations
        if quality_report.issues:
            console.print("\n[red]Issues Found:[/red]")
            for issue in quality_report.issues:
                console.print(f"[red]• {issue}[/red]")
        
        if quality_report.recommendations:
            console.print("\n[yellow]Recommendations:[/yellow]")
            for rec in quality_report.recommendations:
                console.print(f"[yellow]• {rec}[/yellow]")
        
    except Exception as e:
        console.print(f"[yellow]Warning: Quality assessment failed: {e}[/yellow]")


@app.command()
def info():
    """Display system and configuration information."""
    display_banner()
    
    settings = get_settings()
    
    # Basic configuration table
    table = Table(title="System Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Environment", settings.environment.value)
    table.add_row("Debug Mode", str(settings.debug))
    table.add_row("Data Directory", str(settings.storage.data_dir))
    table.add_row("Start Year", str(settings.data_source.start_year))
    table.add_row("End Year", str(settings.data_source.end_year))
    table.add_row("Plano Real Year", str(settings.data_source.plano_real_year))
    table.add_row("FRED API Key", "Set" if settings.api.fred_api_key else "Not Set")
    table.add_row("Storage Format", settings.storage.processed_format)
    table.add_row("Lazy Evaluation", str(settings.storage.use_lazy_evaluation))
    table.add_row("Cache Enabled", str(settings.cache.enable_cache))
    table.add_row("Log Level", settings.log_level)
    
    console.print(table)
    
    # Performance settings
    perf_table = Table(title="Performance Settings")
    perf_table.add_column("Setting", style="magenta")
    perf_table.add_column("Value", style="white")
    
    perf_table.add_row("Max HTTP Connections", str(settings.http.max_connections))
    perf_table.add_row("Request Timeout", f"{settings.api.timeout_seconds}s")
    perf_table.add_row("Polars Thread Pool", str(settings.polars.thread_pool_size) if settings.polars.thread_pool_size > 0 else "Auto")
    perf_table.add_row("Streaming Chunk Size", f"{settings.polars.streaming_chunk_size:,}")
    perf_table.add_row("Cache TTL", f"{settings.cache.cache_ttl_seconds}s")
    
    console.print(perf_table)
    
    # Check Polars version
    try:
        console.print(f"\n[blue]Polars Version:[/blue] {pl.__version__}")
    except:
        console.print("\n[red]Polars not available[/red]")


# Configuration management commands
config_app = typer.Typer(help="Configuration management commands")
app.add_typer(config_app, name="config")


@config_app.command()
def show():
    """Show current configuration in detail."""
    from rich.json import JSON
    
    display_banner()
    
    settings = get_settings()
    effective_config = settings.get_effective_config()
    
    console.print("\n[bold blue]Current Effective Configuration:[/bold blue]")
    console.print(JSON.from_data(effective_config))


@config_app.command() 
def validate():
    """Validate configuration and environment setup."""
    from .config import validate_environment_setup
    
    display_banner()
    
    console.print("[blue]Validating configuration and environment...[/blue]\n")
    
    validation_report = validate_environment_setup()
    
    # Status overview
    status = "[green]VALID[/green]" if validation_report["valid"] else "[red]INVALID[/red]"
    console.print(f"Configuration Status: {status}")
    console.print(f"Environment: {validation_report['environment']}")
    console.print(f"Debug Mode: {validation_report['debug_mode']}")
    console.print(f"API Configured: {validation_report['api_configured']}")
    
    # Dependencies
    deps_table = Table(title="Dependencies")
    deps_table.add_column("Package", style="cyan")
    deps_table.add_column("Status", style="white")
    deps_table.add_column("Version", style="dim")
    
    deps_table.add_row(
        "Polars", 
        "[green]Available[/green]" if validation_report.get("polars_available") else "[red]Missing[/red]",
        validation_report.get("polars_version", "N/A")
    )
    
    deps_table.add_row(
        "Matplotlib",
        "[green]Available[/green]" if validation_report.get("matplotlib_available") else "[red]Missing[/red]",
        "N/A"
    )
    
    console.print(deps_table)
    
    # Issues
    if validation_report["issues"]:
        console.print("\n[red]Issues Found:[/red]")
        for issue in validation_report["issues"]:
            console.print(f"[red]• {issue}[/red]")
    
    # Warnings
    if validation_report["warnings"]:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in validation_report["warnings"]:
            console.print(f"[yellow]• {warning}[/yellow]")
    
    if validation_report["valid"] and not validation_report["warnings"]:
        console.print("\n[green]Configuration is valid and ready to use![/green]")
    elif validation_report["valid"]:
        console.print("\n[yellow]Configuration is valid but has warnings.[/yellow]")
    else:
        console.print("\n[red]Configuration has issues that need to be resolved.[/red]")
        raise typer.Exit(1)


@config_app.command()
def create_template(
    output_path: Path = typer.Option(
        Path(".env.example"),
        "--output",
        "-o",
        help="Output path for template file"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing file"
    )
):
    """Create a .env template file with all configuration options."""
    from .config import create_env_template
    
    display_banner()
    
    if output_path.exists() and not force:
        console.print(f"[red]File {output_path} already exists. Use --force to overwrite.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Creating environment template at {output_path}...[/blue]")
    
    try:
        create_env_template(output_path)
        console.print(f"[green]Template created successfully at {output_path}[/green]")
        console.print(f"[dim]Copy this file to .env and customize the values for your environment.[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to create template: {e}[/red]")
        raise typer.Exit(1)


@config_app.command()
def help():
    """Show detailed configuration help."""
    from .config import print_configuration_help
    
    display_banner()
    print_configuration_help()


if __name__ == "__main__":
    app()