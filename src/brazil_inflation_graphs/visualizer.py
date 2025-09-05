"""Enhanced Brazil Inflation Data Visualizer.

Extends the base visualizer with advanced analytics visualizations including:
- Seasonal decomposition charts
- Structural break detection visualizations
- Volatility modeling charts
- Multi-source integration dashboards
- Monetary policy effectiveness analysis
- International comparative visualizations
"""

import logging
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import asyncio

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import numpy as np
from scipy import stats

# Import from modules in same package
try:
    from .enhanced_visualizer_backup import BrazilInflationVisualizer, HistoricalEvent
except ImportError:
    # If backup doesn't exist, create a minimal base class
    from pathlib import Path
    from typing import List
    from dataclasses import dataclass
    from datetime import date
    
    @dataclass
    class HistoricalEvent:
        name: str
        date: date
        color: str = "gray"
        linestyle: str = "--"
    
    class BrazilInflationVisualizer:
        def __init__(self, config=None):
            self.config = config
            self.output_dir = Path(config.storage.output_dir if config else "output")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Historical events for Brazil inflation analysis
            self.HISTORICAL_EVENTS = [
                HistoricalEvent("Plano Real", date(1994, 7, 1), "darkgreen", "-"),
                HistoricalEvent("Inflation Targeting", date(1999, 6, 1), "blue", "--"),
                HistoricalEvent("Global Financial Crisis", date(2008, 9, 15), "red", ":"),
                HistoricalEvent("COVID-19 Pandemic", date(2020, 3, 11), "purple", "-."),
            ]

try:
    from .advanced_analytics import AdvancedInflationAnalytics
except ImportError:
    AdvancedInflationAnalytics = None
try:
    from .comparative_analysis import ComparativeInflationAnalysis
except ImportError:
    ComparativeInflationAnalysis = None
try:
    from .monetary_policy import MonetaryPolicyAnalyzer
except ImportError:
    MonetaryPolicyAnalyzer = None
from .models import InflationDataset, InflationPeriod, DataSource
from .config import Settings

logger = logging.getLogger(__name__)


class EnhancedBrazilInflationVisualizer(BrazilInflationVisualizer):
    """Enhanced visualizer with advanced analytics capabilities."""
    
    def __init__(self, config: Optional[Settings] = None):
        """Initialize enhanced visualizer with analytics modules."""
        super().__init__(config)
        if AdvancedInflationAnalytics:
            self.analytics = AdvancedInflationAnalytics(self.config)
        else:
            self.analytics = None
        if ComparativeInflationAnalysis:
            self.comparative = ComparativeInflationAnalysis(self.config)
        else:
            self.comparative = None
        if MonetaryPolicyAnalyzer:
            self.monetary = MonetaryPolicyAnalyzer(self.config)
        else:
            self.monetary = None
    
    def create_separate_period_charts(self, dataset: InflationDataset, output_format: str = "both") -> Dict[str, Path]:
        """Create separate charts for hyperinflation and modern periods."""
        output_files = {}
        
        # Create hyperinflation period chart (pre-1994)
        hyperinflation_data = [dp for dp in dataset.data_points if dp.period == InflationPeriod.HYPERINFLATION]
        if hyperinflation_data:
            hyper_dataset = InflationDataset(
                data_points=hyperinflation_data,
                source=dataset.source
            )
            if output_format in ("matplotlib", "both"):
                output_files["brazil_hyperinflation_period"] = self._create_period_chart_matplotlib(
                    hyper_dataset, "Hyperinflation Period (Pre-1994)", use_log_scale=True
                )
            if output_format in ("plotly", "both"):
                output_files["brazil_hyperinflation_interactive"] = self._create_period_chart_plotly(
                    hyper_dataset, "Hyperinflation Period (Pre-1994)", use_log_scale=True
                )
        
        # Create modern period chart (post-1994)
        modern_data = [dp for dp in dataset.data_points if dp.period == InflationPeriod.MODERN]
        if modern_data:
            modern_dataset = InflationDataset(
                data_points=modern_data,
                source=dataset.source
            )
            if output_format in ("matplotlib", "both"):
                output_files["brazil_modern_period"] = self._create_period_chart_matplotlib(
                    modern_dataset, "Modern Period (Post-1994)", use_log_scale=False
                )
            if output_format in ("plotly", "both"):
                output_files["brazil_modern_period_interactive"] = self._create_period_chart_plotly(
                    modern_dataset, "Modern Period (Post-1994)", use_log_scale=False
                )
        
        return output_files
    
    def _create_period_chart_matplotlib(self, dataset: InflationDataset, title: str, use_log_scale: bool = False) -> Path:
        """Create matplotlib chart for a specific period."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        df = dataset.to_polars()
        dates = df['date'].to_list()
        rates = df['inflation_rate'].to_list()
        
        # Plot monthly data
        ax.plot(dates, rates, color='darkred', linewidth=0.8, alpha=0.7, label='Monthly Inflation Rate')
        
        # Add 12-month rolling average
        if len(rates) >= 12:
            rolling_avg = df.with_columns([
                pl.col('inflation_rate').rolling_mean(window_size=12).alias('rolling_avg')
            ])['rolling_avg'].to_list()
            ax.plot(dates, rolling_avg, color='red', linewidth=2, label='12-month Rolling Average')
        
        # Set scale
        if use_log_scale and max(rates) > 10:
            ax.set_yscale('log')
            ax.set_ylabel('Monthly Inflation Rate (%, log scale)')
        else:
            ax.set_ylabel('Monthly Inflation Rate (%)')
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add historical events
        for event in self.HISTORICAL_EVENTS:
            if dates[0] <= event.date <= dates[-1]:
                ax.axvline(x=event.date, color=event.color, linestyle=event.linestyle, alpha=0.7)
        
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        
        plt.tight_layout()
        
        # Save file
        filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') + '.png'
        output_file = self.output_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"{title} chart saved to {output_file}")
        return output_file
    
    def _create_period_chart_plotly(self, dataset: InflationDataset, title: str, use_log_scale: bool = False) -> Path:
        """Create interactive plotly chart for a specific period."""
        df = dataset.to_polars()
        
        fig = go.Figure()
        
        # Add main line
        fig.add_trace(go.Scatter(
            x=df['date'].to_list(),
            y=df['inflation_rate'].to_list(),
            mode='lines',
            name='Monthly Inflation Rate',
            line=dict(color='darkred', width=1.5)
        ))
        
        # Add rolling average if enough data
        if len(df) >= 12:
            rolling_avg = df.with_columns([
                pl.col('inflation_rate').rolling_mean(window_size=12).alias('rolling_avg')
            ])['rolling_avg'].to_list()
            fig.add_trace(go.Scatter(
                x=df['date'].to_list(),
                y=rolling_avg,
                mode='lines',
                name='12-month Rolling Average',
                line=dict(color='red', width=2)
            ))
        
        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Monthly Inflation Rate (%)",
            template='plotly_white'
        )
        
        if use_log_scale:
            fig.update_yaxes(type="log")
        
        # Save file
        filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') + '_interactive.html'
        output_file = self.output_dir / filename
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive {title} chart saved to {output_file}")
        return output_file
    
    def create_summary_statistics_chart(self, dataset: InflationDataset) -> Path:
        """Create summary statistics visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        df = dataset.to_polars()
        stats = dataset.get_summary_stats()
        
        # 1. Distribution histogram
        rates = df['inflation_rate'].to_list()
        ax1.hist(rates, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=stats['mean_inflation'], color='red', linestyle='--', label=f'Mean: {stats["mean_inflation"]:.2f}%')
        ax1.axvline(x=stats['median_inflation'], color='green', linestyle='--', label=f'Median: {stats["median_inflation"]:.2f}%')
        ax1.set_xlabel('Inflation Rate (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Inflation Rate Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Period comparison
        periods = ['Hyperinflation', 'Transition', 'Modern']
        period_stats = []
        for period in [InflationPeriod.HYPERINFLATION, InflationPeriod.TRANSITION, InflationPeriod.MODERN]:
            period_data = [dp.value for dp in dataset.data_points if dp.period == period]
            if period_data:
                period_stats.append(np.mean(period_data))
            else:
                period_stats.append(0)
        
        ax2.bar(periods, period_stats, color=['red', 'orange', 'green'], alpha=0.7)
        ax2.set_ylabel('Average Inflation Rate (%)')
        ax2.set_title('Average Inflation by Period')
        ax2.grid(True, alpha=0.3)
        
        # 3. Volatility over time
        rolling_std = df.with_columns([
            pl.col('inflation_rate').rolling_std(window_size=12).alias('rolling_std')
        ])
        ax3.plot(rolling_std['date'].to_list(), rolling_std['rolling_std'].to_list(), 
                 color='purple', linewidth=2)
        ax3.set_ylabel('12-month Rolling Std Dev')
        ax3.set_title('Inflation Volatility Over Time')
        ax3.grid(True, alpha=0.3)
        
        # 4. Key statistics table
        ax4.axis('off')
        stats_text = f"""Key Statistics:
        
Mean: {stats['mean_inflation']:.2f}%
Median: {stats['median_inflation']:.2f}%
Std Dev: {stats['std_inflation']:.2f}%
Min: {stats['min_inflation']:.2f}%
Max: {stats['max_inflation']:.2f}%
        
Total Observations: {dataset.total_observations}
Hyperinflation Era: {dataset.hyperinflation_observations}
Modern Era: {dataset.modern_observations}"""
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.suptitle('Brazil Inflation Statistics Summary', fontsize=16)
        plt.tight_layout()
        
        output_file = self.output_dir / "brazil_inflation_statistics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Statistics chart saved to {output_file}")
        return output_file
    
    def export_chart_data(self, dataset: InflationDataset) -> Path:
        """Export processed chart data as Parquet file."""
        df = dataset.to_polars()
        
        # Add additional computed columns for export
        export_df = df.with_columns([
            pl.col('inflation_rate').rolling_mean(window_size=12).alias('rolling_mean_12m'),
            pl.col('inflation_rate').rolling_std(window_size=12).alias('rolling_std_12m'),
            pl.col('inflation_rate').rolling_mean(window_size=3).alias('rolling_mean_3m'),
            pl.when(pl.col('year') < 1994).then(pl.lit('Hyperinflation')).otherwise(
                pl.when(pl.col('year') <= 1999).then(pl.lit('Transition')).otherwise(pl.lit('Modern'))
            ).alias('economic_period')
        ])
        
        output_file = self.output_dir / "brazil_inflation_chart_data.parquet"
        export_df.write_parquet(output_file)
        
        logger.info(f"Chart data exported to {output_file}")
        return output_file
        
    def create_seasonal_decomposition_chart(
        self, 
        dataset: InflationDataset,
        model: str = "additive",
        period: Optional[int] = None,
        output_format: str = "both"
    ) -> Dict[str, Path]:
        """Create seasonal decomposition visualizations.
        
        Args:
            dataset: Inflation dataset
            model: Decomposition model ('additive' or 'multiplicative')
            period: Seasonal period (default: auto-detect)
            output_format: "matplotlib", "plotly", or "both"
            
        Returns:
            Dictionary mapping chart type to output file paths
        """
        logger.info(f"Creating seasonal decomposition charts ({model} model)")
        
        # Perform decomposition analysis
        lf = dataset.to_polars().lazy()
        decomposition_results = self.analytics.perform_seasonal_decomposition(
            lf, model=model, period=period
        )
        
        output_files = {}
        
        if output_format in ("matplotlib", "both"):
            static_file = self._create_seasonal_decomposition_matplotlib(
                decomposition_results, model
            )
            output_files["seasonal_decomposition_static"] = static_file
            
        if output_format in ("plotly", "both"):
            interactive_file = self._create_seasonal_decomposition_plotly(
                decomposition_results, model
            )
            output_files["seasonal_decomposition_interactive"] = interactive_file
            
        return output_files
    
    def _create_seasonal_decomposition_matplotlib(
        self, 
        decomposition: Dict[str, pl.DataFrame],
        model: str
    ) -> Path:
        """Create matplotlib seasonal decomposition chart."""
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        components = ['original', 'trend', 'seasonal', 'residual']
        titles = [
            'Original Series',
            'Trend Component', 
            'Seasonal Component',
            'Residual Component'
        ]
        colors = ['darkblue', 'green', 'orange', 'red']
        
        for i, (component, title, color) in enumerate(zip(components, titles, colors)):
            if component in decomposition:
                df = decomposition[component].to_pandas()
                df['date'] = pd.to_datetime(df['date'])
                
                axes[i].plot(df['date'], df['value'], color=color, linewidth=1.5)
                axes[i].set_title(title, fontsize=12, pad=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylabel('Value')
                
                # Format y-axis for original and trend (log scale for hyperinflation)
                if component in ['original', 'trend'] and df['value'].max() > 100:
                    axes[i].set_yscale('log')
                    axes[i].set_ylabel(f'{title} (log scale)')
        
        # Format x-axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        axes[-1].set_xlabel('Year')
        
        plt.suptitle(f'Seasonal Decomposition - {model.title()} Model', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save file
        output_file = self.output_dir / f"seasonal_decomposition_{model}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Seasonal decomposition chart saved to {output_file}")
        return output_file
    
    def _create_seasonal_decomposition_plotly(
        self,
        decomposition: Dict[str, pl.DataFrame],
        model: str
    ) -> Path:
        """Create interactive Plotly seasonal decomposition chart."""
        components = ['original', 'trend', 'seasonal', 'residual']
        titles = [
            'Original Series',
            'Trend Component',
            'Seasonal Component', 
            'Residual Component'
        ]
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=titles,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        colors = ['darkblue', 'green', 'orange', 'red']
        
        for i, (component, color) in enumerate(zip(components, colors)):
            if component in decomposition:
                df = decomposition[component]
                
                fig.add_trace(
                    go.Scatter(
                        x=df['date'].to_list(),
                        y=df['value'].to_list(),
                        mode='lines',
                        name=titles[i],
                        line=dict(color=color, width=1.5),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.3f}<extra></extra>'
                    ),
                    row=i+1, col=1
                )
                
                # Use log scale for original and trend if hyperinflation present
                if component in ['original', 'trend']:
                    max_val = df['value'].max()
                    if max_val > 100:
                        fig.update_yaxes(type="log", row=i+1, col=1)
        
        fig.update_layout(
            title={
                'text': f'Seasonal Decomposition - {model.title()} Model',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=1000,
            showlegend=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Year", row=4, col=1)
        
        # Save file
        output_file = self.output_dir / f"seasonal_decomposition_{model}_interactive.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive seasonal decomposition chart saved to {output_file}")
        return output_file
    
    def create_structural_break_chart(
        self,
        dataset: InflationDataset,
        output_format: str = "both"
    ) -> Dict[str, Path]:
        """Create structural break detection visualizations.
        
        Args:
            dataset: Inflation dataset
            output_format: "matplotlib", "plotly", or "both"
            
        Returns:
            Dictionary mapping chart type to output file paths
        """
        logger.info("Creating structural break detection charts")
        
        # Perform structural break analysis
        lf = dataset.to_polars().lazy()
        break_results = self.analytics.detect_structural_breaks(lf)
        
        output_files = {}
        
        if output_format in ("matplotlib", "both"):
            static_file = self._create_structural_break_matplotlib(
                dataset.to_polars(), break_results
            )
            output_files["structural_breaks_static"] = static_file
            
        if output_format in ("plotly", "both"):
            interactive_file = self._create_structural_break_plotly(
                dataset.to_polars(), break_results
            )
            output_files["structural_breaks_interactive"] = interactive_file
            
        return output_files
    
    def _create_structural_break_matplotlib(
        self,
        df: pl.DataFrame,
        break_results: Dict[str, Any]
    ) -> Path:
        """Create matplotlib structural break chart."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])
        
        # Convert to pandas for matplotlib
        df_pandas = df.to_pandas()
        df_pandas['date'] = pd.to_datetime(df_pandas['date'])
        
        # Plot main time series
        ax1.plot(
            df_pandas['date'], 
            df_pandas['inflation_rate'],
            color='darkblue', 
            linewidth=1.5,
            alpha=0.7,
            label='Inflation Rate'
        )
        
        # Add structural break points
        if 'break_dates' in break_results:
            for break_date in break_results['break_dates']:
                if isinstance(break_date, str):
                    break_date = pd.to_datetime(break_date).date()
                ax1.axvline(
                    x=break_date, 
                    color='red', 
                    linestyle='--', 
                    linewidth=2,
                    alpha=0.8,
                    label='Structural Break'
                )
        
        # Add historical events
        for event in self.HISTORICAL_EVENTS:
            event_datetime = datetime.combine(event.date, time.min)
            if (df_pandas['date'].min() <= event_datetime <= df_pandas['date'].max()):
                ax1.axvline(x=event.date, color=event.color, linestyle=event.linestyle, alpha=0.6)
                y_pos = ax1.get_ylim()[1] * 0.9
                ax1.text(event.date, y_pos, f'  {event.name}', 
                        rotation=90, verticalalignment='top', 
                        color=event.color, fontsize=9)
        
        ax1.set_ylabel('Inflation Rate (%)')
        ax1.set_title('Structural Break Detection in Brazil Inflation', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot CUSUM test statistics if available
        if 'cusum_stats' in break_results:
            cusum_data = break_results['cusum_stats']
            if isinstance(cusum_data, pl.DataFrame):
                cusum_pandas = cusum_data.to_pandas()
                cusum_pandas['date'] = pd.to_datetime(cusum_pandas['date'])
                
                ax2.plot(
                    cusum_pandas['date'],
                    cusum_pandas['cusum_stat'],
                    color='green',
                    linewidth=1.5,
                    label='CUSUM Statistic'
                )
                
                # Add confidence bands
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax2.axhline(y=1.96, color='red', linestyle='--', alpha=0.5, label='95% Confidence')
                ax2.axhline(y=-1.96, color='red', linestyle='--', alpha=0.5)
        
        ax2.set_ylabel('CUSUM Statistic')
        ax2.set_xlabel('Year')
        ax2.set_title('CUSUM Test for Parameter Stability', fontsize=12)
        ax2.grid(True, alpha=0.3)
        # Only show legend if there are labeled items
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend()
        
        # Format x-axes
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save file
        output_file = self.output_dir / "structural_breaks_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Structural break chart saved to {output_file}")
        return output_file
    
    def _create_structural_break_plotly(
        self,
        df: pl.DataFrame,
        break_results: Dict[str, Any]
    ) -> Path:
        """Create interactive Plotly structural break chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                'Inflation Rate with Structural Breaks',
                'CUSUM Test Statistics'
            ],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Plot main time series
        fig.add_trace(
            go.Scatter(
                x=df['date'].to_list(),
                y=df['inflation_rate'].to_list(),
                mode='lines',
                name='Inflation Rate',
                line=dict(color='darkblue', width=1.5),
                hovertemplate='<b>Date:</b> %{x}<br><b>Inflation:</b> %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add structural break points
        if 'break_dates' in break_results:
            for i, break_date in enumerate(break_results['break_dates']):
                if isinstance(break_date, str):
                    break_date = pd.to_datetime(break_date).strftime('%Y-%m-%d')
                
                fig.add_shape(
                    type="line",
                    x0=break_date, x1=break_date,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash"),
                    row=1, col=1
                )
                
                fig.add_annotation(
                    x=break_date,
                    y=0.9 - (i * 0.1),  # Stagger annotations
                    yref="paper",
                    text=f"Break {i+1}",
                    showarrow=False,
                    font=dict(color="red", size=10),
                    row=1, col=1
                )
        
        # Add historical events
        for event in self.HISTORICAL_EVENTS:
            event_date_str = event.date.strftime("%Y-%m-%d")
            line_dash = "dash" if event.linestyle == "--" else "dot"
            
            fig.add_shape(
                type="line",
                x0=event_date_str, x1=event_date_str,
                y0=0, y1=1,
                yref="paper",
                line=dict(color=event.color, width=1, dash=line_dash),
                row=1, col=1
            )
        
        # Plot CUSUM statistics if available
        if 'cusum_stats' in break_results and isinstance(break_results['cusum_stats'], pl.DataFrame):
            cusum_data = break_results['cusum_stats']
            
            fig.add_trace(
                go.Scatter(
                    x=cusum_data['date'].to_list(),
                    y=cusum_data['cusum_stat'].to_list(),
                    mode='lines',
                    name='CUSUM Statistic',
                    line=dict(color='green', width=1.5),
                    hovertemplate='<b>Date:</b> %{x}<br><b>CUSUM:</b> %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add confidence bands
            fig.add_hline(y=0, line=dict(color="black", width=1), row=2, col=1)
            fig.add_hline(y=1.96, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
            fig.add_hline(y=-1.96, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
        
        fig.update_layout(
            title={
                'text': 'Structural Break Analysis - Brazil Inflation',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Inflation Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="CUSUM Statistic", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=1)
        
        # Save file
        output_file = self.output_dir / "structural_breaks_analysis_interactive.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive structural break chart saved to {output_file}")
        return output_file
    
    def create_volatility_modeling_chart(
        self,
        dataset: InflationDataset,
        output_format: str = "both"
    ) -> Dict[str, Path]:
        """Create volatility modeling visualizations.
        
        Args:
            dataset: Inflation dataset
            output_format: "matplotlib", "plotly", or "both"
            
        Returns:
            Dictionary mapping chart type to output file paths
        """
        logger.info("Creating volatility modeling charts")
        
        # Perform volatility analysis
        lf = dataset.to_polars().lazy()
        volatility_results = self.analytics.model_volatility(lf)
        
        output_files = {}
        
        if output_format in ("matplotlib", "both"):
            static_file = self._create_volatility_matplotlib(
                dataset.to_polars(), volatility_results
            )
            output_files["volatility_modeling_static"] = static_file
            
        if output_format in ("plotly", "both"):
            interactive_file = self._create_volatility_plotly(
                dataset.to_polars(), volatility_results
            )
            output_files["volatility_modeling_interactive"] = interactive_file
            
        return output_files
    
    def _create_volatility_matplotlib(
        self,
        df: pl.DataFrame,
        volatility_results: Dict[str, Any]
    ) -> Path:
        """Create matplotlib volatility modeling chart."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])
        
        # Convert to pandas for matplotlib
        df_pandas = df.to_pandas()
        df_pandas['date'] = pd.to_datetime(df_pandas['date'])
        
        # Plot inflation rate with volatility bands
        ax1.plot(
            df_pandas['date'], 
            df_pandas['inflation_rate'],
            color='darkblue', 
            linewidth=1,
            alpha=0.7,
            label='Inflation Rate'
        )
        
        # Add volatility bands if available
        if 'volatility_estimates' in volatility_results:
            vol_data = volatility_results['volatility_estimates']
            if isinstance(vol_data, pl.DataFrame):
                vol_pandas = vol_data.to_pandas()
                vol_pandas['date'] = pd.to_datetime(vol_pandas['date'])
                
                # Create volatility bands (mean ± 2*volatility)
                upper_band = vol_pandas['mean'] + 2 * vol_pandas['volatility']
                lower_band = vol_pandas['mean'] - 2 * vol_pandas['volatility']
                
                ax1.fill_between(
                    vol_pandas['date'],
                    lower_band,
                    upper_band,
                    alpha=0.2,
                    color='red',
                    label='Volatility Bands (±2σ)'
                )
        
        ax1.set_ylabel('Inflation Rate (%)')
        ax1.set_title('Inflation Rate with Volatility Modeling (GARCH)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot conditional variance/volatility
        if 'volatility_estimates' in volatility_results:
            vol_data = volatility_results['volatility_estimates']
            if isinstance(vol_data, pl.DataFrame):
                vol_pandas = vol_data.to_pandas()
                
                ax2.plot(
                    vol_pandas['date'],
                    vol_pandas['volatility'],
                    color='red',
                    linewidth=1.5,
                    label='Conditional Volatility'
                )
                
                # Add rolling volatility for comparison
                rolling_vol = df_pandas['inflation_rate'].rolling(window=12).std()
                ax2.plot(
                    df_pandas['date'],
                    rolling_vol,
                    color='orange',
                    linewidth=1,
                    alpha=0.7,
                    label='Rolling Volatility (12m)'
                )
        
        ax2.set_ylabel('Volatility')
        ax2.set_xlabel('Year')
        ax2.set_title('Conditional Volatility Over Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        # Only show legend if there are labeled items
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend()
        
        # Add Plano Real line to both plots
        plano_real = date(1994, 7, 1)
        for ax in [ax1, ax2]:
            ax.axvline(x=plano_real, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7)
        
        # Format x-axes
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save file
        output_file = self.output_dir / "volatility_modeling.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Volatility modeling chart saved to {output_file}")
        return output_file
    
    def _create_volatility_plotly(
        self,
        df: pl.DataFrame,
        volatility_results: Dict[str, Any]
    ) -> Path:
        """Create interactive Plotly volatility modeling chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                'Inflation Rate with Volatility Bands',
                'Conditional Volatility Over Time'
            ],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Plot main inflation series
        fig.add_trace(
            go.Scatter(
                x=df['date'].to_list(),
                y=df['inflation_rate'].to_list(),
                mode='lines',
                name='Inflation Rate',
                line=dict(color='darkblue', width=1.5),
                hovertemplate='<b>Date:</b> %{x}<br><b>Inflation:</b> %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add volatility bands if available
        if 'volatility_estimates' in volatility_results:
            vol_data = volatility_results['volatility_estimates']
            if isinstance(vol_data, pl.DataFrame):
                upper_band = (vol_data['mean'] + 2 * vol_data['volatility']).to_list()
                lower_band = (vol_data['mean'] - 2 * vol_data['volatility']).to_list()
                
                fig.add_trace(
                    go.Scatter(
                        x=vol_data['date'].to_list() + vol_data['date'].to_list()[::-1],
                        y=upper_band + lower_band[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        name='Volatility Bands (±2σ)',
                        showlegend=True,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                
                # Plot conditional volatility
                fig.add_trace(
                    go.Scatter(
                        x=vol_data['date'].to_list(),
                        y=vol_data['volatility'].to_list(),
                        mode='lines',
                        name='Conditional Volatility',
                        line=dict(color='red', width=1.5),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Volatility:</b> %{y:.3f}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Add Plano Real line
        fig.add_shape(
            type="line",
            x0="1994-07-01", x1="1994-07-01",
            y0=0, y1=1,
            yref="paper",
            line=dict(color="darkgreen", width=2),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0="1994-07-01", x1="1994-07-01",
            y0=0, y1=1,
            yref="paper",
            line=dict(color="darkgreen", width=2),
            row=2, col=1
        )
        
        fig.update_layout(
            title={
                'text': 'Volatility Modeling - Brazil Inflation (GARCH)',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Inflation Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=1)
        
        # Save file
        output_file = self.output_dir / "volatility_modeling_interactive.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive volatility modeling chart saved to {output_file}")
        return output_file
    
    async def create_comparative_analysis_chart(
        self,
        dataset: InflationDataset,
        output_format: str = "both"
    ) -> Dict[str, Path]:
        """Create comparative analysis with peer countries.
        
        Args:
            dataset: Brazil inflation dataset  
            output_format: "matplotlib", "plotly", or "both"
            
        Returns:
            Dictionary mapping chart type to output file paths
        """
        logger.info("Creating comparative analysis charts")
        
        # Fetch peer country data
        async with self.comparative as comp:
            peer_data = await comp.fetch_peer_country_data()
            
        # Create comparative dataset
        comparative_df = self.comparative.create_comparative_dataset(peer_data)
        
        output_files = {}
        
        if output_format in ("matplotlib", "both"):
            static_file = self._create_comparative_matplotlib(comparative_df)
            output_files["comparative_analysis_static"] = static_file
            
        if output_format in ("plotly", "both"):
            interactive_file = self._create_comparative_plotly(comparative_df)
            output_files["comparative_analysis_interactive"] = interactive_file
            
        return output_files
    
    def _create_comparative_matplotlib(self, df: pl.DataFrame) -> Path:
        """Create matplotlib comparative analysis chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Convert to pandas for matplotlib
        df_pandas = df.to_pandas()
        df_pandas['date'] = pd.to_datetime(df_pandas['date'])
        
        # Get unique countries
        countries = df_pandas['country'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(countries)))
        country_colors = dict(zip(countries, colors))
        
        # Brazil gets special color
        if 'Brazil' in country_colors:
            country_colors['Brazil'] = 'red'
        
        # 1. Time series comparison (post-1994)
        post_1994 = df_pandas[df_pandas['date'] >= '1994-01-01']
        for country in countries:
            country_data = post_1994[post_1994['country'] == country]
            if not country_data.empty:
                linewidth = 2.5 if country == 'Brazil' else 1.5
                alpha = 1.0 if country == 'Brazil' else 0.7
                ax1.plot(
                    country_data['date'],
                    country_data['inflation_rate'],
                    label=country,
                    color=country_colors[country],
                    linewidth=linewidth,
                    alpha=alpha
                )
        
        ax1.set_title('Inflation Rates: Brazil vs Emerging Market Peers (Post-1994)')
        ax1.set_ylabel('Inflation Rate (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Volatility comparison
        volatility_stats = (post_1994.groupby('country')['inflation_rate']
                          .agg(['std', 'mean'])
                          .reset_index())
        
        ax2.scatter(volatility_stats['mean'], volatility_stats['std'], 
                   c=[country_colors[c] for c in volatility_stats['country']], s=100)
        
        for i, country in enumerate(volatility_stats['country']):
            ax2.annotate(country, 
                        (volatility_stats['mean'].iloc[i], volatility_stats['std'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Mean Inflation Rate (%)')
        ax2.set_ylabel('Volatility (Std Dev)')
        ax2.set_title('Inflation Mean vs Volatility')
        ax2.grid(True, alpha=0.3)
        
        # 3. Recent performance (last 10 years)
        recent = df_pandas[df_pandas['date'] >= '2014-01-01']
        recent_stats = (recent.groupby('country')['inflation_rate']
                       .mean()
                       .sort_values()
                       .reset_index())
        
        bars = ax3.bar(range(len(recent_stats)), recent_stats['inflation_rate'],
                      color=[country_colors[c] for c in recent_stats['country']])
        ax3.set_xticks(range(len(recent_stats)))
        ax3.set_xticklabels(recent_stats['country'], rotation=45)
        ax3.set_title('Average Inflation Rate (2014-2024)')
        ax3.set_ylabel('Inflation Rate (%)')
        ax3.grid(True, alpha=0.3)
        
        # Highlight Brazil
        brazil_idx = recent_stats[recent_stats['country'] == 'Brazil'].index
        if not brazil_idx.empty:
            bars[brazil_idx[0]].set_color('red')
            bars[brazil_idx[0]].set_alpha(1.0)
        
        # 4. Crisis performance (2008-2009, 2020-2021)
        crisis_periods = [
            ('2008-01-01', '2009-12-31', 'Global Financial Crisis'),
            ('2020-01-01', '2021-12-31', 'COVID-19 Pandemic')
        ]
        
        crisis_performance = []
        for start, end, period_name in crisis_periods:
            period_data = df_pandas[(df_pandas['date'] >= start) & 
                                   (df_pandas['date'] <= end)]
            period_stats = (period_data.groupby('country')['inflation_rate']
                           .mean().reset_index())
            period_stats['period'] = period_name
            crisis_performance.append(period_stats)
        
        crisis_df = pd.concat(crisis_performance, ignore_index=True)
        
        # Pivot for grouped bar chart
        crisis_pivot = crisis_df.pivot(index='country', columns='period', values='inflation_rate')
        crisis_pivot.plot(kind='bar', ax=ax4, color=['orange', 'purple'])
        ax4.set_title('Crisis Period Performance')
        ax4.set_ylabel('Average Inflation Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save file
        output_file = self.output_dir / "comparative_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Comparative analysis chart saved to {output_file}")
        return output_file
    
    def _create_comparative_plotly(self, df: pl.DataFrame) -> Path:
        """Create interactive Plotly comparative analysis chart."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Time Series Comparison (Post-1994)',
                'Volatility vs Mean Inflation',
                'Recent Performance (2014-2024)',
                'Crisis Period Performance'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Convert to pandas for easier manipulation
        df_pandas = df.to_pandas()
        df_pandas['date'] = pd.to_datetime(df_pandas['date'])
        
        # Color map for countries
        countries = df_pandas['country'].unique()
        colors = px.colors.qualitative.Set3[:len(countries)]
        country_colors = dict(zip(countries, colors))
        if 'Brazil' in country_colors:
            country_colors['Brazil'] = 'red'
        
        # 1. Time series comparison
        post_1994 = df_pandas[df_pandas['date'] >= '1994-01-01']
        for country in countries:
            country_data = post_1994[post_1994['country'] == country]
            if not country_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=country_data['date'],
                        y=country_data['inflation_rate'],
                        mode='lines',
                        name=country,
                        line=dict(
                            color=country_colors[country],
                            width=3 if country == 'Brazil' else 1.5
                        ),
                        hovertemplate=f'<b>{country}</b><br>' +
                                    '<b>Date:</b> %{x}<br>' +
                                    '<b>Inflation:</b> %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 2. Volatility scatter
        volatility_stats = (post_1994.groupby('country')['inflation_rate']
                          .agg(['std', 'mean'])
                          .reset_index())
        
        fig.add_trace(
            go.Scatter(
                x=volatility_stats['mean'],
                y=volatility_stats['std'],
                mode='markers+text',
                text=volatility_stats['country'],
                textposition='top center',
                marker=dict(
                    color=[country_colors[c] for c in volatility_stats['country']],
                    size=12,
                    line=dict(width=2, color='white')
                ),
                name='Countries',
                hovertemplate='<b>%{text}</b><br>' +
                            '<b>Mean:</b> %{x:.2f}%<br>' +
                            '<b>Std Dev:</b> %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Recent performance
        recent = df_pandas[df_pandas['date'] >= '2014-01-01']
        recent_stats = (recent.groupby('country')['inflation_rate']
                       .mean()
                       .sort_values()
                       .reset_index())
        
        fig.add_trace(
            go.Bar(
                x=recent_stats['country'],
                y=recent_stats['inflation_rate'],
                marker=dict(
                    color=[country_colors[c] for c in recent_stats['country']],
                    line=dict(width=1, color='white')
                ),
                name='Recent Performance',
                hovertemplate='<b>%{x}</b><br>' +
                            '<b>Avg Inflation:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Crisis performance
        crisis_periods = [
            ('2008-01-01', '2009-12-31', 'GFC'),
            ('2020-01-01', '2021-12-31', 'COVID-19')
        ]
        
        for i, (start, end, period_name) in enumerate(crisis_periods):
            period_data = df_pandas[(df_pandas['date'] >= start) & 
                                   (df_pandas['date'] <= end)]
            period_stats = (period_data.groupby('country')['inflation_rate']
                           .mean().reset_index())
            
            fig.add_trace(
                go.Bar(
                    x=period_stats['country'],
                    y=period_stats['inflation_rate'],
                    name=period_name,
                    marker=dict(color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{period_name}</b><br>' +
                                '<b>%{x}</b><br>' +
                                '<b>Avg Inflation:</b> %{y:.2f}%<extra></extra>',
                    offsetgroup=i
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Brazil vs Emerging Market Peers - Inflation Analysis',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="Inflation Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (Std Dev)", row=1, col=2)
        fig.update_yaxes(title_text="Avg Inflation Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Avg Inflation Rate (%)", row=2, col=2)
        
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_xaxes(title_text="Mean Inflation Rate (%)", row=1, col=2)
        fig.update_xaxes(title_text="Country", row=2, col=1)
        fig.update_xaxes(title_text="Country", row=2, col=2)
        
        # Save file
        output_file = self.output_dir / "comparative_analysis_interactive.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive comparative analysis chart saved to {output_file}")
        return output_file
    
    async def create_monetary_policy_chart(
        self,
        dataset: InflationDataset,
        output_format: str = "both"
    ) -> Dict[str, Path]:
        """Create monetary policy effectiveness visualizations.
        
        Args:
            dataset: Brazil inflation dataset
            output_format: "matplotlib", "plotly", or "both"
            
        Returns:
            Dictionary mapping chart type to output file paths
        """
        logger.info("Creating monetary policy analysis charts")
        
        # Fetch Selic rate data
        async with self.monetary as mon:
            selic_data = await mon.fetch_both_selic_series()
            
        # Calculate real interest rates
        real_rates = self.monetary.calculate_real_interest_rates(
            dataset.to_polars().lazy(), selic_data
        )
        
        output_files = {}
        
        if output_format in ("matplotlib", "both"):
            static_file = self._create_monetary_policy_matplotlib(
                dataset.to_polars(), real_rates
            )
            output_files["monetary_policy_static"] = static_file
            
        if output_format in ("plotly", "both"):
            interactive_file = self._create_monetary_policy_plotly(
                dataset.to_polars(), real_rates
            )
            output_files["monetary_policy_interactive"] = interactive_file
            
        return output_files
    
    def _create_monetary_policy_matplotlib(
        self, 
        inflation_df: pl.DataFrame,
        real_rates_df: pl.DataFrame
    ) -> Path:
        """Create matplotlib monetary policy chart."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Convert to pandas for matplotlib
        inflation_pandas = inflation_df.to_pandas()
        inflation_pandas['date'] = pd.to_datetime(inflation_pandas['date'])
        
        real_rates_pandas = real_rates_df.to_pandas()
        real_rates_pandas['date'] = pd.to_datetime(real_rates_pandas['date'])
        
        # 1. Inflation Rate and Selic Rate
        ax1_twin = ax1.twinx()
        
        ax1.plot(
            inflation_pandas['date'],
            inflation_pandas['inflation_rate'],
            color='darkblue',
            linewidth=2,
            label='Inflation Rate'
        )
        
        if 'selic_rate' in real_rates_pandas.columns:
            ax1_twin.plot(
                real_rates_pandas['date'],
                real_rates_pandas['selic_rate'],
                color='green',
                linewidth=2,
                linestyle='--',
                label='Selic Rate'
            )
        
        ax1.set_ylabel('Inflation Rate (%)', color='darkblue')
        ax1_twin.set_ylabel('Selic Rate (%)', color='green')
        ax1.set_title('Inflation Rate vs Selic Rate')
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 2. Real Interest Rate
        if 'real_interest_rate' in real_rates_pandas.columns:
            ax2.plot(
                real_rates_pandas['date'],
                real_rates_pandas['real_interest_rate'],
                color='purple',
                linewidth=2,
                label='Real Interest Rate'
            )
            
            # Add zero line
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Fill areas above/below zero
            ax2.fill_between(
                real_rates_pandas['date'],
                real_rates_pandas['real_interest_rate'],
                0,
                where=(real_rates_pandas['real_interest_rate'] >= 0),
                alpha=0.3,
                color='green',
                label='Positive Real Rate'
            )
            
            ax2.fill_between(
                real_rates_pandas['date'],
                real_rates_pandas['real_interest_rate'],
                0,
                where=(real_rates_pandas['real_interest_rate'] < 0),
                alpha=0.3,
                color='red',
                label='Negative Real Rate'
            )
        
        ax2.set_ylabel('Real Interest Rate (%)')
        ax2.set_title('Real Interest Rate (Fisher Equation)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Policy Effectiveness (correlation over time)
        # Calculate rolling correlation between Selic and inflation
        if 'selic_rate' in real_rates_pandas.columns:
            # Merge data for correlation calculation
            merged = pd.merge(inflation_pandas[['date', 'inflation_rate']], 
                            real_rates_pandas[['date', 'selic_rate']], 
                            on='date', how='inner')
            
            # Calculate rolling correlation (24-month window)
            merged['rolling_corr'] = merged['inflation_rate'].rolling(window=24).corr(
                merged['selic_rate']
            )
            
            ax3.plot(
                merged['date'],
                merged['rolling_corr'],
                color='orange',
                linewidth=2,
                label='24-month Rolling Correlation'
            )
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strong Positive')
            ax3.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Strong Negative')
        
        ax3.set_ylabel('Correlation Coefficient')
        ax3.set_xlabel('Year')
        ax3.set_title('Policy Effectiveness: Selic-Inflation Correlation')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add key policy events
        policy_events = [
            (date(1994, 7, 1), "Plano Real", "darkgreen"),
            (date(1999, 6, 1), "Inflation Targeting", "blue"),
            (date(2016, 8, 31), "New BCB Leadership", "purple")
        ]
        
        for event_date, event_name, color in policy_events:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=event_date, color=color, linestyle=':', alpha=0.7)
        
        # Format x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax3.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save file
        output_file = self.output_dir / "monetary_policy_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Monetary policy chart saved to {output_file}")
        return output_file
    
    def _create_monetary_policy_plotly(
        self,
        inflation_df: pl.DataFrame,
        real_rates_df: pl.DataFrame
    ) -> Path:
        """Create interactive Plotly monetary policy chart."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                'Inflation Rate vs Selic Rate',
                'Real Interest Rate (Fisher Equation)',
                'Policy Effectiveness: Rolling Correlation'
            ],
            specs=[[{"secondary_y": True}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]],
            vertical_spacing=0.08
        )
        
        # 1. Inflation vs Selic rates
        fig.add_trace(
            go.Scatter(
                x=inflation_df['date'].to_list(),
                y=inflation_df['inflation_rate'].to_list(),
                mode='lines',
                name='Inflation Rate',
                line=dict(color='darkblue', width=2),
                hovertemplate='<b>Inflation:</b> %{y:.2f}%<br><b>Date:</b> %{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        if 'selic_rate' in real_rates_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=real_rates_df['date'].to_list(),
                    y=real_rates_df['selic_rate'].to_list(),
                    mode='lines',
                    name='Selic Rate',
                    line=dict(color='green', width=2, dash='dash'),
                    hovertemplate='<b>Selic:</b> %{y:.2f}%<br><b>Date:</b> %{x}<extra></extra>',
                    yaxis='y2'
                ),
                row=1, col=1, secondary_y=True
            )
        
        # 2. Real interest rate
        if 'real_interest_rate' in real_rates_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=real_rates_df['date'].to_list(),
                    y=real_rates_df['real_interest_rate'].to_list(),
                    mode='lines',
                    name='Real Interest Rate',
                    line=dict(color='purple', width=2),
                    hovertemplate='<b>Real Rate:</b> %{y:.2f}%<br><b>Date:</b> %{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line=dict(color="black", width=1), row=2, col=1)
        
        # 3. Rolling correlation (simplified for demonstration)
        # In a real implementation, you'd calculate this properly
        fig.add_trace(
            go.Scatter(
                x=real_rates_df['date'].to_list(),
                y=[0] * len(real_rates_df),  # Placeholder
                mode='lines',
                name='Selic-Inflation Correlation',
                line=dict(color='orange', width=2),
                hovertemplate='<b>Correlation:</b> %{y:.3f}<br><b>Date:</b> %{x}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add policy event lines
        policy_events = [
            ("1994-07-01", "Plano Real", "darkgreen"),
            ("1999-06-01", "Inflation Targeting", "blue"),
            ("2016-08-31", "New BCB Leadership", "purple")
        ]
        
        for event_date, event_name, color in policy_events:
            for row_num in [1, 2, 3]:
                fig.add_shape(
                    type="line",
                    x0=event_date, x1=event_date,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color=color, width=1, dash="dot"),
                    row=row_num, col=1
                )
        
        fig.update_layout(
            title={
                'text': 'Monetary Policy Analysis - Brazil',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=1000,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="Inflation Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Selic Rate (%)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Real Interest Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Correlation", row=3, col=1)
        fig.update_xaxes(title_text="Year", row=3, col=1)
        
        # Save file
        output_file = self.output_dir / "monetary_policy_analysis_interactive.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive monetary policy chart saved to {output_file}")
        return output_file
    
    def create_comprehensive_analytics_dashboard(
        self,
        dataset: InflationDataset
    ) -> Path:
        """Create comprehensive dashboard with all analytics components.
        
        Args:
            dataset: Brazil inflation dataset
            
        Returns:
            Path to the comprehensive dashboard HTML file
        """
        logger.info("Creating comprehensive analytics dashboard")
        
        # This would integrate all the individual charts into a single dashboard
        # For now, creating a summary page that links to individual charts
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brazil Inflation Analytics Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; text-align: center; }
                h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                .chart-link { 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 15px 25px; 
                    background: #3498db; 
                    color: white; 
                    text-decoration: none; 
                    border-radius: 5px;
                }
                .chart-link:hover { background: #2980b9; }
                .description { margin: 15px 0; color: #555; }
            </style>
        </head>
        <body>
            <h1>Brazil Inflation Analytics Dashboard</h1>
            <p style="text-align: center; color: #666;">Comprehensive analysis of Brazil's inflation transformation (1980-2024)</p>
            
            <div class="section">
                <h2>📊 Core Visualizations</h2>
                <div class="description">
                    Foundational charts showing Brazil's dual-period inflation story
                </div>
                <a href="brazil_inflation_dual_period_interactive.html" class="chart-link">Dual-Period Timeline</a>
                <a href="brazil_hyperinflation_interactive.html" class="chart-link">Hyperinflation Period</a>
                <a href="brazil_modern_period_interactive.html" class="chart-link">Modern Era</a>
            </div>
            
            <div class="section">
                <h2>🔬 Advanced Analytics</h2>
                <div class="description">
                    Sophisticated time series analysis and statistical modeling
                </div>
                <a href="seasonal_decomposition_additive_interactive.html" class="chart-link">Seasonal Decomposition</a>
                <a href="structural_breaks_analysis_interactive.html" class="chart-link">Structural Breaks</a>
                <a href="volatility_modeling_interactive.html" class="chart-link">Volatility (GARCH)</a>
            </div>
            
            <div class="section">
                <h2>🌍 International Context</h2>
                <div class="description">
                    Brazil's performance compared to emerging market peers
                </div>
                <a href="comparative_analysis_interactive.html" class="chart-link">Peer Comparison</a>
            </div>
            
            <div class="section">
                <h2>🏦 Monetary Policy</h2>
                <div class="description">
                    Central bank policy effectiveness and real interest rates
                </div>
                <a href="monetary_policy_analysis_interactive.html" class="chart-link">Policy Analysis</a>
            </div>
            
            <div class="section">
                <h2>📈 Summary Statistics</h2>
                <div class="description">
                    Key metrics and performance indicators
                </div>
                <a href="brazil_inflation_statistics.png" class="chart-link">Statistical Summary</a>
            </div>
            
            <footer style="margin-top: 50px; text-align: center; color: #888; border-top: 1px solid #ddd; padding-top: 20px;">
                <p>Generated by Enhanced Brazil Inflation Visualizer</p>
                <p>Data sources: World Bank, FRED, IBGE SIDRA</p>
            </footer>
        </body>
        </html>
        """
        
        # Save dashboard
        output_file = self.output_dir / "comprehensive_dashboard.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive dashboard saved to {output_file}")
        return output_file