"""Sectoral and Regional Visualization Module.

Creates specialized visualizations for sectoral breakdown and regional
inflation analysis across Brazil's metropolitan areas.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

from .visualizer import BrazilInflationVisualizer
from .config import Settings

logger = logging.getLogger(__name__)


class SectoralRegionalVisualizer:
    """Visualizer for sectoral breakdown and regional inflation analysis."""
    
    # IBGE sectoral categories
    SECTORAL_CATEGORIES = {
        'food_beverages': 'Food and Beverages',
        'housing': 'Housing',
        'household_articles': 'Household Articles', 
        'clothing': 'Clothing',
        'transportation': 'Transportation',
        'health_personal_care': 'Health and Personal Care',
        'personal_expenses': 'Personal Expenses',
        'education': 'Education',
        'communication': 'Communication'
    }
    
    # Brazilian metropolitan areas
    METROPOLITAN_AREAS = {
        'belem': 'Belém', 
        'fortaleza': 'Fortaleza',
        'recife': 'Recife',
        'salvador': 'Salvador',
        'belo_horizonte': 'Belo Horizonte',
        'vitoria': 'Vitória',
        'rio_de_janeiro': 'Rio de Janeiro',
        'sao_paulo': 'São Paulo',
        'curitiba': 'Curitiba',
        'porto_alegre': 'Porto Alegre',
        'campo_grande': 'Campo Grande',
        'cuiaba': 'Cuiabá',
        'brasilia': 'Brasília',
        'goiania': 'Goiânia',
        'aracaju': 'Aracaju'
    }
    
    def __init__(self, config: Optional[Settings] = None):
        """Initialize sectoral/regional visualizer."""
        self.config = config or Settings()
        self.output_dir = Path(self.config.storage.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _ensure_dataframe(self, data: Union[pl.DataFrame, Dict, None]) -> Optional[pl.DataFrame]:
        """Convert input data to Polars DataFrame if needed.
        
        Args:
            data: Input data (DataFrame, dict, or None)
            
        Returns:
            Polars DataFrame or None if input is None
        """
        if data is None:
            return None
            
        if isinstance(data, dict):
            try:
                # Convert dictionary to Polars DataFrame
                df = pl.DataFrame(data)
                logger.info(f"Converted dictionary with {len(data)} columns to DataFrame")
                return df
            except Exception as e:
                logger.error(f"Failed to convert dictionary to DataFrame: {e}")
                return None
                
        if isinstance(data, pl.DataFrame):
            return data
            
        # Handle other data types by trying to convert to DataFrame
        try:
            df = pl.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"Failed to convert {type(data)} to DataFrame: {e}")
            return None
        
    def create_sectoral_breakdown_chart(
        self,
        sectoral_data: Union[pl.DataFrame, Dict, None],
        output_format: str = "both"
    ) -> Dict[str, Path]:
        """Create sectoral breakdown visualizations.
        
        Args:
            sectoral_data: DataFrame with sectoral inflation data (or dict to convert)
            output_format: "matplotlib", "plotly", or "both"
            
        Returns:
            Dictionary mapping chart type to output file paths
        """
        logger.info("Creating sectoral breakdown charts")
        
        # Convert input to DataFrame if needed
        df = self._ensure_dataframe(sectoral_data)
        if df is None:
            logger.warning("No valid sectoral data provided, returning empty results")
            return {}
        
        output_files = {}
        
        if output_format in ("matplotlib", "both"):
            static_file = self._create_sectoral_breakdown_matplotlib(df)
            output_files["sectoral_breakdown_static"] = static_file
            
            heatmap_file = self._create_sectoral_heatmap_matplotlib(df)
            output_files["sectoral_heatmap_static"] = heatmap_file
            
        if output_format in ("plotly", "both"):
            interactive_file = self._create_sectoral_breakdown_plotly(df)
            output_files["sectoral_breakdown_interactive"] = interactive_file
            
        return output_files
    
    def _create_sectoral_breakdown_matplotlib(self, df: pl.DataFrame) -> Path:
        """Create matplotlib sectoral breakdown chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Ensure date column is properly parsed
        if 'date' in df.columns:
            try:
                # Check if date column needs parsing
                date_dtype = df.select(pl.col('date').dtype).item()
                if date_dtype not in [pl.Date, pl.Datetime]:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
            except Exception:
                # If parsing fails, try a different approach or assume it's already correct
                try:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
                except Exception:
                    pass
        
        # Get recent data (last 5 years) for analysis - use all data if date filtering fails
        try:
            max_date = df.select(pl.col('date').max()).item()
            recent_cutoff = max_date - pl.duration(days=5*365)  # Approximate 5 years
            recent_data = df.filter(pl.col('date') >= recent_cutoff)
            
            # If no recent data found, use all data
            if recent_data.shape[0] == 0:
                recent_data = df
        except Exception:
            # If date filtering fails, use all data
            recent_data = df
        
        # 1. Stacked area chart showing sector contributions over time
        sector_columns = [col for col in df.columns if col in self.SECTORAL_CATEGORIES.keys()]
        if sector_columns:
            # Extract date and sector data for plotting
            dates = recent_data.select(pl.col('date')).to_numpy().flatten()
            
            # Create stacked area chart
            sector_arrays = []
            for col in sector_columns:
                sector_arrays.append(recent_data.select(pl.col(col)).to_numpy().flatten())
            
            ax1.stackplot(
                dates,
                *sector_arrays,
                labels=[self.SECTORAL_CATEGORIES.get(col, col) for col in sector_columns],
                alpha=0.8
            )
            
            ax1.set_title('Sectoral Contributions to Inflation (Recent 5 Years)', fontsize=14)
            ax1.set_ylabel('Inflation Rate (%)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
        
        # 2. Average sector performance
        if sector_columns:
            # Calculate means for each sector using Polars
            sector_means_dict = {}
            for col in sector_columns:
                mean_val = recent_data.select(pl.col(col).mean()).item()
                sector_means_dict[col] = mean_val
            
            # Sort by values
            sorted_sectors = sorted(sector_means_dict.items(), key=lambda x: x[1])
            sector_names = [item[0] for item in sorted_sectors]
            sector_values = [item[1] for item in sorted_sectors]
            
            bars = ax2.barh(
                range(len(sector_values)),
                sector_values,
                color=plt.cm.Set3(np.arange(len(sector_values)))
            )
            
            ax2.set_yticks(range(len(sector_names)))
            ax2.set_yticklabels([self.SECTORAL_CATEGORIES.get(col, col) for col in sector_names])
            ax2.set_xlabel('Average Inflation Rate (%)')
            ax2.set_title('Average Sectoral Inflation (Recent 5 Years)', fontsize=14)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Color bars based on performance
            median_val = np.median(sector_values)
            for i, bar in enumerate(bars):
                if sector_values[i] > median_val:
                    bar.set_color('lightcoral')
                else:
                    bar.set_color('lightblue')
        
        # 3. Sector volatility analysis
        if sector_columns:
            # Calculate standard deviation for each sector using Polars
            sector_volatility_dict = {}
            for col in sector_columns:
                std_val = recent_data.select(pl.col(col).std()).item()
                sector_volatility_dict[col] = std_val
            
            # Sort by values
            sorted_volatility = sorted(sector_volatility_dict.items(), key=lambda x: x[1])
            volatility_names = [item[0] for item in sorted_volatility]
            volatility_values = [item[1] for item in sorted_volatility]
            
            ax3.bar(
                range(len(volatility_values)),
                volatility_values,
                color='orange',
                alpha=0.7
            )
            
            ax3.set_xticks(range(len(volatility_names)))
            ax3.set_xticklabels(
                [self.SECTORAL_CATEGORIES.get(col, col)[:8] + '...' if len(self.SECTORAL_CATEGORIES.get(col, col)) > 8 
                 else self.SECTORAL_CATEGORIES.get(col, col) for col in volatility_names],
                rotation=45, ha='right'
            )
            ax3.set_ylabel('Volatility (Std Dev)')
            ax3.set_title('Sectoral Inflation Volatility', fontsize=14)
            ax3.grid(True, alpha=0.3)
        
        # 4. Sector correlation matrix
        if sector_columns and len(sector_columns) > 2:
            # Calculate correlation matrix using Polars
            sector_df = recent_data.select(sector_columns)
            correlation_matrix = sector_df.corr()
            
            # Convert to numpy for plotting
            corr_values = correlation_matrix.to_numpy()
            
            im = ax4.imshow(corr_values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
            cbar.set_label('Correlation Coefficient')
            
            # Set ticks and labels
            ax4.set_xticks(range(len(sector_columns)))
            ax4.set_yticks(range(len(sector_columns)))
            ax4.set_xticklabels([self.SECTORAL_CATEGORIES.get(col, col)[:8] for col in sector_columns], rotation=45, ha='right')
            ax4.set_yticklabels([self.SECTORAL_CATEGORIES.get(col, col)[:8] for col in sector_columns])
            ax4.set_title('Sectoral Correlation Matrix', fontsize=14)
            
            # Add correlation values to cells
            for i in range(len(sector_columns)):
                for j in range(len(sector_columns)):
                    text = ax4.text(j, i, f'{corr_values[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        # Save file
        output_file = self.output_dir / "sectoral_breakdown_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Sectoral breakdown chart saved to {output_file}")
        return output_file
    
    def _create_sectoral_heatmap_matplotlib(self, df: pl.DataFrame) -> Path:
        """Create sectoral inflation heatmap over time."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Ensure date column is properly parsed
        if 'date' in df.columns:
            try:
                # Check if date column needs parsing
                date_dtype = df.select(pl.col('date').dtype).item()
                if date_dtype not in [pl.Date, pl.Datetime]:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
            except Exception:
                # If parsing fails, try a different approach or assume it's already correct
                try:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
                except Exception:
                    pass
        
        # Get sector columns
        sector_columns = [col for col in df.columns if col in self.SECTORAL_CATEGORIES.keys()]
        
        if sector_columns:
            # Create year-month column for aggregation
            df_with_month = df.with_columns(
                pl.col('date').dt.strftime('%Y-%m').alias('year_month')
            )
            
            # Sample data for demonstration (use every 3rd row to avoid overcrowding)
            sample_data = df_with_month.filter(pl.int_range(pl.len()) % 3 == 0)
            
            # Create heatmap data by selecting sector columns and transposing
            heatmap_df = sample_data.select(['year_month'] + sector_columns)
            
            # Get data for heatmap
            year_months = heatmap_df.select(pl.col('year_month')).to_numpy().flatten()
            heatmap_values = heatmap_df.select(sector_columns).to_numpy().T
            
            # Create heatmap
            sns.heatmap(
                heatmap_values,
                cmap='RdYlBu_r',
                center=0,
                ax=ax,
                cbar_kws={'label': 'Inflation Rate (%)'},
                yticklabels=[self.SECTORAL_CATEGORIES.get(col, col) for col in sector_columns],
                xticklabels=year_months[::max(1, len(year_months)//10)]  # Show only every 10th label
            )
            
            ax.set_title('Sectoral Inflation Heatmap Over Time', fontsize=16, pad=20)
            ax.set_xlabel('Year-Month')
            ax.set_ylabel('Sector')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save file
        output_file = self.output_dir / "sectoral_inflation_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Sectoral heatmap saved to {output_file}")
        return output_file
    
    def _create_sectoral_breakdown_plotly(self, df: pl.DataFrame) -> Path:
        """Create interactive Plotly sectoral breakdown chart."""
        # Ensure date column is properly parsed
        if 'date' in df.columns:
            try:
                # Check if date column needs parsing
                date_dtype = df.select(pl.col('date').dtype).item()
                if date_dtype not in [pl.Date, pl.Datetime]:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
            except Exception:
                # If parsing fails, try a different approach or assume it's already correct
                try:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
                except Exception:
                    pass
        
        # Get sector columns
        sector_columns = [col for col in df.columns if col in self.SECTORAL_CATEGORIES.keys()]
        
        if not sector_columns:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No sectoral data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
        else:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Sectoral Time Series',
                    'Average Sectoral Performance', 
                    'Volatility by Sector',
                    'Interactive Heatmap'
                ],
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Color palette for sectors
            colors = px.colors.qualitative.Set3[:len(sector_columns)]
            
            # 1. Time series for each sector
            dates = df.select(pl.col('date')).to_numpy().flatten()
            
            for i, sector in enumerate(sector_columns):
                sector_values = df.select(pl.col(sector)).to_numpy().flatten()
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=sector_values,
                        mode='lines',
                        name=self.SECTORAL_CATEGORIES.get(sector, sector),
                        line=dict(color=colors[i], width=2),
                        hovertemplate=f'<b>{self.SECTORAL_CATEGORIES.get(sector, sector)}</b><br>' +
                                    '<b>Date:</b> %{x}<br>' +
                                    '<b>Inflation:</b> %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # 2. Average performance (recent 5 years)
            try:
                max_date = df.select(pl.col('date').max()).item()
                recent_cutoff = max_date - pl.duration(days=5*365)  # Approximate 5 years
                recent_data = df.filter(pl.col('date') >= recent_cutoff)
                
                # If no recent data found, use all data
                if recent_data.shape[0] == 0:
                    recent_data = df
            except Exception:
                # If date filtering fails, use all data
                recent_data = df
            
            if recent_data.shape[0] > 0:
                # Calculate means for each sector using Polars
                sector_means_dict = {}
                for col in sector_columns:
                    mean_val = recent_data.select(pl.col(col).mean()).item()
                    sector_means_dict[col] = mean_val
                
                # Sort by values
                sorted_sectors = sorted(sector_means_dict.items(), key=lambda x: x[1])
                sector_names = [item[0] for item in sorted_sectors]
                sector_values = [item[1] for item in sorted_sectors]
                
                fig.add_trace(
                    go.Bar(
                        x=sector_values,
                        y=[self.SECTORAL_CATEGORIES.get(col, col) for col in sector_names],
                        orientation='h',
                        name='Avg Performance',
                        marker=dict(color=colors[:len(sector_names)]),
                        hovertemplate='<b>%{y}</b><br><b>Avg Inflation:</b> %{x:.2f}%<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # 3. Volatility analysis
            sector_volatility_dict = {}
            for col in sector_columns:
                std_val = df.select(pl.col(col).std()).item()
                sector_volatility_dict[col] = std_val
            
            # Sort by values
            sorted_volatility = sorted(sector_volatility_dict.items(), key=lambda x: x[1])
            volatility_names = [item[0] for item in sorted_volatility]
            volatility_values = [item[1] for item in sorted_volatility]
            
            fig.add_trace(
                go.Bar(
                    x=[self.SECTORAL_CATEGORIES.get(col, col) for col in volatility_names],
                    y=volatility_values,
                    name='Volatility',
                    marker=dict(color='orange'),
                    hovertemplate='<b>%{x}</b><br><b>Volatility:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 4. Recent heatmap data (simplified)
            if recent_data.shape[0] > 0 and len(sector_columns) > 1:
                # Sample recent data for heatmap (every 6th observation)
                sample_indices = list(range(0, recent_data.shape[0], 6))
                sample_recent = recent_data.filter(pl.int_range(pl.len()).is_in(sample_indices))
                
                # Get dates and format them
                sample_dates = sample_recent.select(pl.col('date')).to_numpy().flatten()
                date_strings = [str(d)[:7] for d in sample_dates]  # YYYY-MM format
                
                # Get heatmap data
                heatmap_data = sample_recent.select(sector_columns).to_numpy().T
                
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data,
                        x=date_strings,
                        y=[self.SECTORAL_CATEGORIES.get(col, col) for col in sector_columns],
                        colorscale='RdYlBu_r',
                        name='Heatmap',
                        hovertemplate='<b>%{y}</b><br><b>%{x}</b><br><b>Inflation:</b> %{z:.2f}%<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title={
                'text': 'Sectoral Inflation Analysis - Brazil',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Inflation Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Average Inflation Rate (%)", row=1, col=2)
        fig.update_xaxes(title_text="Sector", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (Std Dev)", row=2, col=1)
        
        # Save file
        output_file = self.output_dir / "sectoral_breakdown_interactive.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive sectoral breakdown saved to {output_file}")
        return output_file
    
    def create_regional_analysis_chart(
        self,
        regional_data: Union[pl.DataFrame, Dict, None],
        output_format: str = "both"
    ) -> Dict[str, Path]:
        """Create regional inflation analysis visualizations.
        
        Args:
            regional_data: DataFrame with regional inflation data (or dict to convert)
            output_format: "matplotlib", "plotly", or "both"
            
        Returns:
            Dictionary mapping chart type to output file paths
        """
        logger.info("Creating regional analysis charts")
        
        # Convert input to DataFrame if needed
        df = self._ensure_dataframe(regional_data)
        if df is None:
            logger.warning("No valid regional data provided, returning empty results")
            return {}
        
        output_files = {}
        
        if output_format in ("matplotlib", "both"):
            static_file = self._create_regional_analysis_matplotlib(df)
            output_files["regional_analysis_static"] = static_file
            
        if output_format in ("plotly", "both"):
            interactive_file = self._create_regional_analysis_plotly(df)
            output_files["regional_analysis_interactive"] = interactive_file
            
        return output_files
    
    def _create_regional_analysis_matplotlib(self, df: pl.DataFrame) -> Path:
        """Create matplotlib regional analysis chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Ensure date column is properly parsed
        if 'date' in df.columns:
            try:
                # Check if date column needs parsing
                date_dtype = df.select(pl.col('date').dtype).item()
                if date_dtype not in [pl.Date, pl.Datetime]:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
            except Exception:
                # If parsing fails, try a different approach or assume it's already correct
                try:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
                except Exception:
                    pass
        
        # Get regional columns
        regional_columns = [col for col in df.columns if col in self.METROPOLITAN_AREAS.keys()]
        
        if regional_columns:
            # Get recent data
            try:
                max_date = df.select(pl.col('date').max()).item()
                recent_cutoff = max_date - pl.duration(days=5*365)  # Approximate 5 years
                recent_data = df.filter(pl.col('date') >= recent_cutoff)
                
                # If no recent data found, use all data
                if recent_data.shape[0] == 0:
                    recent_data = df
            except Exception:
                # If date filtering fails, use all data
                recent_data = df
            
            # 1. Time series of major metropolitan areas
            major_areas = regional_columns[:6]  # Limit to top 6 for readability
            colors = plt.cm.tab10(np.arange(len(major_areas)))
            
            dates = recent_data.select(pl.col('date')).to_numpy().flatten()
            
            for i, area in enumerate(major_areas):
                area_values = recent_data.select(pl.col(area)).to_numpy().flatten()
                ax1.plot(
                    dates,
                    area_values,
                    label=self.METROPOLITAN_AREAS.get(area, area),
                    color=colors[i],
                    linewidth=2,
                    alpha=0.8
                )
            
            ax1.set_title('Regional Inflation: Major Metropolitan Areas (Recent 5 Years)', fontsize=14)
            ax1.set_ylabel('Inflation Rate (%)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. Average regional performance
            # Calculate means for each region using Polars
            regional_means_dict = {}
            for col in regional_columns:
                mean_val = recent_data.select(pl.col(col).mean()).item()
                regional_means_dict[col] = mean_val
            
            # Sort by values (descending)
            sorted_regions = sorted(regional_means_dict.items(), key=lambda x: x[1], reverse=True)
            region_names = [item[0] for item in sorted_regions]
            region_values = [item[1] for item in sorted_regions]
            
            bars = ax2.bar(
                range(len(region_values)),
                region_values,
                color='lightblue',
                alpha=0.8
            )
            
            ax2.set_xticks(range(len(region_names)))
            ax2.set_xticklabels(
                [self.METROPOLITAN_AREAS.get(area, area)[:8] for area in region_names],
                rotation=45, ha='right'
            )
            ax2.set_ylabel('Average Inflation Rate (%)')
            ax2.set_title('Average Regional Inflation (Recent 5 Years)', fontsize=14)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Highlight highest and lowest
            if len(bars) > 0:
                bars[0].set_color('lightcoral')  # Highest (first in sorted list)
                bars[-1].set_color('lightgreen')  # Lowest (last in sorted list)
            
            # 3. Regional dispersion over time
            # Calculate monthly dispersion across regions
            all_dates = df.select(pl.col('date')).to_numpy().flatten()
            monthly_dispersion = []
            
            for i in range(df.shape[0]):
                row_values = []
                for col in regional_columns:
                    val = df.select(pl.col(col)).to_numpy().flatten()[i]
                    if val is not None:
                        row_values.append(val)
                if len(row_values) > 1:
                    monthly_dispersion.append(np.std(row_values))
                else:
                    monthly_dispersion.append(0)
            
            ax3.plot(
                all_dates,
                monthly_dispersion,
                color='purple',
                linewidth=2,
                label='Regional Dispersion'
            )
            
            ax3.set_title('Regional Inflation Dispersion Over Time', fontsize=14)
            ax3.set_ylabel('Standard Deviation (%)')
            ax3.set_xlabel('Year')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. Regional ranking changes (simplified)
            # Show ranking for first and last year of recent data
            if recent_data.shape[0] > 1:
                first_row = recent_data.head(1)
                last_row = recent_data.tail(1)
                
                first_values = {col: first_row.select(pl.col(col)).item() for col in regional_columns}
                last_values = {col: last_row.select(pl.col(col)).item() for col in regional_columns}
                
                # Calculate rankings (higher values = lower rank numbers)
                first_sorted = sorted(first_values.items(), key=lambda x: x[1], reverse=True)
                last_sorted = sorted(last_values.items(), key=lambda x: x[1], reverse=True)
                
                first_ranking = {area: idx + 1 for idx, (area, _) in enumerate(first_sorted)}
                last_ranking = {area: idx + 1 for idx, (area, _) in enumerate(last_sorted)}
                
                # Create ranking change visualization
                areas_sample = regional_columns[:10]  # Limit for readability
                x_positions = np.arange(len(areas_sample))
                
                first_year = first_row.select(pl.col('date')).item().year
                last_year = last_row.select(pl.col('date')).item().year
                
                ax4.scatter(
                    x_positions - 0.2,
                    [first_ranking[area] for area in areas_sample],
                    color='lightblue',
                    s=60,
                    alpha=0.7,
                    label=f'{first_year}'
                )
                
                ax4.scatter(
                    x_positions + 0.2,
                    [last_ranking[area] for area in areas_sample],
                    color='lightcoral',
                    s=60,
                    alpha=0.7,
                    label=f'{last_year}'
                )
                
                # Connect with lines to show movement
                for i, area in enumerate(areas_sample):
                    ax4.plot(
                        [i - 0.2, i + 0.2],
                        [first_ranking[area], last_ranking[area]],
                        color='gray',
                        alpha=0.5,
                        linestyle='--'
                    )
                
                ax4.set_xticks(x_positions)
                ax4.set_xticklabels(
                    [self.METROPOLITAN_AREAS.get(area, area)[:6] for area in areas_sample],
                    rotation=45, ha='right'
                )
                ax4.set_ylabel('Inflation Ranking')
                ax4.set_title('Regional Ranking Changes', fontsize=14)
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.invert_yaxis()  # Lower rank numbers at top
        
        plt.tight_layout()
        
        # Save file
        output_file = self.output_dir / "regional_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Regional analysis chart saved to {output_file}")
        return output_file
    
    def _create_regional_analysis_plotly(self, df: pl.DataFrame) -> Path:
        """Create interactive Plotly regional analysis chart."""
        # Ensure date column is properly parsed
        if 'date' in df.columns:
            try:
                # Check if date column needs parsing
                date_dtype = df.select(pl.col('date').dtype).item()
                if date_dtype not in [pl.Date, pl.Datetime]:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
            except Exception:
                # If parsing fails, try a different approach or assume it's already correct
                try:
                    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
                except Exception:
                    pass
        
        # Get regional columns
        regional_columns = [col for col in df.columns if col in self.METROPOLITAN_AREAS.keys()]
        
        if not regional_columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No regional data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
        else:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Regional Time Series',
                    'Average Regional Performance',
                    'Regional Dispersion',
                    'Geographic Distribution'
                ]
            )
            
            # Colors for regions
            colors = px.colors.qualitative.Set3[:len(regional_columns)]
            
            # 1. Time series for major metropolitan areas
            dates = df.select(pl.col('date')).to_numpy().flatten()
            
            major_areas = regional_columns[:8]  # Limit for clarity
            for i, area in enumerate(major_areas):
                area_values = df.select(pl.col(area)).to_numpy().flatten()
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=area_values,
                        mode='lines',
                        name=self.METROPOLITAN_AREAS.get(area, area),
                        line=dict(color=colors[i], width=2),
                        hovertemplate=f'<b>{self.METROPOLITAN_AREAS.get(area, area)}</b><br>' +
                                    '<b>Date:</b> %{x}<br>' +
                                    '<b>Inflation:</b> %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # 2. Average performance (recent data)
            try:
                max_date = df.select(pl.col('date').max()).item()
                recent_cutoff = max_date - pl.duration(days=5*365)  # Approximate 5 years
                recent_data = df.filter(pl.col('date') >= recent_cutoff)
                
                # If no recent data found, use all data
                if recent_data.shape[0] == 0:
                    recent_data = df
            except Exception:
                # If date filtering fails, use all data
                recent_data = df
            
            if recent_data.shape[0] > 0:
                # Calculate means for each region using Polars
                regional_means_dict = {}
                for col in regional_columns:
                    mean_val = recent_data.select(pl.col(col).mean()).item()
                    regional_means_dict[col] = mean_val
                
                # Sort by values
                sorted_regions = sorted(regional_means_dict.items(), key=lambda x: x[1])
                region_names = [item[0] for item in sorted_regions]
                region_values = [item[1] for item in sorted_regions]
                
                fig.add_trace(
                    go.Bar(
                        x=region_values,
                        y=[self.METROPOLITAN_AREAS.get(area, area) for area in region_names],
                        orientation='h',
                        name='Avg Performance',
                        marker=dict(color='lightblue'),
                        hovertemplate='<b>%{y}</b><br><b>Avg Inflation:</b> %{x:.2f}%<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # 3. Regional dispersion
            # Calculate regional dispersion for each time point
            all_dates = df.select(pl.col('date')).to_numpy().flatten()
            regional_dispersion = []
            
            for i in range(df.shape[0]):
                row_values = []
                for col in regional_columns:
                    val = df.select(pl.col(col)).to_numpy().flatten()[i]
                    if val is not None:
                        row_values.append(val)
                if len(row_values) > 1:
                    regional_dispersion.append(np.std(row_values))
                else:
                    regional_dispersion.append(0)
            
            fig.add_trace(
                go.Scatter(
                    x=all_dates,
                    y=regional_dispersion,
                    mode='lines',
                    name='Regional Dispersion',
                    line=dict(color='purple', width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Dispersion:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 4. Recent regional comparison (scatter plot)
            if recent_data.shape[0] > 0:
                # Calculate means and volatility for each region
                recent_means_dict = {}
                recent_volatility_dict = {}
                
                for col in regional_columns:
                    mean_val = recent_data.select(pl.col(col).mean()).item()
                    std_val = recent_data.select(pl.col(col).std()).item()
                    recent_means_dict[col] = mean_val
                    recent_volatility_dict[col] = std_val
                
                mean_values = list(recent_means_dict.values())
                volatility_values = list(recent_volatility_dict.values())
                area_names = list(recent_means_dict.keys())
                
                fig.add_trace(
                    go.Scatter(
                        x=mean_values,
                        y=volatility_values,
                        mode='markers+text',
                        text=[self.METROPOLITAN_AREAS.get(area, area)[:3] for area in area_names],
                        textposition='top center',
                        name='Regional Comparison',
                        marker=dict(
                            size=12,
                            color=colors[:len(area_names)],
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate='<b>%{text}</b><br>' +
                                    '<b>Mean:</b> %{x:.2f}%<br>' +
                                    '<b>Volatility:</b> %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title={
                'text': 'Regional Inflation Analysis - Brazil Metropolitan Areas',
                'x': 0.5,
                'font': {'size': 18}
            },
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Inflation Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Average Inflation Rate (%)", row=1, col=2)
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Regional Dispersion", row=2, col=1)
        fig.update_xaxes(title_text="Mean Inflation Rate (%)", row=2, col=2)
        fig.update_yaxes(title_text="Volatility", row=2, col=2)
        
        # Save file
        output_file = self.output_dir / "regional_analysis_interactive.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Interactive regional analysis saved to {output_file}")
        return output_file