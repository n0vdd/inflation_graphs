"""Publication-Ready Export Pipeline.

Provides comprehensive export capabilities for all Brazil inflation
visualizations in multiple formats optimized for different use cases:
- Academic publication (high-resolution, vector formats)
- Web publication (optimized file sizes, responsive)
- Print media (CMYK color space, specific dimensions)
- Interactive presentations (HTML, embedded JavaScript)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from datetime import datetime
import zipfile

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_svg import FigureCanvasSVG
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image, ImageDraw, ImageFont
import polars as pl
import pandas as pd

from .enhanced_visualizer import EnhancedBrazilInflationVisualizer
from .sectoral_visualizer import SectoralRegionalVisualizer
from .models import InflationDataset
from .config import Settings

logger = logging.getLogger(__name__)


class PublicationExportPipeline:
    """Comprehensive export pipeline for publication-ready visualizations."""
    
    # Publication format specifications
    EXPORT_FORMATS = {
        'academic': {
            'dpi': 300,
            'formats': ['png', 'svg', 'pdf'],
            'dimensions': [(8, 6), (10, 8), (12, 9)],  # inches
            'color_space': 'RGB',
            'font_size': 12,
            'line_width_multiplier': 1.2
        },
        'web': {
            'dpi': 150,
            'formats': ['png', 'webp'],
            'dimensions': [(12, 8), (16, 10)],
            'color_space': 'sRGB',
            'font_size': 11,
            'line_width_multiplier': 1.0
        },
        'print': {
            'dpi': 300,
            'formats': ['pdf', 'eps'],
            'dimensions': [(8.5, 11), (11, 8.5)],  # Letter size
            'color_space': 'CMYK',
            'font_size': 10,
            'line_width_multiplier': 1.5
        },
        'presentation': {
            'dpi': 150,
            'formats': ['png', 'svg'],
            'dimensions': [(16, 9), (12, 6.75)],  # Widescreen
            'color_space': 'RGB',
            'font_size': 14,
            'line_width_multiplier': 2.0
        }
    }
    
    # Color palettes optimized for different output types
    COLOR_PALETTES = {
        'academic': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'accent': '#2ca02c',
            'neutral': '#7f7f7f',
            'background': '#ffffff',
            'grid': '#e0e0e0'
        },
        'print': {
            'primary': '#000080',
            'secondary': '#800000',
            'accent': '#008000',
            'neutral': '#404040',
            'background': '#ffffff',
            'grid': '#c0c0c0'
        },
        'web': {
            'primary': '#3498db',
            'secondary': '#e74c3c',
            'accent': '#2ecc71',
            'neutral': '#95a5a6',
            'background': '#ffffff',
            'grid': '#ecf0f1'
        }
    }
    
    def __init__(self, config: Optional[Settings] = None):
        """Initialize export pipeline."""
        self.config = config or Settings()
        self.output_dir = Path(self.config.storage.output_dir)
        self.export_dir = self.output_dir / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizers
        self.enhanced_viz = EnhancedBrazilInflationVisualizer(self.config)
        self.sectoral_viz = SectoralRegionalVisualizer(self.config)
        
        # Configure matplotlib for high-quality output
        self._configure_matplotlib()
        
        # Configure plotly for export
        self._configure_plotly()
        
    def _configure_matplotlib(self):
        """Configure matplotlib for publication-quality output."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.5,
            'lines.markersize': 6
        })
        
    def _configure_plotly(self):
        """Configure plotly for publication output."""
        # Set default theme for exports
        pio.templates.default = "plotly_white"
        
        # Configure export settings
        self.plotly_config = {
            'displayModeBar': False,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'brazil_inflation_chart',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
        
    def export_all_visualizations(
        self,
        dataset: InflationDataset,
        sectoral_data: Optional[pl.DataFrame] = None,
        regional_data: Optional[pl.DataFrame] = None,
        export_types: List[str] = ['academic', 'web'],
        include_metadata: bool = True
    ) -> Dict[str, Path]:
        """Export all visualizations in specified formats.
        
        Args:
            dataset: Main inflation dataset
            sectoral_data: Optional sectoral breakdown data
            regional_data: Optional regional inflation data
            export_types: List of export format types
            include_metadata: Whether to include metadata files
            
        Returns:
            Dictionary mapping export type to archive file path
        """
        logger.info(f"Starting comprehensive export for types: {export_types}")
        
        export_archives = {}
        
        for export_type in export_types:
            if export_type not in self.EXPORT_FORMATS:
                logger.warning(f"Unknown export type: {export_type}")
                continue
                
            logger.info(f"Exporting for {export_type} publication")
            
            # Create export subdirectory
            type_dir = self.export_dir / export_type
            type_dir.mkdir(parents=True, exist_ok=True)
            
            # Export all chart types
            exported_files = {}
            
            # 1. Core inflation visualizations
            core_files = self._export_core_visualizations(dataset, export_type, type_dir)
            exported_files.update(core_files)
            
            # 2. Advanced analytics
            analytics_files = self._export_analytics_visualizations(dataset, export_type, type_dir)
            exported_files.update(analytics_files)
            
            # 3. Sectoral analysis (if data available)
            if sectoral_data is not None:
                sectoral_files = self._export_sectoral_visualizations(sectoral_data, export_type, type_dir)
                exported_files.update(sectoral_files)
            
            # 4. Regional analysis (if data available)
            if regional_data is not None:
                regional_files = self._export_regional_visualizations(regional_data, export_type, type_dir)
                exported_files.update(regional_files)
            
            # 5. Create metadata if requested
            if include_metadata:
                metadata_file = self._create_export_metadata(dataset, export_type, type_dir)
                exported_files['metadata'] = metadata_file
            
            # 6. Create archive
            archive_file = self._create_export_archive(exported_files, export_type)
            export_archives[export_type] = archive_file
            
            logger.info(f"✅ {export_type} export completed: {len(exported_files)} files")
        
        # Create master archive with all types
        if len(export_archives) > 1:
            master_archive = self._create_master_archive(export_archives)
            export_archives['master'] = master_archive
        
        return export_archives
    
    def _export_core_visualizations(
        self, 
        dataset: InflationDataset,
        export_type: str,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export core inflation visualizations."""
        logger.info(f"Exporting core visualizations for {export_type}")
        
        format_spec = self.EXPORT_FORMATS[export_type]
        exported_files = {}
        
        # Set matplotlib parameters for this export type
        self._apply_format_style(export_type)
        
        # 1. Dual-period overview
        try:
            dual_files = self.enhanced_viz.create_separate_period_charts(
                dataset, output_format="matplotlib"
            )
            
            for chart_type, source_file in dual_files.items():
                if source_file.suffix == '.png':
                    exported_files.update(
                        self._export_matplotlib_chart(
                            source_file, f"dual_period_{chart_type}", format_spec, output_dir
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to export dual-period charts: {e}")
        
        # 2. Summary statistics
        try:
            stats_file = self.enhanced_viz.create_summary_statistics_chart(dataset)
            exported_files.update(
                self._export_matplotlib_chart(
                    stats_file, "summary_statistics", format_spec, output_dir
                )
            )
        except Exception as e:
            logger.error(f"Failed to export summary statistics: {e}")
        
        return exported_files
    
    def _export_analytics_visualizations(
        self,
        dataset: InflationDataset,
        export_type: str,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export advanced analytics visualizations."""
        logger.info(f"Exporting analytics visualizations for {export_type}")
        
        format_spec = self.EXPORT_FORMATS[export_type]
        exported_files = {}
        
        # 1. Seasonal decomposition
        try:
            seasonal_files = self.enhanced_viz.create_seasonal_decomposition_chart(
                dataset, model="additive", output_format="matplotlib"
            )
            
            for chart_type, source_file in seasonal_files.items():
                if source_file.suffix == '.png':
                    exported_files.update(
                        self._export_matplotlib_chart(
                            source_file, f"seasonal_{chart_type}", format_spec, output_dir
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to export seasonal analysis: {e}")
        
        # 2. Structural breaks
        try:
            break_files = self.enhanced_viz.create_structural_break_chart(
                dataset, output_format="matplotlib"
            )
            
            for chart_type, source_file in break_files.items():
                if source_file.suffix == '.png':
                    exported_files.update(
                        self._export_matplotlib_chart(
                            source_file, f"structural_{chart_type}", format_spec, output_dir
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to export structural break analysis: {e}")
        
        # 3. Volatility modeling
        try:
            volatility_files = self.enhanced_viz.create_volatility_modeling_chart(
                dataset, output_format="matplotlib"
            )
            
            for chart_type, source_file in volatility_files.items():
                if source_file.suffix == '.png':
                    exported_files.update(
                        self._export_matplotlib_chart(
                            source_file, f"volatility_{chart_type}", format_spec, output_dir
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to export volatility analysis: {e}")
        
        return exported_files
    
    def _export_sectoral_visualizations(
        self,
        sectoral_data: pl.DataFrame,
        export_type: str,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export sectoral analysis visualizations."""
        logger.info(f"Exporting sectoral visualizations for {export_type}")
        
        format_spec = self.EXPORT_FORMATS[export_type]
        exported_files = {}
        
        try:
            sectoral_files = self.sectoral_viz.create_sectoral_breakdown_chart(
                sectoral_data, output_format="matplotlib"
            )
            
            for chart_type, source_file in sectoral_files.items():
                if source_file.suffix == '.png':
                    exported_files.update(
                        self._export_matplotlib_chart(
                            source_file, f"sectoral_{chart_type}", format_spec, output_dir
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to export sectoral analysis: {e}")
        
        return exported_files
    
    def _export_regional_visualizations(
        self,
        regional_data: pl.DataFrame,
        export_type: str,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export regional analysis visualizations."""
        logger.info(f"Exporting regional visualizations for {export_type}")
        
        format_spec = self.EXPORT_FORMATS[export_type]
        exported_files = {}
        
        try:
            regional_files = self.sectoral_viz.create_regional_analysis_chart(
                regional_data, output_format="matplotlib"
            )
            
            for chart_type, source_file in regional_files.items():
                if source_file.suffix == '.png':
                    exported_files.update(
                        self._export_matplotlib_chart(
                            source_file, f"regional_{chart_type}", format_spec, output_dir
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to export regional analysis: {e}")
        
        return exported_files
    
    def _export_matplotlib_chart(
        self,
        source_file: Path,
        base_name: str,
        format_spec: Dict,
        output_dir: Path
    ) -> Dict[str, Path]:
        """Export matplotlib chart in multiple formats and dimensions."""
        exported_files = {}
        
        # Load source image
        try:
            source_img = Image.open(source_file)
        except Exception as e:
            logger.error(f"Failed to load source image {source_file}: {e}")
            return exported_files
        
        for fmt in format_spec['formats']:
            for i, (width_inch, height_inch) in enumerate(format_spec['dimensions']):
                # Calculate pixel dimensions
                dpi = format_spec['dpi']
                width_px = int(width_inch * dpi)
                height_px = int(height_inch * dpi)
                
                # Create dimension suffix
                dim_suffix = f"_{width_inch}x{height_inch}in"
                filename = f"{base_name}{dim_suffix}.{fmt}"
                output_file = output_dir / filename
                
                try:
                    # Resize image maintaining aspect ratio
                    resized_img = source_img.copy()
                    resized_img.thumbnail((width_px, height_px), Image.Resampling.LANCZOS)
                    
                    # Create new image with exact dimensions and white background
                    final_img = Image.new('RGB', (width_px, height_px), 'white')
                    
                    # Center the resized image
                    x_offset = (width_px - resized_img.width) // 2
                    y_offset = (height_px - resized_img.height) // 2
                    final_img.paste(resized_img, (x_offset, y_offset))
                    
                    # Add attribution/watermark if needed
                    if format_spec.get('add_attribution', False):
                        self._add_attribution(final_img)
                    
                    # Save in requested format
                    if fmt.lower() == 'png':
                        final_img.save(output_file, 'PNG', dpi=(dpi, dpi))
                    elif fmt.lower() == 'jpg' or fmt.lower() == 'jpeg':
                        final_img.save(output_file, 'JPEG', quality=95, dpi=(dpi, dpi))
                    elif fmt.lower() == 'webp':
                        final_img.save(output_file, 'WEBP', quality=90)
                    
                    exported_files[f"{base_name}_{fmt}_{i}"] = output_file
                    
                except Exception as e:
                    logger.error(f"Failed to export {filename}: {e}")
                    
        return exported_files
    
    def _apply_format_style(self, export_type: str):
        """Apply matplotlib style for specific export type."""
        format_spec = self.EXPORT_FORMATS[export_type]
        palette = self.COLOR_PALETTES.get(export_type, self.COLOR_PALETTES['academic'])
        
        plt.rcParams.update({
            'font.size': format_spec['font_size'],
            'lines.linewidth': 1.5 * format_spec['line_width_multiplier'],
            'axes.prop_cycle': plt.cycler('color', [
                palette['primary'], palette['secondary'], palette['accent']
            ]),
            'axes.facecolor': palette['background'],
            'figure.facecolor': palette['background'],
            'grid.color': palette['grid']
        })
        
    def _add_attribution(self, img: Image.Image):
        """Add attribution watermark to image."""
        draw = ImageDraw.Draw(img)
        
        # Attribution text
        text = "Brazil Inflation Analytics | Data: World Bank, FRED, IBGE"
        
        # Try to use a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Add text in bottom right
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = img.width - text_width - 20
        y = img.height - text_height - 10
        
        # Add semi-transparent background
        draw.rectangle([x-5, y-2, x+text_width+5, y+text_height+2], fill=(255, 255, 255, 180))
        
        # Add text
        draw.text((x, y), text, fill=(100, 100, 100), font=font)
        
    def _create_export_metadata(
        self,
        dataset: InflationDataset,
        export_type: str,
        output_dir: Path
    ) -> Path:
        """Create metadata file for export."""
        metadata = {
            'export_info': {
                'export_type': export_type,
                'export_timestamp': datetime.now().isoformat(),
                'export_specifications': self.EXPORT_FORMATS[export_type],
                'color_palette': self.COLOR_PALETTES.get(export_type, {})
            },
            'data_info': {
                'source': dataset.source.value,
                'fetch_timestamp': dataset.fetch_timestamp.isoformat(),
                'total_observations': len(dataset.data),
                'date_range': {
                    'start': dataset.data['date'].min().strftime('%Y-%m-%d'),
                    'end': dataset.data['date'].max().strftime('%Y-%m-%d')
                },
                'metadata': dataset.metadata
            },
            'visualization_info': {
                'methodology': {
                    'dual_period_approach': 'Logarithmic scale for hyperinflation (pre-1994), linear for modern era',
                    'seasonal_decomposition': 'X-13ARIMA-SEATS methodology',
                    'structural_breaks': 'Chow tests and CUSUM analysis',
                    'volatility_modeling': 'ARCH/GARCH estimation'
                },
                'key_events': [
                    {'date': '1994-07-01', 'name': 'Plano Real', 'description': 'Currency reform and stabilization'},
                    {'date': '1999-01-01', 'name': 'Inflation Targeting', 'description': 'Adoption of inflation targeting regime'},
                    {'date': '2008-09-15', 'name': 'Global Crisis', 'description': 'Lehman Brothers collapse'},
                    {'date': '2020-03-11', 'name': 'COVID-19', 'description': 'WHO declares pandemic'}
                ]
            },
            'citation': {
                'suggested_citation': 'Brazil Inflation Analytics Dashboard. Generated using World Bank, FRED, and IBGE data.',
                'data_sources': [
                    'World Bank: https://data.worldbank.org',
                    'FRED: https://fred.stlouisfed.org', 
                    'IBGE: https://sidra.ibge.gov.br'
                ]
            }
        }
        
        metadata_file = output_dir / "export_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata_file
    
    def _create_export_archive(
        self,
        exported_files: Dict[str, Path],
        export_type: str
    ) -> Path:
        """Create ZIP archive of exported files."""
        archive_name = f"brazil_inflation_{export_type}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        archive_path = self.export_dir / archive_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_key, file_path in exported_files.items():
                if file_path.exists():
                    # Create organized folder structure in archive
                    if 'metadata' in file_key:
                        arcname = f"metadata/{file_path.name}"
                    elif 'sectoral' in file_key:
                        arcname = f"sectoral_analysis/{file_path.name}"
                    elif 'regional' in file_key:
                        arcname = f"regional_analysis/{file_path.name}"
                    elif 'analytics' in file_key or any(x in file_key for x in ['seasonal', 'structural', 'volatility']):
                        arcname = f"advanced_analytics/{file_path.name}"
                    else:
                        arcname = f"core_visualizations/{file_path.name}"
                    
                    zipf.write(file_path, arcname)
        
        logger.info(f"Created export archive: {archive_path}")
        return archive_path
    
    def _create_master_archive(self, export_archives: Dict[str, Path]) -> Path:
        """Create master archive containing all export types."""
        master_name = f"brazil_inflation_complete_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        master_path = self.export_dir / master_name
        
        with zipfile.ZipFile(master_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for export_type, archive_path in export_archives.items():
                if export_type != 'master' and archive_path.exists():
                    zipf.write(archive_path, f"{export_type}/{archive_path.name}")
        
        logger.info(f"Created master export archive: {master_path}")
        return master_path
    
    def create_publication_ready_figures(
        self,
        dataset: InflationDataset,
        figures_config: Dict[str, Any]
    ) -> Dict[str, Path]:
        """Create specific figures optimized for publication.
        
        Args:
            dataset: Inflation dataset
            figures_config: Configuration for specific figures
            
        Returns:
            Dictionary mapping figure names to file paths
        """
        logger.info("Creating publication-ready figures")
        
        pub_dir = self.export_dir / "publication_figures"
        pub_dir.mkdir(parents=True, exist_ok=True)
        
        publication_figures = {}
        
        # Apply academic formatting
        self._apply_format_style('academic')
        
        # Create high-impact figures for publication
        for fig_name, fig_config in figures_config.items():
            try:
                if fig_name == 'main_transformation_chart':
                    fig_path = self._create_main_transformation_figure(dataset, fig_config, pub_dir)
                    publication_figures[fig_name] = fig_path
                elif fig_name == 'comparative_performance_chart':
                    # This would need comparative data
                    pass
                elif fig_name == 'policy_effectiveness_chart':
                    # This would need monetary policy data
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to create {fig_name}: {e}")
        
        return publication_figures
    
    def _create_main_transformation_figure(
        self,
        dataset: InflationDataset,
        config: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """Create the main transformation figure for publication."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
        
        # Convert data
        df = dataset.to_polars().to_pandas()
        df['date'] = pd.to_datetime(df['date'])
        
        # Split periods
        hyperinflation = df[df['period'] == 'hyperinflation']
        modern = df[df['period'].isin(['modern', 'transition'])]
        
        # Plot hyperinflation with log scale
        if not hyperinflation.empty:
            ax1.plot(hyperinflation['date'], hyperinflation['inflation_rate'], 
                    color='#d32f2f', linewidth=2, alpha=0.8)
            ax1.set_yscale('log')
            ax1.set_ylabel('Monthly Inflation Rate (%, log scale)', fontsize=12)
            ax1.set_title('A. Hyperinflation Period (1980-1994)', fontsize=14, fontweight='bold', loc='left')
            ax1.grid(True, alpha=0.3)
            
            # Add Plano Real annotation
            ax1.axvline(x=pd.Timestamp('1994-07-01'), color='#2e7d32', linestyle='-', linewidth=3)
            ax1.text(pd.Timestamp('1994-07-01'), ax1.get_ylim()[1]*0.3, 
                    '  Plano Real\n  July 1994', rotation=90, verticalalignment='center',
                    color='#2e7d32', fontweight='bold', fontsize=11)
        
        # Plot modern period with linear scale
        if not modern.empty:
            ax2.plot(modern['date'], modern['inflation_rate'], 
                    color='#1976d2', linewidth=2, alpha=0.8, label='Monthly Rate')
            
            # Add moving average
            modern['ma_12'] = modern['inflation_rate'].rolling(window=12, center=True).mean()
            ax2.plot(modern['date'], modern['ma_12'], 
                    color='#ff6f00', linewidth=3, alpha=0.9, label='12-Month Moving Average')
            
            ax2.set_ylabel('Monthly Inflation Rate (%)', fontsize=12)
            ax2.set_xlabel('Year', fontsize=12)
            ax2.set_title('B. Modern Period (1994-2024)', fontsize=14, fontweight='bold', loc='left')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            # Add key events
            events = [
                (pd.Timestamp('1999-01-01'), 'Inflation\nTargeting', '#1976d2'),
                (pd.Timestamp('2008-09-15'), 'Global\nCrisis', '#ff5722'),
                (pd.Timestamp('2020-03-11'), 'COVID-19', '#e91e63')
            ]
            
            for event_date, label, color in events:
                if modern['date'].min() <= event_date <= modern['date'].max():
                    ax2.axvline(x=event_date, color=color, linestyle='--', alpha=0.7, linewidth=2)
                    ax2.text(event_date, ax2.get_ylim()[1]*0.8, label, 
                            rotation=90, verticalalignment='top', color=color, 
                            fontsize=9, fontweight='bold', ha='right')
        
        plt.tight_layout()
        
        # Save in multiple formats
        base_name = "brazil_inflation_transformation"
        formats = ['png', 'svg', 'pdf']
        saved_files = []
        
        for fmt in formats:
            output_file = output_dir / f"{base_name}.{fmt}"
            if fmt == 'png':
                plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            elif fmt == 'svg':
                plt.savefig(output_file, format='svg', bbox_inches='tight', facecolor='white')
            elif fmt == 'pdf':
                plt.savefig(output_file, format='pdf', bbox_inches='tight', facecolor='white')
            saved_files.append(output_file)
        
        plt.close()
        
        logger.info(f"Created main transformation figure: {len(saved_files)} formats")
        return saved_files[0]  # Return PNG version as primary