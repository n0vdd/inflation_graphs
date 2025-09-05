"""Interactive Dashboard for Brazil Inflation Analytics.

Plotly Dash application providing comprehensive interactive visualization
of Brazil's inflation data with advanced analytics components.
"""

import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import polars as pl
import pandas as pd
import numpy as np

from .enhanced_visualizer import EnhancedBrazilInflationVisualizer
from .advanced_analytics import AdvancedInflationAnalytics
from .comparative_analysis import ComparativeInflationAnalysis
from .monetary_policy import MonetaryPolicyAnalyzer
from .models import InflationDataset, DataSource
from .config import Settings

logger = logging.getLogger(__name__)


class BrazilInflationDashboard:
    """Interactive dashboard for Brazil inflation analytics."""
    
    def __init__(self, config: Optional[Settings] = None):
        """Initialize dashboard with configuration."""
        self.config = config or Settings()
        self.visualizer = EnhancedBrazilInflationVisualizer(self.config)
        self.analytics = AdvancedInflationAnalytics(self.config)
        self.comparative = ComparativeInflationAnalysis(self.config)
        self.monetary = MonetaryPolicyAnalyzer(self.config)
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
            ]
        )
        
        # Cache for data and analysis results
        self._data_cache = {}
        self._analysis_cache = {}
        
    def create_layout(self) -> html.Div:
        """Create the main dashboard layout."""
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(
                        "Brazil Inflation Analytics Dashboard",
                        className="text-center mb-4",
                        style={"color": "#2c3e50", "font-weight": "bold"}
                    ),
                    html.P(
                        "Comprehensive analysis of Brazil's inflation transformation (1980-2024)",
                        className="text-center text-muted mb-4",
                        style={"font-size": "1.1rem"}
                    ),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("📊 Dashboard Controls", className="mb-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Data Source:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='source-dropdown',
                                        options=[
                                            {'label': 'World Bank', 'value': 'worldbank'},
                                            {'label': 'FRED', 'value': 'fred'},
                                            {'label': 'IBGE', 'value': 'ibge'},
                                            {'label': 'All Sources', 'value': 'all'}
                                        ],
                                        value='all',
                                        clearable=False
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Time Period:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='period-dropdown',
                                        options=[
                                            {'label': 'Full Timeline (1980-2024)', 'value': 'full'},
                                            {'label': 'Hyperinflation (1980-1994)', 'value': 'hyperinflation'},
                                            {'label': 'Modern Era (1994-2024)', 'value': 'modern'},
                                            {'label': 'Recent (2010-2024)', 'value': 'recent'}
                                        ],
                                        value='full',
                                        clearable=False
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Analysis Type:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='analysis-dropdown',
                                        options=[
                                            {'label': 'Overview', 'value': 'overview'},
                                            {'label': 'Seasonal Analysis', 'value': 'seasonal'},
                                            {'label': 'Structural Breaks', 'value': 'breaks'},
                                            {'label': 'Volatility Modeling', 'value': 'volatility'},
                                            {'label': 'Comparative Analysis', 'value': 'comparative'},
                                            {'label': 'Monetary Policy', 'value': 'monetary'}
                                        ],
                                        value='overview',
                                        clearable=False
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Export Options:", className="fw-bold"),
                                    dbc.ButtonGroup([
                                        dbc.Button("📊 PNG", id="export-png", size="sm", color="primary"),
                                        dbc.Button("📈 SVG", id="export-svg", size="sm", color="secondary"),
                                        dbc.Button("📄 PDF", id="export-pdf", size="sm", color="success")
                                    ])
                                ], width=3)
                            ])
                        ])
                    ], className="mb-4")
                ])
            ]),
            
            # Key Metrics Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📈", className="card-title text-primary"),
                            html.H3(id="current-inflation", className="mb-0"),
                            html.P("Current Inflation", className="text-muted small")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊", className="card-title text-success"),
                            html.H3(id="avg-modern", className="mb-0"),
                            html.P("Modern Era Avg", className="text-muted small")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("⚡", className="card-title text-danger"),
                            html.H3(id="volatility-reduction", className="mb-0"),
                            html.P("Volatility Reduction", className="text-muted small")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🌍", className="card-title text-info"),
                            html.H3(id="peer-ranking", className="mb-0"),
                            html.P("Peer Ranking", className="text-muted small")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🎯", className="card-title text-warning"),
                            html.H3(id="inflation-target", className="mb-0"),
                            html.P("vs Target", className="text-muted small")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🏦", className="card-title text-secondary"),
                            html.H3(id="real-rate", className="mb-0"),
                            html.P("Real Interest Rate", className="text-muted small")
                        ])
                    ])
                ], width=2)
            ], className="mb-4"),
            
            # Main Content Area
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-content",
                        type="graph",
                        children=[
                            html.Div(id="main-content")
                        ]
                    )
                ])
            ]),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.Footer([
                        html.P([
                            "Data sources: ",
                            html.A("World Bank", href="https://data.worldbank.org", target="_blank"),
                            " • ",
                            html.A("FRED", href="https://fred.stlouisfed.org", target="_blank"),
                            " • ",
                            html.A("IBGE", href="https://sidra.ibge.gov.br", target="_blank")
                        ], className="text-center text-muted"),
                        html.P(
                            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            className="text-center text-muted small"
                        )
                    ])
                ])
            ])
        ], fluid=True)
    
    def create_overview_content(self, df: pl.DataFrame) -> html.Div:
        """Create overview dashboard content."""
        # Create dual-period chart
        fig = self._create_dual_period_overview_chart(df)
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=fig,
                        style={"height": "600px"},
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                        }
                    )
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🔍 Key Insights")),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Brazil experienced hyperinflation exceeding 2000% annually before 1994"),
                                html.Li("The Plano Real (July 1994) successfully stabilized inflation"),
                                html.Li("Modern era (post-1994) shows dramatically reduced volatility"),
                                html.Li("Inflation targeting (1999) further improved monetary policy effectiveness"),
                                html.Li("Recent performance aligns with emerging market peers")
                            ])
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("📊 Economic Transformation")),
                        dbc.CardBody([
                            self._create_transformation_metrics_table(df)
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def create_seasonal_analysis_content(self, df: pl.DataFrame) -> html.Div:
        """Create seasonal analysis content."""
        # Perform seasonal decomposition
        decomp_results = self._get_or_compute_seasonal_analysis(df)
        
        fig = self._create_seasonal_decomposition_chart(decomp_results)
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=fig,
                        style={"height": "800px"},
                        config={'displayModeBar': True, 'displaylogo': False}
                    )
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🌊 Seasonal Patterns")),
                        dbc.CardBody([
                            html.P("Seasonal decomposition reveals underlying patterns in Brazil's inflation:"),
                            html.Ul([
                                html.Li("Trend component shows the long-term economic transformation"),
                                html.Li("Seasonal patterns are more pronounced in the modern era"),
                                html.Li("Residual component highlights periods of economic stress"),
                                html.Li("X-13ARIMA-SEATS methodology provides robust decomposition")
                            ])
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def create_structural_breaks_content(self, df: pl.DataFrame) -> html.Div:
        """Create structural breaks analysis content."""
        break_results = self._get_or_compute_break_analysis(df)
        
        fig = self._create_structural_breaks_chart(df, break_results)
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=fig,
                        style={"height": "700px"},
                        config={'displayModeBar': True, 'displaylogo': False}
                    )
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("📈 Structural Break Analysis")),
                        dbc.CardBody([
                            html.P("Statistical tests identify significant regime changes:"),
                            self._create_breaks_summary_table(break_results)
                        ])
                    ])
                ], width=12)
            ])
        ])
    
    def _create_dual_period_overview_chart(self, df: pl.DataFrame) -> go.Figure:
        """Create the main dual-period overview chart."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                'Hyperinflation Period (Pre-1994) - Logarithmic Scale',
                'Modern Period (Post-1994) - Linear Scale'
            ],
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]
        )
        
        # Split data by period
        hyperinflation_data = df.filter(pl.col('period') == 'hyperinflation')
        modern_data = df.filter(pl.col('period').is_in(['modern', 'transition']))
        
        # Plot hyperinflation data with log scale
        if hyperinflation_data.height > 0:
            fig.add_trace(
                go.Scatter(
                    x=hyperinflation_data['date'].to_list(),
                    y=hyperinflation_data['inflation_rate'].to_list(),
                    mode='lines',
                    name='Hyperinflation Period',
                    line=dict(color='darkred', width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Inflation:</b> %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
            fig.update_yaxes(type="log", row=1, col=1)
        
        # Plot modern data with linear scale  
        if modern_data.height > 0:
            fig.add_trace(
                go.Scatter(
                    x=modern_data['date'].to_list(),
                    y=modern_data['inflation_rate'].to_list(),
                    mode='lines',
                    name='Modern Period',
                    line=dict(color='darkblue', width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Inflation:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add 12-month moving average for modern period
            modern_pandas = modern_data.to_pandas()
            modern_pandas['ma_12'] = modern_pandas['inflation_rate'].rolling(window=12, center=True).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=modern_pandas['date'],
                    y=modern_pandas['ma_12'],
                    mode='lines',
                    name='12-Month Moving Average',
                    line=dict(color='orange', width=3, dash='dash'),
                    hovertemplate='<b>Date:</b> %{x}<br><b>12m Avg:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add Plano Real line
        fig.add_shape(
            type="line",
            x0="1994-07-01", x1="1994-07-01",
            y0=0, y1=1,
            yref="paper",
            line=dict(color="darkgreen", width=3),
            row=1, col=1
        )
        
        # Add historical events for modern period
        events = [
            ("1999-01-01", "Inflation Targeting", "blue"),
            ("2008-09-15", "Global Crisis", "orange"),
            ("2020-03-11", "COVID-19", "red")
        ]
        
        for event_date, event_name, color in events:
            fig.add_shape(
                type="line",
                x0=event_date, x1=event_date,
                y0=0, y1=1,
                yref="paper",
                line=dict(color=color, width=2, dash="dash"),
                row=2, col=1
            )
        
        fig.update_layout(
            title={
                'text': 'Brazil Inflation: Dual-Period Analysis',
                'x': 0.5,
                'font': {'size': 18}
            },
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Inflation Rate (% per month, log scale)", row=1, col=1)
        fig.update_yaxes(title_text="Inflation Rate (% per month)", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=1)
        
        return fig
    
    def _create_seasonal_decomposition_chart(self, decomp_results: Dict) -> go.Figure:
        """Create seasonal decomposition chart."""
        components = ['original', 'trend', 'seasonal', 'residual']
        titles = ['Original Series', 'Trend Component', 'Seasonal Component', 'Residual Component']
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=titles,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        colors = ['darkblue', 'green', 'orange', 'red']
        
        for i, (component, color) in enumerate(zip(components, colors)):
            if component in decomp_results:
                df = decomp_results[component]
                
                fig.add_trace(
                    go.Scatter(
                        x=df['date'].to_list(),
                        y=df['value'].to_list(),
                        mode='lines',
                        name=titles[i],
                        line=dict(color=color, width=1.5),
                        hovertemplate=f'<b>{titles[i]}:</b> %{{y:.3f}}<br><b>Date:</b> %{{x}}<extra></extra>'
                    ),
                    row=i+1, col=1
                )
                
                # Use log scale for original and trend if needed
                if component in ['original', 'trend']:
                    max_val = df['value'].max()
                    if max_val > 100:
                        fig.update_yaxes(type="log", row=i+1, col=1)
        
        fig.update_layout(
            title={
                'text': 'Seasonal Decomposition Analysis',
                'x': 0.5,
                'font': {'size': 18}
            },
            showlegend=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_structural_breaks_chart(self, df: pl.DataFrame, break_results: Dict) -> go.Figure:
        """Create structural breaks chart."""
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
                fig.add_shape(
                    type="line",
                    x0=break_date, x1=break_date,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash"),
                    row=1, col=1
                )
        
        # Add CUSUM if available (placeholder for now)
        if 'cusum_stats' in break_results:
            # This would use actual CUSUM results
            fig.add_hline(y=0, line=dict(color="black", width=1), row=2, col=1)
            fig.add_hline(y=1.96, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
            fig.add_hline(y=-1.96, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
        
        fig.update_layout(
            title={
                'text': 'Structural Break Detection',
                'x': 0.5,
                'font': {'size': 18}
            },
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_transformation_metrics_table(self, df: pl.DataFrame) -> html.Table:
        """Create transformation metrics table."""
        # Calculate key metrics
        hyperinflation = df.filter(pl.col('period') == 'hyperinflation')
        modern = df.filter(pl.col('period') == 'modern')
        
        if hyperinflation.height > 0 and modern.height > 0:
            hyper_mean = hyperinflation['inflation_rate'].mean()
            hyper_std = hyperinflation['inflation_rate'].std()
            modern_mean = modern['inflation_rate'].mean()
            modern_std = modern['inflation_rate'].std()
            
            data = [
                ['Mean Inflation Rate', f'{hyper_mean:.1f}%', f'{modern_mean:.1f}%'],
                ['Volatility (Std Dev)', f'{hyper_std:.1f}%', f'{modern_std:.1f}%'],
                ['Maximum Monthly Rate', f'{hyperinflation["inflation_rate"].max():.1f}%', f'{modern["inflation_rate"].max():.1f}%'],
                ['Observations', f'{hyperinflation.height:,}', f'{modern.height:,}']
            ]
        else:
            data = [['No data available', '-', '-']]
        
        return html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Metric'),
                    html.Th('Hyperinflation Era'),
                    html.Th('Modern Era')
                ])
            ]),
            html.Tbody([
                html.Tr([html.Td(cell) for cell in row])
                for row in data
            ])
        ], className='table table-striped')
    
    def _create_breaks_summary_table(self, break_results: Dict) -> html.Table:
        """Create structural breaks summary table."""
        if 'break_dates' in break_results and break_results['break_dates']:
            data = []
            for i, break_date in enumerate(break_results['break_dates']):
                data.append([
                    f'Break {i+1}',
                    str(break_date),
                    'Significant regime change detected'
                ])
        else:
            data = [['No breaks detected', '-', '-']]
        
        return html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Break Point'),
                    html.Th('Date'),
                    html.Th('Description')
                ])
            ]),
            html.Tbody([
                html.Tr([html.Td(cell) for cell in row])
                for row in data
            ])
        ], className='table table-striped')
    
    def _get_or_compute_seasonal_analysis(self, df: pl.DataFrame) -> Dict:
        """Get or compute seasonal analysis results."""
        cache_key = 'seasonal_analysis'
        if cache_key not in self._analysis_cache:
            # Convert to LazyFrame for analytics
            lf = df.lazy()
            self._analysis_cache[cache_key] = self.analytics.perform_seasonal_decomposition(lf)
        return self._analysis_cache[cache_key]
    
    def _get_or_compute_break_analysis(self, df: pl.DataFrame) -> Dict:
        """Get or compute structural break analysis results."""
        cache_key = 'break_analysis'
        if cache_key not in self._analysis_cache:
            # Convert to LazyFrame for analytics
            lf = df.lazy()
            self._analysis_cache[cache_key] = self.analytics.detect_structural_breaks(lf)
        return self._analysis_cache[cache_key]
    
    def setup_callbacks(self):
        """Setup Dash callbacks for interactivity."""
        
        @self.app.callback(
            Output('main-content', 'children'),
            [Input('analysis-dropdown', 'value'),
             Input('source-dropdown', 'value'),
             Input('period-dropdown', 'value')]
        )
        def update_main_content(analysis_type, source, period):
            """Update main content based on selections."""
            # Load data based on selections (simplified for demo)
            # In real implementation, this would load from cache or recompute
            
            # Mock data for demonstration
            dates = pd.date_range('1980-01-01', '2024-01-01', freq='M')
            inflation_rates = np.random.lognormal(1, 1.5, len(dates))
            
            # Create mock periods
            periods = ['hyperinflation' if d.year < 1994 else 'modern' for d in dates]
            
            df = pl.DataFrame({
                'date': dates,
                'inflation_rate': inflation_rates,
                'period': periods
            })
            
            if analysis_type == 'overview':
                return self.create_overview_content(df)
            elif analysis_type == 'seasonal':
                return self.create_seasonal_analysis_content(df)
            elif analysis_type == 'breaks':
                return self.create_structural_breaks_content(df)
            else:
                return html.Div([
                    dbc.Alert(
                        f"Analysis type '{analysis_type}' is under development",
                        color="info"
                    )
                ])
        
        @self.app.callback(
            [Output('current-inflation', 'children'),
             Output('avg-modern', 'children'),
             Output('volatility-reduction', 'children'),
             Output('peer-ranking', 'children'),
             Output('inflation-target', 'children'),
             Output('real-rate', 'children')],
            [Input('source-dropdown', 'value')]
        )
        def update_metrics(source):
            """Update key metrics cards."""
            # Mock values for demonstration
            return ['3.2%', '5.1%', '-85%', '#3/8', '+0.8pp', '6.5%']
    
    def run(self, host='127.0.0.1', port=8050, debug=True):
        """Run the dashboard server."""
        self.app.layout = self.create_layout()
        self.setup_callbacks()
        
        logger.info(f"Starting Brazil Inflation Dashboard on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


# Factory function for easy dashboard creation
def create_dashboard(config: Optional[Settings] = None) -> BrazilInflationDashboard:
    """Create and configure the Brazil inflation dashboard."""
    return BrazilInflationDashboard(config)