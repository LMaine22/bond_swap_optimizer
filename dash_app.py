import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io
import os
import json
import tempfile
import base64
from datetime import datetime
import threading
import time
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set pandas options to suppress specific warnings
pd.set_option('future.no_silent_downcasting', True)
import traceback

# Import your existing modules
from src.config import SCENARIO_PRESETS, switch_scenario
from src.data_handler import load_and_prepare_data, pre_filter_bonds
from src.genetic_algorithm import evolve_nsga
from src.pruning import prune_passengers, compute_metrics
from src.tr_engine import compute_simple_tr_analysis

# Custom dark theme CSS
CUSTOM_CSS = """
/* Dark Financial Theme */
body {
    background-color: #0f1419 !important;
    color: #ffffff !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Main container */
.container-fluid {
    background-color: #0f1419 !important;
    min-height: 100vh;
}

/* Cards */
.card {
    background-color: #1a1f2e !important;
    border: 1px solid #2d3748 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
}

.card-header {
    background-color: #2d3748 !important;
    border-bottom: 1px solid #4a5568 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

.card-body {
    background-color: #1a1f2e !important;
    color: #e2e8f0 !important;
}

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.btn-primary:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3) !important;
}

.btn-primary:disabled {
    background: #374151 !important;
    color: #9ca3af !important;
}

/* Upload area */
.upload-area {
    border: 2px dashed #4a5568 !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%) !important;
    transition: all 0.3s ease !important;
}

.upload-area:hover {
    border-color: #3b82f6 !important;
    background: linear-gradient(135deg, #2d3748 0%, #374151 100%) !important;
}

/* Form controls */
.form-control, .form-select {
    background-color: #2d3748 !important;
    border: 1px solid #4a5568 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

.form-control:focus, .form-select:focus {
    background-color: #374151 !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    color: #ffffff !important;
}

/* Dropdowns */
.Select-control {
    background-color: #2d3748 !important;
    border: 1px solid #4a5568 !important;
    color: #ffffff !important;
}

/* Alerts */
.alert-success {
    background-color: #065f46 !important;
    border-color: #059669 !important;
    color: #d1fae5 !important;
}

.alert-danger {
    background-color: #7f1d1d !important;
    border-color: #dc2626 !important;
    color: #fecaca !important;
}

.alert-warning {
    background-color: #78350f !important;
    border-color: #f59e0b !important;
    color: #fef3c7 !important;
}

.alert-info {
    background-color: #1e3a8a !important;
    border-color: #3b82f6 !important;
    color: #dbeafe !important;
}

/* Tables */
.dash-table-container {
    background-color: #1a1f2e !important;
    color: #ffffff !important;
}

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
    background-color: #1a1f2e !important;
    color: #ffffff !important;
}

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
    background-color: #2d3748 !important;
    color: #ffffff !important;
    border: 1px solid #4a5568 !important;
}

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
    background-color: #1a1f2e !important;
    color: #e2e8f0 !important;
    border: 1px solid #374151 !important;
}

/* Badges */
.badge {
    border-radius: 6px !important;
    font-weight: 500 !important;
}

.badge.bg-info {
    background-color: #3b82f6 !important;
}

.badge.bg-success {
    background-color: #10b981 !important;
}

.badge.bg-warning {
    background-color: #f59e0b !important;
}

/* Tabs */
.nav-tabs {
    border-bottom: 1px solid #4a5568 !important;
}

.nav-tabs .nav-link {
    background-color: #2d3748 !important;
    border: 1px solid #4a5568 !important;
    color: #e2e8f0 !important;
}

.nav-tabs .nav-link.active {
    background-color: #3b82f6 !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
}

/* Metrics cards */
.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%) !important;
    border: 1px solid #374151 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}

.metric-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3) !important;
}

.metric-value {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}

.metric-label {
    font-size: 0.875rem !important;
    color: #9ca3af !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Progress indicators */
.progress {
    background-color: #374151 !important;
    border-radius: 8px !important;
}

.progress-bar {
    background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%) !important;
}

/* Icons */
.bi {
    margin-right: 0.5rem !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 8px !important;
    height: 8px !important;
}

::-webkit-scrollbar-track {
    background: #1a1f2e !important;
}

::-webkit-scrollbar-thumb {
    background: #4a5568 !important;
    border-radius: 4px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: #6b7280 !important;
}
"""

# Initialize Dash app with custom theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True
)

app.title = "Bond Swap Optimizer Pro"

# Inject custom CSS via index_string
app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
        {CUSTOM_CSS}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# Global variables for storing data
portfolio_data = None
optimization_results = None
current_scenario_config = None


def format_currency(value):
    """Format currency values"""
    try:
        return f"${float(value):,.0f}"
    except:
        return "$0"


def format_percent(value):
    """Format percentage values"""
    try:
        return f"{float(value):.2%}"
    except:
        return "0.00%"


def create_header():
    """Create the main header"""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="bi bi-graph-up-arrow me-3", style={'color': '#3b82f6'}),
                    "Bond Swap Optimizer Pro"
                ], className="text-center mb-2", style={'color': '#ffffff', 'font-weight': '700'}),
                html.P(
                    "Professional bond portfolio optimization and analysis platform",
                    className="text-center",
                    style={'color': '#9ca3af', 'font-size': '1.1rem', 'margin-bottom': '2rem'}
                )
            ])
        ])
    ])


def create_scenario_controls():
    """Create scenario selection and parameter controls"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="bi bi-sliders2"),
                "Scenario Configuration"
            ], className="mb-0", style={'color': '#ffffff'}),
            dbc.Badge("Configure swap parameters", color="info", className="ms-2")
        ]),
        dbc.CardBody([
            # Scenario selection
            html.Label("Scenario Type", className="form-label fw-bold mb-2", style={'color': '#e2e8f0'}),
            dcc.Dropdown(
                id='scenario-dropdown',
                options=[
                    {'label': '🏛️ Tax Loss Harvesting', 'value': 'tax_loss'},
                    {'label': '🛡️ Conservative Swap', 'value': 'conservative'},
                    {'label': '🧹 Yield Cleanup', 'value': 'yield_cleanup'},
                    {'label': '⚙️ Custom Configuration', 'value': 'custom'}
                ],
                value='yield_cleanup',
                className="mb-3",
                style={
                    'backgroundColor': '#2d3748',
                    'color': '#ffffff'
                }
            ),

            # Scenario description
            html.Div(id='scenario-description', className="mb-3"),

            # Collapsible custom parameters
            dbc.Collapse([
                html.Hr(style={'border-color': '#4a5568'}),
                html.H5("Custom Parameters", style={'color': '#3b82f6'}),

                # Swap size
                dbc.Row([
                    dbc.Col([
                        html.Label("Min Swap Size ($)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='min-swap-size', type='number', value=7500000, step=500000)
                    ], md=6),
                    dbc.Col([
                        html.Label("Max Swap Size ($)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='max-swap-size', type='number', value=100000000, step=1000000)
                    ], md=6)
                ], className="mb-3"),

                # Loss constraints
                dbc.Row([
                    dbc.Col([
                        html.Label("Min Loss ($)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='min-loss', type='number', value=0, step=50000)
                    ], md=6),
                    dbc.Col([
                        html.Label("Max Loss ($)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='max-loss', type='number', value=750000, step=100000)
                    ], md=6)
                ], className="mb-3"),

                # Recovery period
                dbc.Row([
                    dbc.Col([
                        html.Label("Min Recovery (months)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='min-recovery', type='number', value=0.0, step=0.5)
                    ], md=6),
                    dbc.Col([
                        html.Label("Max Recovery (months)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='max-recovery', type='number', value=18.0, step=1.0)
                    ], md=6)
                ], className="mb-3"),

                # Yield constraints
                dbc.Row([
                    dbc.Col([
                        html.Label("Target Buy Yield (%)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='target-buy-yield', type='number', value=5.5, step=0.1)
                    ], md=4),
                    dbc.Col([
                        html.Label("Min Sold Yield (%)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='min-sold-yield', type='number', value=1.5, step=0.1)
                    ], md=4),
                    dbc.Col([
                        html.Label("Max Sold Yield (%)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='max-sold-yield', type='number', value=4.0, step=0.1)
                    ], md=4)
                ], className="mb-3"),

                # Additional constraints
                dbc.Row([
                    dbc.Col([
                        html.Label("Max Individual Bond Yield (%)", className="form-label", style={'color': '#e2e8f0'}),
                        dbc.Input(id='max-individual-yield', type='number', value=5.0, step=0.1)
                    ], md=6),
                    dbc.Col([
                        dbc.Checklist(
                            id='enforce-min-loss',
                            options=[{'label': ' Enforce Minimum Loss (tax swaps)', 'value': 'enforce'}],
                            value=[],
                            className="mt-4",
                            style={'color': '#e2e8f0'}
                        )
                    ], md=6)
                ])
            ], id='custom-parameters-collapse', is_open=False)
        ])
    ], className="mb-4")


def create_file_upload():
    """Create file upload area"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="bi bi-cloud-upload"),
                "Portfolio Upload"
            ], className="mb-0", style={'color': '#ffffff'}),
            dbc.Badge("Upload bond portfolio data", color="success", className="ms-2")
        ]),
        dbc.CardBody([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.I(className="bi bi-cloud-upload-fill", style={'font-size': '3rem', 'color': '#3b82f6'}),
                    html.Div([
                        html.H5("Drag and Drop or Click to Select Files",
                                style={'color': '#ffffff', 'margin-top': '1rem'}),
                        html.P("Supports .parquet, .xlsx, .xls files", style={'color': '#9ca3af', 'margin-bottom': '0'})
                    ])
                ], className="text-center upload-area", style={'padding': '2rem'}),
                className="mb-3",
                multiple=False
            ),
            html.Div(id='upload-status', className="mt-3")
        ])
    ], className="mb-4")


def create_optimization_controls():
    """Create optimization trigger and progress"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="bi bi-cpu"),
                "Run Optimization"
            ], className="mb-0", style={'color': '#ffffff'}),
            dbc.Badge("Find optimal bond swaps", color="warning", className="ms-2")
        ]),
        dbc.CardBody([
            html.Div(id='portfolio-summary', className="mb-3"),
            dbc.Button(
                [
                    html.I(className="bi bi-play-fill me-2"),
                    "Find Optimal Swaps"
                ],
                id='run-optimization',
                color="primary",
                size="lg",
                className="w-100 mb-3",
                disabled=True,
                style={'font-weight': '600'}
            ),
            html.Div(id='optimization-progress'),
            html.Div(id='optimization-status')
        ])
    ], className="mb-4")


def create_results_section():
    """Create results display section"""
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="bi bi-graph-up"),
                    "Optimization Results"
                ], className="mb-0", style={'color': '#ffffff'}),
                dbc.Badge(id="results-badge", color="success", className="ms-2")
            ]),
            dbc.CardBody([
                # Summary metrics
                html.Div(id='results-summary', className="mb-4"),

                # Tabs for different views
                dbc.Tabs([
                    dbc.Tab(label="📊 All Options", tab_id="all-options"),
                    dbc.Tab(label="🔍 Option Details", tab_id="option-details"),
                    dbc.Tab(label="📈 Analysis Charts", tab_id="charts"),
                    dbc.Tab(label="💾 Downloads", tab_id="downloads")
                ], id="results-tabs", active_tab="all-options"),

                html.Div(id='results-content', className="mt-4")
            ])
        ])
    ], id='results-section', style={'display': 'none'})


# Main layout
app.layout = html.Div([
    # Data stores
    dcc.Store(id='portfolio-store'),
    dcc.Store(id='results-store'),
    dcc.Store(id='scenario-store'),
    dcc.Interval(id='optimization-interval', interval=1000, disabled=True),

    # Main container
    dbc.Container([
        # Header
        create_header(),

        # Main content
        dbc.Row([
            # Sidebar
            dbc.Col([
                create_scenario_controls(),
                create_file_upload(),
                create_optimization_controls()
            ], md=4),

            # Main content area
            dbc.Col([
                create_results_section()
            ], md=8)
        ])
    ], fluid=True, className="py-4")
], style={'backgroundColor': '#0f1419', 'minHeight': '100vh'})


# Callbacks

@app.callback(
    [Output('scenario-description', 'children'),
     Output('custom-parameters-collapse', 'is_open'),
     Output('min-swap-size', 'value'),
     Output('max-swap-size', 'value'),
     Output('min-loss', 'value'),
     Output('max-loss', 'value'),
     Output('min-recovery', 'value'),
     Output('max-recovery', 'value'),
     Output('target-buy-yield', 'value'),
     Output('min-sold-yield', 'value'),
     Output('max-sold-yield', 'value'),
     Output('max-individual-yield', 'value'),
     Output('enforce-min-loss', 'value')],
    Input('scenario-dropdown', 'value')
)
def update_scenario_config(scenario_type):
    """Update scenario configuration when dropdown changes"""
    if not scenario_type:
        return no_update

    config = SCENARIO_PRESETS[scenario_type]
    is_custom = scenario_type == 'custom'

    description = dbc.Alert([
        html.H6(f"{scenario_type.replace('_', ' ').title()} Scenario", className="mb-2", style={'color': '#ffffff'}),
        html.P(config['description'], className="mb-0", style={'color': '#e2e8f0'})
    ], color="info", className="border-0", style={
        'backgroundColor': '#1e3a8a',
        'borderLeft': '4px solid #3b82f6'
    })

    enforce_value = ['enforce'] if config['enforce_min_loss'] else []

    return (
        description,
        is_custom,
        config['min_swap_size'],
        config['max_swap_size'],
        config['min_loss'],
        config['max_loss'],
        config['min_recovery_months'],
        config['max_recovery_months'],
        config['target_buy_yield'],
        config.get('min_sold_wavg_yield', 0.0),
        config['max_sold_wavg_yield'],
        config['max_individual_bond_yield'],
        enforce_value
    )


@app.callback(
    [Output('portfolio-store', 'data'),
     Output('upload-status', 'children'),
     Output('run-optimization', 'disabled')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_file_upload(contents, filename):
    """Handle portfolio file upload"""
    if contents is None:
        return None, "", True

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Save to temporary file
        suffix = '.parquet' if filename.endswith('.parquet') else '.xlsx'

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(decoded)
            tmp_path = tmp_file.name

        # Load using existing function
        portfolio_df = load_and_prepare_data(tmp_path)
        os.unlink(tmp_path)  # Clean up

        if portfolio_df is not None and not portfolio_df.empty:
            # Store as JSON for Dash
            portfolio_json = portfolio_df.to_json(date_format='iso', orient='split')

            status = dbc.Alert([
                html.I(className="bi bi-check-circle-fill me-2"),
                html.Strong("Success! "),
                f"Loaded {len(portfolio_df)} bonds, Total Value: {format_currency(portfolio_df['market'].sum())}"
            ], color="success", className="border-0")

            return portfolio_json, status, False
        else:
            status = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                html.Strong("Error! "),
                "Could not load portfolio data"
            ], color="danger", className="border-0")
            return None, status, True

    except Exception as e:
        status = dbc.Alert([
            html.I(className="bi bi-exclamation-triangle-fill me-2"),
            html.Strong("Error! "),
            f"Failed to process file: {str(e)}"
        ], color="danger", className="border-0")
        return None, status, True


@app.callback(
    Output('portfolio-summary', 'children'),
    Input('portfolio-store', 'data')
)
def update_portfolio_summary(portfolio_json):
    """Update portfolio summary display"""
    if not portfolio_json:
        return ""

    try:
        portfolio_df = pd.read_json(portfolio_json, orient='split')

        return dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(f"{len(portfolio_df)}", className="metric-value", style={'color': '#3b82f6'}),
                    html.P("Total Bonds", className="metric-label")
                ], className="metric-card")
            ], md=4),
            dbc.Col([
                html.Div([
                    html.H4(format_currency(portfolio_df['market'].sum()), className="metric-value",
                            style={'color': '#10b981'}),
                    html.P("Market Value", className="metric-label")
                ], className="metric-card")
            ], md=4),
            dbc.Col([
                html.Div([
                    html.H4(format_currency(portfolio_df['par'].sum()), className="metric-value",
                            style={'color': '#f59e0b'}),
                    html.P("Par Value", className="metric-label")
                ], className="metric-card")
            ], md=4)
        ])
    except Exception as e:
        return dbc.Alert(f"Error displaying portfolio summary: {str(e)}", color="warning")


@app.callback(
    [Output('results-store', 'data'),
     Output('optimization-status', 'children'),
     Output('optimization-interval', 'disabled'),
     Output('results-section', 'style')],
    Input('run-optimization', 'n_clicks'),
    [State('portfolio-store', 'data'),
     State('scenario-dropdown', 'value'),
     State('min-swap-size', 'value'),
     State('max-swap-size', 'value'),
     State('min-loss', 'value'),
     State('max-loss', 'value'),
     State('min-recovery', 'value'),
     State('max-recovery', 'value'),
     State('target-buy-yield', 'value'),
     State('min-sold-yield', 'value'),
     State('max-sold-yield', 'value'),
     State('max-individual-yield', 'value'),
     State('enforce-min-loss', 'value')],
    prevent_initial_call=True
)
def run_optimization(n_clicks, portfolio_json, scenario_type, min_swap, max_swap, min_loss, max_loss,
                     min_recovery, max_recovery, target_yield, min_sold_yield, max_sold_yield,
                     max_individual_yield, enforce_min_loss):
    """Run the bond swap optimization"""
    if not n_clicks or not portfolio_json:
        return no_update

    try:
        # Load portfolio data
        portfolio_df = pd.read_json(portfolio_json, orient='split')

        # Update configuration
        import src.config as config
        config.MIN_SWAP_SIZE_DOLLARS = min_swap
        config.MAX_SWAP_SIZE_DOLLARS = max_swap
        config.MIN_TOTAL_SWAP_LOSS_DOLLARS = min_loss
        config.MAX_TOTAL_SWAP_LOSS_DOLLARS = max_loss
        config.MIN_RECOVERY_PERIOD_MONTHS = min_recovery
        config.MAX_RECOVERY_PERIOD_MONTHS = max_recovery
        config.TARGET_BUY_BACK_YIELD = target_yield / 100.0
        config.SOLD_WAVG_PROJ_YIELD_MIN = min_sold_yield / 100.0
        config.SOLD_WAVG_PROJ_YIELD_MAX = max_sold_yield / 100.0
        config.MAX_INDIVIDUAL_BOND_YIELD = max_individual_yield / 100.0
        config.ENFORCE_MIN_SWAP_LOSS = 'enforce' in (enforce_min_loss or [])

        # Show progress
        progress = html.Div([
            dbc.Progress(value=25, className="mb-2"),
            html.P("Filtering eligible bonds...", style={'color': '#e2e8f0'})
        ])

        # Filter bonds
        bond_candidates_df = pre_filter_bonds(portfolio_df)

        if bond_candidates_df.empty:
            status = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "No eligible bonds after filtering"
            ], color="warning", className="border-0")
            return None, status, True, {'display': 'none'}

        # Update progress
        progress = html.Div([
            dbc.Progress(value=50, className="mb-2"),
            html.P("Running genetic algorithm optimization...", style={'color': '#e2e8f0'})
        ])

        # Run optimization
        pareto_raw = evolve_nsga(bond_candidates_df)

        if not pareto_raw:
            status = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "No feasible solutions found"
            ], color="danger", className="border-0")
            return None, status, True, {'display': 'none'}

        # Update progress
        progress = html.Div([
            dbc.Progress(value=75, className="mb-2"),
            html.P("Processing and pruning solutions...", style={'color': '#e2e8f0'})
        ])

        # Process results
        results = []
        for i, option in enumerate(pareto_raw):
            try:
                raw_mask = option["mask"].astype(bool)
                pruned_mask, audit = prune_passengers(bond_candidates_df, raw_mask)
                metrics = compute_metrics(bond_candidates_df, pruned_mask)

                # Check feasibility
                if (config.MIN_SWAP_SIZE_DOLLARS <= metrics.market <= config.MAX_SWAP_SIZE_DOLLARS and
                        metrics.loss <= config.MAX_TOTAL_SWAP_LOSS_DOLLARS and
                        metrics.recovery_months >= config.MIN_RECOVERY_PERIOD_MONTHS and
                        metrics.recovery_months <= config.MAX_RECOVERY_PERIOD_MONTHS and
                        metrics.sold_wavg >= config.SOLD_WAVG_PROJ_YIELD_MIN and
                        metrics.sold_wavg <= config.SOLD_WAVG_PROJ_YIELD_MAX and
                        metrics.delta_income > 0):

                    if config.ENFORCE_MIN_SWAP_LOSS and metrics.loss < config.MIN_TOTAL_SWAP_LOSS_DOLLARS:
                        continue

                    # Get selected bonds
                    selected_bonds = bond_candidates_df.loc[pruned_mask].copy()

                    results.append({
                        "option_id": f"OPT_{len(results) + 1:03d}",
                        "metrics": {
                            "par": float(metrics.par),
                            "book": float(metrics.book),
                            "market": float(metrics.market),
                            "loss": float(metrics.loss),
                            "delta_income": float(metrics.delta_income),
                            "sold_wavg": float(metrics.sold_wavg),
                            "recovery_months": float(metrics.recovery_months),
                            "count": int(metrics.count)
                        },
                        "selected_bonds": selected_bonds.to_json(date_format='iso', orient='split')
                    })
            except Exception as e:
                print(f"Error processing option {i}: {e}")
                continue

        if not results:
            status = dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "No feasible solutions after processing"
            ], color="warning", className="border-0")
            return None, status, True, {'display': 'none'}

        # Sort by delta income
        results.sort(key=lambda x: (-x["metrics"]["delta_income"], x["metrics"]["recovery_months"]))

        # Final progress
        progress = html.Div([
            dbc.Progress(value=100, className="mb-2"),
            html.P("Optimization complete!", style={'color': '#e2e8f0'})
        ])

        status = dbc.Alert([
            html.I(className="bi bi-check-circle-fill me-2"),
            html.Strong("Optimization Complete! "),
            f"Found {len(results)} unique solutions"
        ], color="success", className="border-0")

        return results, status, True, {'display': 'block'}

    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()  # Print full traceback to console for debugging

        status = dbc.Alert([
            html.I(className="bi bi-exclamation-triangle-fill me-2"),
            html.Strong("Error! "),
            f"Optimization failed: {error_msg}"
        ], color="danger", className="border-0")
        return None, status, True, {'display': 'none'}


@app.callback(
    [Output('results-summary', 'children'),
     Output('results-badge', 'children')],
    Input('results-store', 'data')
)
def update_results_summary(results):
    """Update results summary metrics"""
    if not results:
        return "", ""

    best_income = max(r["metrics"]["delta_income"] for r in results)
    avg_recovery = np.mean([r["metrics"]["recovery_months"] for r in results])
    avg_size = np.mean([r["metrics"]["market"] for r in results])

    summary = dbc.Row([
        dbc.Col([
            html.Div([
                html.H4(f"{len(results)}", className="metric-value", style={'color': '#3b82f6'}),
                html.P("Total Options", className="metric-label")
            ], className="metric-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.H4(format_currency(best_income), className="metric-value", style={'color': '#10b981'}),
                html.P("Best Δ-Income", className="metric-label")
            ], className="metric-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.H4(f"{avg_recovery:.1f} mo", className="metric-value", style={'color': '#60a5fa'}),
                html.P("Avg Recovery", className="metric-label")
            ], className="metric-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.H4(format_currency(avg_size), className="metric-value", style={'color': '#f59e0b'}),
                html.P("Avg Size", className="metric-label")
            ], className="metric-card")
        ], md=3)
    ])

    badge = f"{len(results)} solutions found"

    return summary, badge


@app.callback(
    Output('results-content', 'children'),
    [Input('results-tabs', 'active_tab'),
     Input('results-store', 'data')]
)
def update_results_content(active_tab, results):
    """Update results content based on active tab"""
    if not results:
        return ""

    if active_tab == "all-options":
        return create_options_table(results)
    elif active_tab == "option-details":
        return create_option_details(results)
    elif active_tab == "charts":
        return create_analysis_charts(results)
    elif active_tab == "downloads":
        return create_download_section(results)

    return ""


def create_options_table(results):
    """Create table of all options"""
    if not results:
        return ""

    # Prepare data for table
    table_data = []
    for i, result in enumerate(results):
        m = result["metrics"]
        table_data.append({
            "Rank": i + 1,
            "Option ID": result["option_id"],
            "Δ-Income": format_currency(m["delta_income"]),
            "Loss": format_currency(m["loss"]),
            "Recovery": f"{m['recovery_months']:.1f} months",
            "Size": format_currency(m["market"]),
            "Sold Yield": format_percent(m["sold_wavg"]),
            "Bonds": m["count"]
        })

    return dash_table.DataTable(
        data=table_data,
        columns=[{"name": col, "id": col} for col in table_data[0].keys()],
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'backgroundColor': '#1a1f2e',
            'color': '#e2e8f0',
            'border': '1px solid #374151'
        },
        style_header={
            'backgroundColor': '#2d3748',
            'fontWeight': 'bold',
            'color': '#ffffff',
            'border': '1px solid #4a5568'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 0},
                'backgroundColor': '#065f46',
                'color': '#ffffff',
            }
        ],
        page_size=15,
        sort_action="native",
        filter_action="native"
    )


def create_option_details(results):
    """Create detailed view of selected option"""
    if not results:
        return ""

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Select Option for Details:", className="form-label", style={'color': '#e2e8f0'}),
                dcc.Dropdown(
                    id='option-selector',
                    options=[{'label': r["option_id"], 'value': i} for i, r in enumerate(results)],
                    value=0,
                    className="mb-3"
                )
            ], md=6)
        ]),
        html.Div(id='option-detail-content')
    ])


def create_analysis_charts(results):
    """Create analysis charts"""
    if not results:
        return ""

    # Prepare data for charts
    df_results = pd.DataFrame([
        {
            "Option": r["option_id"],
            "Delta Income": r["metrics"]["delta_income"],
            "Recovery (months)": r["metrics"]["recovery_months"],
            "Loss": r["metrics"]["loss"],
            "Size": r["metrics"]["market"],
            "Bonds": r["metrics"]["count"]
        }
        for r in results
    ])

    # Risk-Return scatter plot
    fig1 = px.scatter(
        df_results,
        x="Recovery (months)",
        y="Delta Income",
        size="Size",
        color="Loss",
        hover_data=["Option", "Bonds"],
        title="Risk-Return Profile of All Options",
        color_continuous_scale="RdYlBu_r"
    )
    fig1.update_layout(
        height=400,
        plot_bgcolor='#1a1f2e',
        paper_bgcolor='#1a1f2e',
        font_color='#e2e8f0',
        title_font_color='#ffffff'
    )

    # Income distribution
    fig2 = px.histogram(
        df_results,
        x="Delta Income",
        nbins=20,
        title="Distribution of Income Enhancement",
        labels={"count": "Number of Options"}
    )
    fig2.update_layout(
        height=300,
        plot_bgcolor='#1a1f2e',
        paper_bgcolor='#1a1f2e',
        font_color='#e2e8f0',
        title_font_color='#ffffff'
    )

    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2)
    ])


def create_download_section(results):
    """Create download section"""
    if not results:
        return ""

    # Prepare CSV data
    csv_data = []
    for i, result in enumerate(results):
        m = result["metrics"]
        csv_data.append({
            "Rank": i + 1,
            "Option_ID": result["option_id"],
            "Delta_Income": m["delta_income"],
            "Loss": m["loss"],
            "Recovery_Months": m["recovery_months"],
            "Market_Value": m["market"],
            "Sold_Yield_Pct": m["sold_wavg"] * 100,
            "Bond_Count": m["count"]
        })

    df_download = pd.DataFrame(csv_data)
    csv_string = df_download.to_csv(index=False, encoding='utf-8')
    csv_base64 = base64.b64encode(csv_string.encode()).decode()

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("Download Results", style={'color': '#ffffff'}),
                html.P("Export your optimization results for further analysis or client presentation.",
                       style={'color': '#9ca3af'}),

                html.A(
                    dbc.Button([
                        html.I(className="bi bi-download me-2"),
                        "Download CSV Report"
                    ], color="primary", size="lg"),
                    href=f"data:text/csv;base64,{csv_base64}",
                    download=f"bond_swap_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
            ])
        ])
    ])


@app.callback(
    Output('option-detail-content', 'children'),
    [Input('option-selector', 'value')],
    State('results-store', 'data')
)
def update_option_details(selected_option_idx, results):
    """Update detailed option view"""
    if selected_option_idx is None or not results:
        return ""

    option = results[selected_option_idx]
    m = option["metrics"]

    # Load selected bonds
    try:
        selected_bonds = pd.read_json(option["selected_bonds"], orient='split')
    except:
        selected_bonds = pd.DataFrame()

    # Metrics cards
    metrics_cards = dbc.Row([
        dbc.Col([
            html.Div([
                html.H4(format_currency(m["delta_income"]), className="metric-value", style={'color': '#10b981'}),
                html.P("Income Enhancement", className="metric-label")
            ], className="metric-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.H4(format_currency(m["loss"]), className="metric-value", style={'color': '#ef4444'}),
                html.P("Total Loss", className="metric-label")
            ], className="metric-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.H4(f"{m['recovery_months']:.1f} mo", className="metric-value", style={'color': '#60a5fa'}),
                html.P("Recovery Period", className="metric-label")
            ], className="metric-card")
        ], md=3),
        dbc.Col([
            html.Div([
                html.H4(f"{m['count']}", className="metric-value", style={'color': '#3b82f6'}),
                html.P("Bonds Selected", className="metric-label")
            ], className="metric-card")
        ], md=3)
    ], className="mb-4")

    # Selected bonds table
    bonds_table = ""
    if not selected_bonds.empty:
        display_bonds = selected_bonds[["CUSIP", "par", "book", "market", "loss", "delta_income"]].head(10)

        # Format for display
        for col in ["par", "book", "market", "loss", "delta_income"]:
            if col in display_bonds.columns:
                display_bonds[col] = display_bonds[col].apply(lambda x: format_currency(x))

        bonds_table = html.Div([
            html.H5("Selected Bonds (Top 10)", className="mt-4 mb-3", style={'color': '#ffffff'}),
            dash_table.DataTable(
                data=display_bonds.to_dict('records'),
                columns=[{"name": col.replace('_', ' ').title(), "id": col} for col in display_bonds.columns],
                style_cell={
                    'textAlign': 'left',
                    'padding': '8px',
                    'backgroundColor': '#1a1f2e',
                    'color': '#e2e8f0',
                    'border': '1px solid #374151'
                },
                style_header={
                    'backgroundColor': '#2d3748',
                    'fontWeight': 'bold',
                    'color': '#ffffff',
                    'border': '1px solid #4a5568'
                },
                page_size=10
            )
        ])

    return html.Div([
        html.H4(f"Option Details: {option['option_id']}", className="mb-4", style={'color': '#ffffff'}),
        metrics_cards,
        bonds_table
    ])


if __name__ == '__main__':
    app.run(debug=True, port=8050)