import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import json
import tempfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Import your existing modules
from src.config import SCENARIO_PRESETS, switch_scenario, print_scenario_summary
from src.data_handler import load_and_prepare_data, pre_filter_bonds
from src.genetic_algorithm import evolve_nsga
from src.pruning import prune_passengers, compute_metrics
from src.reporting import generate_report
from src.tr_engine import compute_simple_tr_analysis

# Page configuration
st.set_page_config(
    page_title="Bond Swap Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .constraint-check {
        display: flex;
        align-items: center;
        margin: 0.2rem 0;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'scenario_config' not in st.session_state:
        st.session_state.scenario_config = None
    if 'optimization_running' not in st.session_state:
        st.session_state.optimization_running = False


def format_money(value):
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


def check_constraint(condition, label):
    """Display constraint check with checkmark"""
    icon = "✅" if condition else "❌"
    return f"{icon} {label}"


def create_scenario_config_ui():
    """Create UI for scenario configuration"""
    st.sidebar.header("📊 Scenario Configuration")

    # Scenario selection
    scenario_options = list(SCENARIO_PRESETS.keys())
    selected_scenario = st.sidebar.selectbox(
        "Select Scenario Type",
        scenario_options,
        help="Choose a preset scenario or custom configuration"
    )

    # Load scenario preset
    scenario_config = SCENARIO_PRESETS[selected_scenario].copy()

    # Show scenario description
    st.sidebar.info(f"**{selected_scenario.title()} Scenario**\n\n{scenario_config['description']}")

    # Custom parameter adjustment
    if selected_scenario == "custom" or st.sidebar.checkbox("🔧 Customize Parameters", value=False):
        st.sidebar.subheader("Custom Parameters")

        # Swap size constraints
        col1, col2 = st.sidebar.columns(2)
        with col1:
            scenario_config["min_swap_size"] = st.number_input(
                "Min Swap Size ($)",
                value=scenario_config["min_swap_size"],
                step=500_000,
                format="%d"
            )
        with col2:
            scenario_config["max_swap_size"] = st.number_input(
                "Max Swap Size ($)",
                value=scenario_config["max_swap_size"],
                step=1_000_000,
                format="%d"
            )

        # Loss constraints
        col1, col2 = st.sidebar.columns(2)
        with col1:
            scenario_config["min_loss"] = st.number_input(
                "Min Loss ($)",
                value=scenario_config["min_loss"],
                step=50_000,
                format="%d"
            )
        with col2:
            scenario_config["max_loss"] = st.number_input(
                "Max Loss ($)",
                value=scenario_config["max_loss"],
                step=100_000,
                format="%d"
            )

        # Recovery period
        col1, col2 = st.sidebar.columns(2)
        with col1:
            scenario_config["min_recovery_months"] = st.number_input(
                "Min Recovery (months)",
                value=scenario_config["min_recovery_months"],
                step=0.5,
                format="%.1f"
            )
        with col2:
            scenario_config["max_recovery_months"] = st.number_input(
                "Max Recovery (months)",
                value=scenario_config["max_recovery_months"],
                step=1.0,
                format="%.1f"
            )

        # Yield constraints
        col1, col2 = st.sidebar.columns(2)
        with col1:
            scenario_config["target_buy_yield"] = st.number_input(
                "Target Buy Yield (%)",
                value=scenario_config["target_buy_yield"],
                step=0.1,
                format="%.1f"
            )
        with col2:
            scenario_config["max_sold_wavg_yield"] = st.number_input(
                "Max Sold Avg Yield (%)",
                value=scenario_config["max_sold_wavg_yield"],
                step=0.1,
                format="%.1f"
            )

        # Individual bond yield limit
        scenario_config["max_individual_bond_yield"] = st.sidebar.number_input(
            "Max Individual Bond Yield (%)",
            value=scenario_config.get("max_individual_bond_yield", 6.0),
            step=0.1,
            format="%.1f"
        )

        # Minimum loss enforcement
        scenario_config["enforce_min_loss"] = st.sidebar.checkbox(
            "Enforce Minimum Loss (for tax swaps)",
            value=scenario_config["enforce_min_loss"]
        )

    return selected_scenario, scenario_config


def load_portfolio_file(uploaded_file):
    """Load and validate portfolio file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet' if uploaded_file.name.endswith(
                '.parquet') else '.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load data using existing function
        portfolio_df = load_and_prepare_data(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        return portfolio_df
    except Exception as e:
        st.error(f"Error loading portfolio file: {str(e)}")
        return None


def run_optimization(portfolio_df, scenario_config):
    """Run the bond swap optimization"""
    try:
        # Update configuration
        import src.config as config
        for key, value in scenario_config.items():
            if key == "min_swap_size":
                config.MIN_SWAP_SIZE_DOLLARS = value
            elif key == "max_swap_size":
                config.MAX_SWAP_SIZE_DOLLARS = value
            elif key == "min_loss":
                config.MIN_TOTAL_SWAP_LOSS_DOLLARS = value
            elif key == "max_loss":
                config.MAX_TOTAL_SWAP_LOSS_DOLLARS = value
            elif key == "min_recovery_months":
                config.MIN_RECOVERY_PERIOD_MONTHS = value
            elif key == "max_recovery_months":
                config.MAX_RECOVERY_PERIOD_MONTHS = value
            elif key == "target_buy_yield":
                config.TARGET_BUY_BACK_YIELD = value / 100.0
            elif key == "max_sold_wavg_yield":
                config.SOLD_WAVG_PROJ_YIELD_MAX = value / 100.0
            elif key == "max_individual_bond_yield":
                config.MAX_INDIVIDUAL_BOND_YIELD = value / 100.0
            elif key == "enforce_min_loss":
                config.ENFORCE_MIN_SWAP_LOSS = value

        # Filter eligible bonds
        bond_candidates_df = pre_filter_bonds(portfolio_df)

        if bond_candidates_df.empty:
            return None, "No eligible bonds after filtering"

        # Run genetic algorithm optimization
        pareto_raw = evolve_nsga(bond_candidates_df)

        if not pareto_raw:
            return None, "No feasible solutions found"

        # Process all solutions
        processed_options = []
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
                        metrics.sold_wavg <= config.SOLD_WAVG_PROJ_YIELD_MAX and
                        metrics.delta_income > 0):

                    if config.ENFORCE_MIN_SWAP_LOSS and metrics.loss < config.MIN_TOTAL_SWAP_LOSS_DOLLARS:
                        continue

                    processed_options.append({
                        "option_id": f"OPT_{i + 1:03d}",
                        "mask": pruned_mask,
                        "metrics": metrics,
                        "bond_candidates": bond_candidates_df,
                        "selected_bonds": bond_candidates_df.loc[pruned_mask].copy()
                    })
            except Exception:
                continue

        # Sort by delta income
        processed_options.sort(key=lambda x: (-x["metrics"].delta_income, x["metrics"].recovery_months))

        return processed_options, f"Found {len(processed_options)} unique solutions"

    except Exception as e:
        return None, f"Optimization error: {str(e)}"


def display_results(results):
    """Display optimization results"""
    if not results:
        st.warning("No results to display")
        return

    # Summary metrics
    st.subheader("📈 Optimization Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Options", len(results))
    with col2:
        best_income = max(r["metrics"].delta_income for r in results)
        st.metric("Best Δ-Income", format_money(best_income))
    with col3:
        avg_recovery = np.mean([r["metrics"].recovery_months for r in results])
        st.metric("Avg Recovery", f"{avg_recovery:.1f} months")
    with col4:
        avg_size = np.mean([r["metrics"].market for r in results])
        st.metric("Avg Swap Size", format_money(avg_size))

    # Results table
    st.subheader("🏆 All Swap Options")

    # Prepare data for display
    display_data = []
    for i, result in enumerate(results):
        m = result["metrics"]
        display_data.append({
            "Rank": i + 1,
            "Option ID": result["option_id"],
            "Δ-Income": m.delta_income,
            "Loss": m.loss,
            "Recovery (months)": round(m.recovery_months, 2),
            "Swap Size": m.market,
            "Sold Yield": m.sold_wavg * 100,
            "Bonds": m.count
        })

    results_df = pd.DataFrame(display_data)

    # Format currency columns
    results_df["Δ-Income_fmt"] = results_df["Δ-Income"].apply(format_money)
    results_df["Loss_fmt"] = results_df["Loss"].apply(format_money)
    results_df["Swap Size_fmt"] = results_df["Swap Size"].apply(format_money)
    results_df["Sold Yield_fmt"] = results_df["Sold Yield"].apply(lambda x: f"{x:.2f}%")

    # Display formatted table
    display_columns = ["Rank", "Option ID", "Δ-Income_fmt", "Loss_fmt", "Recovery (months)",
                       "Swap Size_fmt", "Sold Yield_fmt", "Bonds"]

    column_config = {
        "Δ-Income_fmt": "Δ-Income",
        "Loss_fmt": "Loss",
        "Swap Size_fmt": "Swap Size",
        "Sold Yield_fmt": "Sold Yield (%)"
    }

    st.dataframe(
        results_df[display_columns],
        column_config=column_config,
        hide_index=True,
        use_container_width=True
    )

    # Option selection for detailed view
    st.subheader("🔍 Detailed Analysis")

    selected_option_id = st.selectbox(
        "Select option for detailed analysis:",
        [r["option_id"] for r in results],
        index=0
    )

    # Find selected option
    selected_option = next(r for r in results if r["option_id"] == selected_option_id)

    # Display detailed metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💰 Financial Metrics**")
        m = selected_option["metrics"]

        # Check constraints
        constraints = []
        constraints.append(check_constraint(
            st.session_state.scenario_config["min_swap_size"] <= m.market <= st.session_state.scenario_config[
                "max_swap_size"],
            f"Swap Size: {format_money(m.market)}"
        ))
        constraints.append(check_constraint(
            m.loss <= st.session_state.scenario_config["max_loss"],
            f"Loss: {format_money(m.loss)}"
        ))
        constraints.append(check_constraint(
            st.session_state.scenario_config["min_recovery_months"] <= m.recovery_months <=
            st.session_state.scenario_config["max_recovery_months"],
            f"Recovery: {m.recovery_months:.2f} months"
        ))
        constraints.append(check_constraint(
            m.sold_wavg <= st.session_state.scenario_config["max_sold_wavg_yield"] / 100.0,
            f"Sold Yield: {format_percent(m.sold_wavg)}"
        ))

        for constraint in constraints:
            st.markdown(constraint)

        st.markdown(f"**Par Total:** {format_money(m.par)}")
        st.markdown(f"**Book Total:** {format_money(m.book)}")
        st.markdown(f"**Market Total:** {format_money(m.market)}")
        st.markdown(f"**Net G/L:** {format_money(m.market - m.book)}")

    with col2:
        st.markdown("**📊 Performance Metrics**")
        st.markdown(f"**Income Enhancement:** {format_money(m.delta_income)}")
        st.markdown(f"**Recovery Period:** {m.recovery_months:.2f} months")
        st.markdown(f"**Current Income:** {format_money(m.income)}")
        st.markdown(f"**Enhanced Income:** {format_money(m.income + m.delta_income)}")
        st.markdown(
            f"**Target Buy Yield:** {format_percent(st.session_state.scenario_config['target_buy_yield'] / 100.0)}")
        st.markdown(
            f"**Yield Pickup:** {format_percent(st.session_state.scenario_config['target_buy_yield'] / 100.0 - m.sold_wavg)}")
        st.markdown(f"**Bond Count:** {m.count}")

    # Selected bonds table
    st.subheader("📋 Selected Bonds")
    selected_bonds = selected_option["selected_bonds"]

    if not selected_bonds.empty:
        # Format bonds table
        bonds_display = selected_bonds[["CUSIP", "par", "book", "market", "loss", "delta_income"]].copy()

        # Format currency columns
        for col in ["par", "book", "market", "loss", "delta_income"]:
            bonds_display[f"{col}_fmt"] = bonds_display[col].apply(format_money)

        # Add description if available
        display_cols = ["CUSIP"]
        if "Description" in selected_bonds.columns:
            display_cols.append("Description")
        display_cols.extend(["par_fmt", "book_fmt", "market_fmt", "loss_fmt", "delta_income_fmt"])

        column_config = {
            "par_fmt": "Par",
            "book_fmt": "Book",
            "market_fmt": "Market",
            "loss_fmt": "Loss",
            "delta_income_fmt": "Δ-Income"
        }

        st.dataframe(
            bonds_display[display_cols],
            column_config=column_config,
            hide_index=True,
            use_container_width=True
        )

    # Charts
    st.subheader("📈 Analysis Charts")

    # Scatter plot of all options
    fig = px.scatter(
        results_df,
        x="Recovery (months)",
        y="Δ-Income",
        size="Swap Size",
        color="Loss",
        hover_data=["Option ID", "Bonds"],
        title="Risk-Return Profile of All Options",
        labels={
            "Recovery (months)": "Recovery Period (months)",
            "Δ-Income": "Income Enhancement ($)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download options
    st.subheader("💾 Download Results")

    # CSV download
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📄 Download Results CSV",
        data=csv_buffer.getvalue(),
        file_name=f"bond_swap_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # JSON download for selected option
    selected_data = {
        "option_id": selected_option["option_id"],
        "metrics": {
            "par": float(m.par),
            "book": float(m.book),
            "market": float(m.market),
            "loss": float(m.loss),
            "delta_income": float(m.delta_income),
            "sold_wavg": float(m.sold_wavg),
            "recovery_months": float(m.recovery_months),
            "count": int(m.count)
        },
        "selected_cusips": selected_bonds["CUSIP"].tolist(),
        "scenario_config": st.session_state.scenario_config
    }

    st.download_button(
        label=f"📋 Download {selected_option_id} Details",
        data=json.dumps(selected_data, indent=2),
        file_name=f"{selected_option_id}_details.json",
        mime="application/json"
    )


def main():
    """Main Streamlit application"""
    initialize_session_state()

    # Title
    st.markdown('<h1 class="main-header">Bond Swap Optimizer</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    scenario_name, scenario_config = create_scenario_config_ui()
    st.session_state.scenario_config = scenario_config

    # Main content area
    st.header("📁 Portfolio Upload")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your bond portfolio file",
        type=['parquet', 'xlsx', 'xls'],
        help="Upload a parquet file or Excel file containing your bond portfolio data"
    )

    if uploaded_file is not None:
        # Load portfolio data
        with st.spinner("Loading portfolio data..."):
            portfolio_df = load_portfolio_file(uploaded_file)
            st.session_state.portfolio_data = portfolio_df

        if portfolio_df is not None:
            # Display portfolio summary
            st.success(f"✅ Portfolio loaded successfully!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Bonds", len(portfolio_df))
            with col2:
                st.metric("Total Market Value", format_money(portfolio_df['market'].sum()))
            with col3:
                st.metric("Total Par Value", format_money(portfolio_df['par'].sum()))

            # Show scenario summary
            st.header("⚙️ Current Scenario Configuration")
            config_col1, config_col2 = st.columns(2)

            with config_col1:
                st.markdown(f"**Scenario:** {scenario_name.title()}")
                st.markdown(
                    f"**Swap Size:** {format_money(scenario_config['min_swap_size'])} - {format_money(scenario_config['max_swap_size'])}")
                st.markdown(
                    f"**Loss Range:** {format_money(scenario_config['min_loss'])} - {format_money(scenario_config['max_loss'])}")
                st.markdown(
                    f"**Recovery Period:** {scenario_config['min_recovery_months']:.1f} - {scenario_config['max_recovery_months']:.1f} months")

            with config_col2:
                st.markdown(f"**Target Buy Yield:** {scenario_config['target_buy_yield']:.2f}%")
                st.markdown(f"**Max Sold Avg Yield:** {scenario_config['max_sold_wavg_yield']:.2f}%")
                st.markdown(f"**Max Individual Yield:** {scenario_config.get('max_individual_bond_yield', 6.0):.2f}%")
                st.markdown(f"**Enforce Min Loss:** {'Yes' if scenario_config['enforce_min_loss'] else 'No'}")

            # Run optimization button
            st.header("🚀 Run Optimization")

            if st.button("🔍 Find Optimal Swaps", type="primary", use_container_width=True):
                st.session_state.optimization_running = True

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Run optimization
                status_text.text("🔄 Filtering eligible bonds...")
                progress_bar.progress(20)

                status_text.text("🧬 Running genetic algorithm optimization...")
                progress_bar.progress(40)

                results, message = run_optimization(portfolio_df, scenario_config)

                progress_bar.progress(80)
                status_text.text("📊 Processing results...")

                progress_bar.progress(100)
                status_text.text("✅ Optimization complete!")

                st.session_state.optimization_results = results
                st.session_state.optimization_running = False

                if results:
                    st.success(f"🎉 {message}")
                else:
                    st.error(f"❌ {message}")

    # Display results if available
    if st.session_state.optimization_results:
        st.header("📊 Results")
        display_results(st.session_state.optimization_results)

    elif st.session_state.portfolio_data is None:
        st.info("👆 Please upload a portfolio file to begin optimization")


if __name__ == "__main__":
    main()