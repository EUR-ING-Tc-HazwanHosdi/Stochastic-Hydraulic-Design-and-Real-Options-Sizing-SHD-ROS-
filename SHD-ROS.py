import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="SHD-ROS Municipal Sewerage Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0E4C92; margin-bottom: 0.1rem; }
    .sub-header { font-size: 1.1rem; font-weight: 400; color: #555555; margin-bottom: 2rem; }
    .benchmark-box { background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">SHD-ROS Municipal Sewerage Simulator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Stochastic Hydraulic Design & Real Options Sizing Framework | Municipal & Industrial Infrastructure Planning</div>',
    unsafe_allow_html=True,
)

# --- Sidebar Parameter Controls ---
st.sidebar.header("🛠️ Simulation Parameters")

with st.sidebar.expander("1. Inflow & Jump-Diffusion Settings", expanded=True):
    Q0 = st.number_input("Initial Flow Q₀ (MLD)", value=20.0, step=1.0, format="%.1f")
    years = st.slider("Planning Horizon (Years)", min_value=10, max_value=40, value=30, step=5)
    mu_Q = st.slider("Deterministic Growth Rate (μ_Q)", min_value=0.0, max_value=0.08, value=0.03, step=0.005)
    sigma_Q = st.slider("Inflow Volatility (σ_Q)", min_value=0.01, max_value=0.20, value=0.08, step=0.01)
    lambda_jump = st.slider("Jump Intensity (λ - Shocks/Year)", min_value=0.0, max_value=0.5, value=0.10, step=0.05)
    jump_mean = st.slider("Mean Shock Magnitude (J_Q)", min_value=0.05, max_value=0.50, value=0.20, step=0.05)

with st.sidebar.expander("2. Economic & Cost Parameters", False):
    r = st.slider("Discount Rate (r)", min_value=0.02, max_value=0.12, value=0.06, step=0.005)
    capex_rigid = st.number_input("Rigid Plant CapEx (RM Million)", value=45.0, step=2.0)
    capex_modular = st.number_input("Modular Train CapEx (RM Million)", value=15.0, step=1.0)
    unit_opex = st.number_input("Base Energy OpEx (RM Million/MLD/year)", value=0.35, step=0.05)
    penalty_cost = st.number_input("Non-Compliance Penalty (RM Million/year overflow)", value=5.0, step=0.5)

with st.sidebar.expander("3. Real Options & Triggers", False):
    q_crit_ratio = st.slider("Critical Expansion Trigger Q_crit (% of Capacity)", min_value=0.70, max_value=0.95, value=0.85, step=0.05)
    buffer_window = st.slider("Buffer Window τ_buffer (Years)", min_value=1, max_value=5, value=2, step=1)
    land_option_cost = st.number_input("Land Option Pre-purchase Cost (RM Million)", value=1.5, step=0.2)

num_simulations = st.sidebar.slider("Monte Carlo Scenarios (seed = 42)", min_value=100, max_value=2000, value=2000, step=100)


# --- Monte Carlo Simulation Engine ---
@st.cache_data
def run_monte_carlo(Q0, years, mu_Q, sigma_Q, lambda_jump, jump_mean, num_simulations):
    rng = np.random.default_rng(42)
    dt = 1.0
    time_steps = years + 1

    flows = np.zeros((num_simulations, time_steps))
    flows[:, 0] = Q0

    for t in range(1, time_steps):
        z = rng.standard_normal(num_simulations)
        poisson_shocks = rng.poisson(lambda_jump * dt, num_simulations)
        jump_sizes = rng.exponential(jump_mean, num_simulations) * poisson_shocks

        d_flow = flows[:, t - 1] * (mu_Q * dt + sigma_Q * np.sqrt(dt) * z + jump_sizes)
        flows[:, t] = np.maximum(flows[:, t - 1] + d_flow, Q0)

    return flows


# --- Policy Evaluation Function ---
def evaluate_strategies(
    flow_paths,
    capex_rigid,
    capex_modular,
    unit_opex,
    penalty_cost,
    r,
    q_crit_ratio,
    buffer_window,
    land_option_cost,
):
    num_sims, t_steps = flow_paths.shape

    rigid_npvs = np.zeros(num_sims)
    shd_ros_npvs = np.zeros(num_sims)
    expansions_per_path = np.zeros(num_sims)
    penalties_per_path = np.zeros(num_sims)

    path0_cap = np.zeros(t_steps)
    path0_trig = np.zeros(t_steps)

    disc_factors = (1.0 + r) ** np.arange(1, t_steps)

    for i in range(num_sims):
        path = flow_paths[i]

        # 1. Traditional Rigid Strategy
        rigid_npvs[i] = capex_rigid + np.sum((path[1:] * unit_opex) / disc_factors)

        # 2. SHD-ROS Flexible Strategy
        initial_capacity = np.percentile(path, 50) * 0.7
        cost_flexible = capex_modular + land_option_cost
        current_cap = initial_capacity
        expansion_locked_timer = 0
        exp_count = 0
        pen_count = 0

        if i == 0:
            path0_cap[0] = current_cap
            path0_trig[0] = q_crit_ratio * current_cap

        for t in range(1, t_steps):
            op = path[t] * unit_opex
            penalty = 0.0

            if path[t] >= q_crit_ratio * current_cap and expansion_locked_timer == 0:
                cost_flexible += capex_modular / ((1.0 + r) ** t)
                current_cap += initial_capacity
                exp_count += 1
                expansion_locked_timer = buffer_window

            if expansion_locked_timer > 0:
                expansion_locked_timer -= 1
                if path[t] > current_cap:
                    penalty = penalty_cost
                    pen_count += 1

            cost_flexible += (op + penalty) / ((1.0 + r) ** t)

            if i == 0:
                path0_cap[t] = current_cap
                path0_trig[t] = q_crit_ratio * current_cap

        shd_ros_npvs[i] = cost_flexible
        expansions_per_path[i] = exp_count
        penalties_per_path[i] = pen_count

    return {
        "rigid_npvs": rigid_npvs,
        "shd_ros_npvs": shd_ros_npvs,
        "expansions_per_path": expansions_per_path,
        "penalties_per_path": penalties_per_path,
        "path0_cap": path0_cap,
        "path0_trig": path0_trig,
    }


flow_paths = run_monte_carlo(Q0, years, mu_Q, sigma_Q, lambda_jump, jump_mean, num_simulations)
time_axis = np.arange(years + 1)

sim_out = evaluate_strategies(
    flow_paths=flow_paths,
    capex_rigid=capex_rigid,
    capex_modular=capex_modular,
    unit_opex=unit_opex,
    penalty_cost=penalty_cost,
    r=r,
    q_crit_ratio=q_crit_ratio,
    buffer_window=buffer_window,
    land_option_cost=land_option_cost,
)

rigid_npvs = sim_out["rigid_npvs"]
shd_ros_npvs = sim_out["shd_ros_npvs"]
vof_series = rigid_npvs - shd_ros_npvs

# --- Dashboard Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Inflow & Jump Simulation",
    "💰 ENPV & Cost Risk Analysis",
    "🏗️ Real Options Triggers",
    "📋 Executive Summary Report",
    "📖 Parameter Legend & Guide",
])

with tab1:
    st.subheader("Stochastic Inflow Trajectories (Jump-Diffusion Model)")
    st.markdown(
        "Simulating municipal sewage inflow ($Q_t$) accounting for baseline urban growth ($\\mu_Q$), "
        "continuous volatility ($\\sigma_Q$), and discrete development surges ($\\lambda, J_Q$)."
    )

    fig_flow = go.Figure()
    for i in range(min(50, num_simulations)):
        fig_flow.add_trace(
            go.Scatter(
                x=time_axis,
                y=flow_paths[i],
                mode="lines",
                line=dict(color="rgba(14, 76, 146, 0.16)", width=1),
                showlegend=False,
            )
        )

    mean_path = np.mean(flow_paths, axis=0)
    fig_flow.add_trace(
        go.Scatter(
            x=time_axis,
            y=mean_path,
            mode="lines",
            line=dict(color="#082A52", width=3),
            name=f"Ensemble Mean Flow ({num_simulations:,} runs)",
        )
    )

    fig_flow.update_layout(
        xaxis_title="Planning Year (t)",
        yaxis_title="Influent Flow (MLD)",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig_flow, use_container_width=True)

with tab2:
    st.subheader("Expected Net Present Value (ENPV) Risk Distribution")
    st.markdown(
        "Comparing the **Traditional Rigid Strategy** (front-loaded monolithic CapEx) against "
        "the **SHD-ROS Flexible Strategy** (land option pre-purchase + staged modular expansion)."
    )

    mean_rig = np.mean(rigid_npvs)
    mean_flx = np.mean(shd_ros_npvs)
    mean_vof = np.mean(vof_series)
    pct_savings = (mean_vof / mean_rig * 100) if mean_rig != 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Rigid ENPV", f"RM {mean_rig:.2f}M", delta=f"SD: RM {np.std(rigid_npvs):.2f}M", delta_color="off")
    col2.metric("Mean SHD-ROS ENPV", f"RM {mean_flx:.2f}M", delta=f"-{pct_savings:.2f}% Mean Cost", delta_color="inverse")
    col3.metric("Mean Value of Flexibility (VoF)", f"RM {mean_vof:.2f}M", delta=f"P90 VoF: RM {np.percentile(vof_series, 90):.2f}M", delta_color="normal")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=rigid_npvs, nbinsx=35, name="Traditional Rigid Strategy", marker_color="#E63946", opacity=0.65))
    fig_hist.add_trace(go.Histogram(x=shd_ros_npvs, nbinsx=35, name="SHD-ROS Flexible Strategy", marker_color="#2A9D8F", opacity=0.65))
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Lifecycle ENPV (RM Million)",
        yaxis_title="Simulation Frequency",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("Real Options Modular Capacity & Dynamic Trigger Staging (Sample Path 0)")
    st.markdown(
        f"Tracking active installed capacity ($C_t$) and the dynamic expansion trigger threshold "
        f"($Q_{{\\text{{crit}}}} \\cdot C_t = {q_crit_ratio:.0%}$ of active capacity) along Sample Path 0."
    )

    fig_trig = go.Figure()
    fig_trig.add_trace(
        go.Scatter(
            x=time_axis,
            y=sim_out["path0_cap"],
            mode="lines",
            line=dict(color="#2A9D8F", width=2.2, shape="hv"),
            name="Active Installed Capacity (Path 0)",
        )
    )
    fig_trig.add_trace(
        go.Scatter(
            x=time_axis,
            y=sim_out["path0_trig"],
            mode="lines",
            line=dict(color="#E63946", width=2, dash="dash", shape="hv"),
            name=f"Expansion Trigger ({q_crit_ratio:.0%} of Active Cap.)",
        )
    )
    fig_trig.add_trace(
        go.Scatter(
            x=time_axis,
            y=flow_paths[0],
            mode="lines+markers",
            name="Simulated Inflow (Path 0)",
            line=dict(color="#0E4C92", width=2),
            marker=dict(size=5),
        )
    )

    fig_trig.update_layout(
        xaxis_title="Planning Horizon (Years)",
        yaxis_title="Flow / Capacity (MLD)",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_trig, use_container_width=True)

with tab4:
    st.subheader("Executive Decision Support & Policy Evaluation Summary")

    st.markdown(
        """
        <div class="benchmark-box">
        <h4>🇲🇾 Malaysian Urban Development & Sewerage Benchmarks</h4>
        <ul>
            <li><b>MSIG Volumetric Basis:</b> Baseline flow of 20.0 MLD corresponds to approximately 95,238 PE at the MSIG benchmark of 210 L/PE/day.</li>
            <li><b>Growth & Shock Calibration:</b> Baseline drift (3.0%/yr) reflects PLANMalaysia urban growth corridors, while jump shocks (λ = 0.10/yr, mean J_Q = 20%) capture discrete commercial or industrial tie-ins.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upfront_rigid = capex_rigid
    upfront_flex = capex_modular + land_option_cost

    summary_df = pd.DataFrame({
        "Metric": [
            "Upfront CapEx at t = 0",
            "Mean Lifecycle ENPV",
            "ENPV Std. Dev. (SD)",
            "P10 / P90 Lifecycle ENPV",
            "Mean Expansions per Path",
            "Mean Penalty Exposure",
        ],
        "Traditional Rigid Strategy": [
            f"RM {upfront_rigid:.2f}M",
            f"RM {np.mean(rigid_npvs):.2f}M",
            f"RM {np.std(rigid_npvs):.2f}M",
            f"RM {np.percentile(rigid_npvs, 10):.2f}M / RM {np.percentile(rigid_npvs, 90):.2f}M",
            "0 (Upfront build)",
            "0.00 penalty-yrs",
        ],
        "SHD-ROS Flexible Strategy": [
            f"RM {upfront_flex:.2f}M",
            f"RM {np.mean(shd_ros_npvs):.2f}M",
            f"RM {np.std(shd_ros_npvs):.2f}M",
            f"RM {np.percentile(shd_ros_npvs, 10):.2f}M / RM {np.percentile(shd_ros_npvs, 90):.2f}M",
            f"{np.mean(sim_out['expansions_per_path']):.2f} / {years} yrs",
            f"{np.mean(sim_out['penalties_per_path']):.2f} penalty-yrs",
        ],
    })
    st.table(summary_df)

    st.info(
        f"💡 **Key Takeaway**: At default baseline settings (`seed = 42`, `{num_simulations:,}` runs), "
        f"SHD-ROS lowers upfront CapEx from RM {upfront_rigid:.2f}M to RM {upfront_flex:.2f}M "
        f"and yields a mean Value of Flexibility (VoF) of **RM {np.mean(vof_series):.2f}M** "
        f"(P90 VoF: **RM {np.percentile(vof_series, 90):.2f}M**)."
    )

with tab5:
    st.subheader("📖 Sidebar Parameter Legend & Selection Guide")
    st.markdown(
        "Reference guide for the simulation parameters, real options rules, and Malaysian regulatory context "
        "(SPAN MSIG / PLANMalaysia)."
    )

    legend_table_data = {
        "Parameter": [
            "Initial Flow (Q₀)",
            "Planning Horizon (years)",
            "Growth Drift (μ_Q)",
            "Volatility (σ_Q)",
            "Jump Intensity (λ)",
            "Mean Shock Magnitude (J_Q)",
            "Discount Rate (r)",
            "Rigid CapEx (capex_rigid)",
            "Modular CapEx (capex_modular)",
            "Unit OpEx (unit_opex)",
            "Penalty Cost (penalty_cost)",
            "Trigger Ratio (Q_crit)",
            "Buffer Window (τ_buffer)",
            "Land Option Cost (land_option_cost)",
        ],
        "Baseline Default": [
            "20.0 MLD", "30 Years", "3.0% / yr", "8.0% / yr^(1/2)", "0.10 / yr", "20%",
            "6.0% / yr", "RM 45.00M", "RM 15.00M", "RM 0.35M/MLD/yr", "RM 5.00M/yr",
            "85%", "2 Years", "RM 1.50M",
        ],
        "Role in SHD-ROS Policy Evaluation": [
            "Baseline average daily sewage inflow at t = 0 (≈95,238 PE at 210 L/PE/day).",
            "Total simulation horizon in annual steps (Δt = 1 yr).",
            "Deterministic baseline urban catchment growth rate.",
            "Continuous diffusion volatility representing seasonal and wet-weather infiltration variance.",
            "Poisson arrival rate of discrete commercial/industrial development shocks.",
            "Mean proportional jump size when a discrete shock occurs.",
            "Annual discount rate applied to future CapEx, OpEx, and penalties.",
            "Front-loaded capital outlay at t = 0 under the rigid baseline strategy.",
            "Capital cost per modular treatment train added when triggered.",
            "Volumetric operating cost per MLD per year.",
            "Annual penalty incurred if flow exceeds active capacity during a buffer lag window.",
            "Utilization ratio (Q_crit × C_t) that triggers modular expansion when lock timer is 0.",
            "Construction/commissioning lag window (in years) following an expansion trigger.",
            "Upfront land option pre-purchase cost at t = 0 to preserve expansion space.",
        ],
    }

    st.table(pd.DataFrame(legend_table_data))

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #777;'>SHD-ROS Simulation Engine | Streamlit & Python | Municipal Sewerage Planning</p>",
    unsafe_allow_html=True,
)
