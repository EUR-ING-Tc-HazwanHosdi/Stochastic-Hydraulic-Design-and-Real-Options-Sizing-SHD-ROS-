import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="SHD-ROS Industrial Simulation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0E4C92; margin-bottom: 0.1rem; }
    .sub-header { font-size: 1.1rem; font-weight: 400; color: #555555; margin-bottom: 2rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0E4C92; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">SHD-ROS Municipal Sewerage Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Stochastic Hydraulic Design & Real Options Sizing Framework | Developed for Industrial & Municipal Infrastructure Planning</div>', unsafe_allow_html=True)

# --- Sidebar Parameter Controls ---
st.sidebar.header("🛠️ Simulation Parameters")

with st.sidebar.expander("1. Inflow & Jump-Diffusion Settings", expanded=True):
    Q0 = st.number_input("Initial Flow Q₀ (MLD)", value=20.0, step=1.0, format="%.1f")
    years = st.slider("Planning Horizon (Years)", min_value=10, max_value=40, value=30, step=5)
    mu_Q = st.slider("Deterministic Growth Rate (μ_Q)", min_value=0.0, max_value=0.08, value=0.03, step=0.005)
    sigma_Q = st.slider("Inflow Volatility (σ_Q)", min_value=0.01, max_value=0.20, value=0.08, step=0.01)
    lambda_jump = st.slider("Jump Intensity (λ - Shocks/Year)", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
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

num_simulations = st.sidebar.slider("Monte Carlo Scenarios", min_value=100, max_value=2000, value=500, step=100)

# --- Monte Carlo Simulation Engine ---
@st.cache_data
def run_monte_carlo(Q0, years, mu_Q, sigma_Q, lambda_jump, jump_mean, num_simulations):
    np.random.seed(42)
    dt = 1.0 # Annual steps
    time_steps = years + 1
    
    # Storage for flows: [simulations, time_steps]
    flows = np.zeros((num_simulations, time_steps))
    flows[:, 0] = Q0
    
    for t in range(1, time_steps):
        # Geometric Brownian Motion component with Jump Diffusion (Poisson process)
        z = np.random.normal(0, 1, num_simulations)
        poisson_shakes = np.random.poisson(lambda_jump * dt, num_simulations)
        jump_sizes = np.random.exponential(jump_mean, num_simulations) * poisson_shakes
        
        # dQt = mu*Q*dt + sigma*Q*dW + J*Q*dN
        d_flow = flows[:, t-1] * (mu_Q * dt + sigma_Q * np.sqrt(dt) * z + jump_sizes)
        flows[:, t] = np.maximum(flows[:, t-1] + d_flow, Q0)
        
    return flows

flow_paths = run_monte_carlo(Q0, years, mu_Q, sigma_Q, lambda_jump, jump_mean, num_simulations)
time_axis = np.arange(years + 1)

# --- Financial Evaluation Function ---
def evaluate_strategies(flow_paths, capex_rigid, capex_modular, unit_opex, penalty_cost, r, q_crit_ratio, land_option_cost):
    num_sims, t_steps = flow_paths.shape
    
    rigid_npvs = []
    shd_ros_npvs = []
    
    rigid_capacity = np.max(flow_paths) * 1.1 # Oversized to handle peak
    
    for i in range(num_sims):
        path = flow_paths[i]
        
        # --- 1. Rigid Strategy ---
        # Front-loaded CapEx at t=0, low utilization, zero capacity breach
        cost_rigid = capex_rigid
        for t in range(1, t_steps):
            op = path[t] * unit_opex
            cost_rigid += op / ((1 + r) ** t)
        rigid_npvs.append(cost_rigid)
        
        # --- 2. SHD-ROS Flexible Strategy ---
        # Starts small (1 modular train = 35% of max expected flow)
        initial_capacity = np.percentile(path, 50) * 0.7
        cost_flexible = capex_modular + land_option_cost
        current_cap = initial_capacity
        expansion_locked_timer = 0
        
        for t in range(1, t_steps):
            op = path[t] * unit_opex
            penalty = 0
            
            # Check capacity breach
            if path[t] >= q_crit_ratio * current_cap:
                if expansion_locked_timer == 0:
                    # Trigger modular expansion
                    cost_flexible += capex_modular / ((1 + r) ** t)
                    current_cap += initial_capacity
                    expansion_locked_timer = buffer_window # buffer period
            
            if expansion_locked_timer > 0:
                expansion_locked_timer -= 1
                if path[t] > current_cap:
                    # Minor non-compliance risk during buffer lag
                    penalty = penalty_cost
                    
            cost_flexible += (op + penalty) / ((1 + r) ** t)
            
        shd_ros_npvs.append(cost_flexible)
        
    return np.array(rigid_npvs), np.array(shd_ros_npvs)

rigid_npvs, shd_ros_npvs = evaluate_strategies(flow_paths, capex_rigid, capex_modular, unit_opex, penalty_cost, r, q_crit_ratio, land_option_cost)

# --- Dashboard Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Inflow & Jump Simulation", "💰 ENPV & Cost Risk Analysis", "🏗️ Real Options Triggers", "📋 Executive Summary Report"])

with tab1:
    st.subheader("Stochastic Inflow Trajectories (Jump-Diffusion Model)")
    st.markdown("Simulating municipal sewage inflow ($Q_t$) accounting for baseline urban growth ($\mu_Q$), seasonal volatility ($\sigma_Q$), and sudden development surges ($\lambda, J_Q$).")
    
    # Plotting flow paths using Plotly
    fig_flow = go.Figure()
    # Plot a sample of paths to keep rendering fast
    for i in range(min(50, num_simulations)):
        fig_flow.add_trace(go.Scatter(x=time_axis, y=flow_paths[i], mode='lines', line=dict(color='rgba(14, 76, 146, 0.15)', width=1), showlegend=False))
    
    # Mean path
    mean_path = np.mean(flow_paths, axis=0)
    fig_flow.add_trace(go.Scatter(x=time_axis, y=mean_path, mode='lines', line=dict(color='#0E4C92', width=3), name='Expected Mean Flow'))
    
    fig_flow.update_layout(xaxis_title="Planning Year (t)", yaxis_title="Influent Flow (MLD)", template="plotly_white", height=450)
    st.plotly_chart(fig_flow, use_container_width=True)

with tab2:
    st.subheader("Expected Net Present Value (ENPV) Risk Distribution")
    st.markdown("Comparing the **Traditional Rigid Strategy** (Over-Sizing Trap / High Initial CapEx) against the **SHD-ROS Flexible Strategy** (Phased Modular Staging).")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Rigid ENPV", f"RM {np.mean(rigid_npvs):.2f}M", delta="High CapEx Exposure")
    col2.metric("Mean SHD-ROS ENPV", f"RM {np.mean(shd_ros_npvs):.2f}M", delta=f"-{(np.mean(rigid_npvs)-np.mean(shd_ros_npvs))/np.mean(rigid_npvs)*100:.1f}% Savings", delta_color="inverse")
    col3.metric("Value of Flexibility (VoF)", f"RM {np.mean(rigid_npvs) - np.mean(shd_ros_npvs):.2f}M", delta="Net Lifecycle Benefit")
    
    # Distribution Plot
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=rigid_npvs, name='Traditional Rigid Design', marker_color='#E63946', opacity=0.7))
    fig_hist.add_trace(go.Histogram(x=shd_ros_npvs, name='SHD-ROS Flexible Design', marker_color='#2A9D8F', opacity=0.7))
    fig_hist.update_layout(barmode='overlay', xaxis_title="Lifecycle ENPV (RM Million)", yaxis_title="Simulation Frequency", template="plotly_white", height=420)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("Real Options Modular Expansion Triggers")
    st.markdown("Visualizing how modular capacity triggers ($Q_{crit}$) activate secondary phases only when demand crosses critical statistical thresholds, bypassing the Under-Sizing Trap.")
    
    sample_path = flow_paths[0]
    crit_threshold_line = np.ones_like(sample_path) * (np.max(flow_paths)*0.85 * q_crit_ratio)
    
    fig_trig = go.Figure()
    fig_trig.add_trace(go.Scatter(x=time_axis, y=sample_path, mode='lines+markers', name='Simulated Inflow Path', line=dict(color='#0E4C92', width=2)))
    fig_trig.add_trace(go.Scatter(x=time_axis, y=crit_threshold_line, mode='lines', name='Critical Trigger Line (Q_crit)', line=dict(color='#E63946', width=2, dash='dash')))
    
    fig_trig.update_layout(xaxis_title="Time (Years)", yaxis_title="Flow Rate (MLD)", template="plotly_white", height=400)
    st.plotly_chart(fig_trig, use_container_width=True)

with tab4:
    st.subheader("Executive Decision Support & Recommendations")
    st.markdown("### Summary of Optimization Results")
    
    summary_df = pd.DataFrame({
        "Metric": ["Initial CapEx Exposure", "Mean Lifecycle ENPV", "Asset Utilization Efficiency", "Regulatory Compliance Risk"],
        "Traditional Rigid Plan": [f"RM {capex_rigid}M (Front-loaded)", f"RM {np.mean(rigid_npvs):.2f}M", "Low (Initial Years)", "Low (Oversized)"],
        "SHD-ROS Framework": [f"RM {capex_modular + land_option_cost}M (Staged)", f"RM {np.mean(shd_ros_npvs):.2f}M", "Optimized (High)", "Controlled via Triggers"]
    })
    
    st.table(summary_df)
    
    st.info("💡 **Key Industry Takeaway**: Implementing the SHD-ROS framework reduces upfront capital exposure by deferring secondary civil works until stochastic jump-diffusion thresholds ($\lambda, J_Q$) are breached, successfully balancing lifecycle energy OpEx and regulatory penalty limits.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #777;'>SHD-ROS Simulation Engine | Powered by Streamlit & Python | Municipal Infrastructure Planning</p>", unsafe_allow_html=True)
