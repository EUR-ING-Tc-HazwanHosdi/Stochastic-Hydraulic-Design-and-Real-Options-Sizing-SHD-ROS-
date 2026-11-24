# Stochastic Hydraulic Design and Real Options Sizing (SHD-ROS)

**Stochastic Hydraulic Design and Real Options Sizing of Municipal Sewerage Systems: Balancing Capital Outlays, Energy Efficiency, and Regulatory Compliance**

- **Interactive Web App:** [https://shd-ros.streamlit.app/](https://shd-ros.streamlit.app/)
- **Code & Data Archive (Zenodo):** [https://doi.org/10.5281/zenodo.23655268](https://doi.org/10.5281/zenodo.23655268)
- **Author:** EUR ING Tc. Hazwan B. Hosdi (Planning and Engineering Department (CADCS), Indah Water Konsortium Sdn Bhd / WorldQuant University)

---

## Overview

Traditional municipal sewerage planning often relies on static 20- to 30-year demographic projections, exposing utilities to either:
- **The Over-Sizing Trap:** Front-loading high capital expenditure (`CapEx`) into monolithic assets that remain underutilized when demand growth lags projections.
- **The Under-Sizing Trap:** Facing premature capacity exhaustion, hydraulic bottlenecks, emergency retrofits, and statutory non-compliance penalties when urban growth, wet-weather infiltration/inflow (`I/I`), or discrete commercial/industrial development shocks exceed baseline forecasts.

The **SHD-ROS** framework models influent sewage flow uncertainty using a continuous-time jump-diffusion process and evaluates **Real Options** staging policies across Monte Carlo realizations, combining:
1. **"On-System" Land Option Pre-Purchasing** (`CapEx_land` at $t = 0$) to preserve physical footprint corridors for future treatment modules.
2. **"In-System" Modular Expansion Triggers** (`Q_crit` of active capacity `C_t`, subject to a construction/verification lag window `τ_buffer`) to stage modular treatment trains (`CapEx_mod`) as demand materializes.

---

## Baseline Simulation Parameters & Benchmark Results

All baseline results in the manuscript and default Streamlit simulation (`N_sim = 2,000`, `seed = 42`, `T = 30` years, `Δt = 1` year) are parameterized against Malaysian Sewerage Industry Guidelines (SPAN MSIG, 210 L/PE/day) and utility planning benchmarks:

| Parameter / Metric | Baseline Value / Result | Notes |
| :--- | :--- | :--- |
| Initial Inflow ($Q_0$) | `20.0 MLD` ($\approx 95,238\text{ PE}$) | At $210\text{ L/PE/day}$ MSIG benchmark |
| Planning Horizon ($T$, $\Delta t$) | `30 years`, `1 yr` step | Annual discrete simulation steps |
| Drift ($\mu_Q$), Volatility ($\sigma_Q$) | `0.03 yr⁻¹` (`3.0%`), `0.08 yr⁻¹/²` (`8.0%`) | Baseline growth and continuous variance |
| Jump Intensity ($\lambda$), Mean Jump ($J_Q$) | `0.10 shocks/yr`, `0.20` (`20%`, exponential) | Discrete development shocks |
| Discount Rate ($r$) | `6.0% / yr` | Social / infrastructure discount rate |
| Rigid CapEx ($t = 0$) | `RM 45.00M` | Monolithic upfront plant outlay |
| Flexible Initial Outlay ($t = 0$) | `RM 16.50M` (`RM 15.00M` train + `RM 1.50M` land) | `63.33%` (`-RM 28.50M`) upfront CapEx reduction |
| Unit OpEx ($c_{\text{opex}}$) & Penalty ($c_{\text{pen}}$) | `RM 0.35M / MLD / yr`, `RM 5.00M / yr` | Penalty applies if $Q_{i,t} > C_{i,t}$ during active lag |
| Trigger Ratio ($Q_{\text{crit}}$) & Buffer ($\tau_{\text{buffer}}$) | `0.85` (`85%` of active $C_t$), `2 years` | Dynamic active-capacity trigger |
| **Mean Lifecycle ENPV (Rigid vs. SHD-ROS)** | **RM 233.42M** vs. **RM 225.50M** | Mean Value of Flexibility (`VoF`) = **RM 7.92M** (`3.39%`) |
| **P10 / P90 Lifecycle ENPV (Rigid)** | **RM 171.82M** / **RM 313.82M** | `SD = RM 64.20M` |
| **P10 / P90 Lifecycle ENPV (SHD-ROS)** | **RM 162.53M** / **RM 306.10M** | `SD = RM 64.91M`; P90 `VoF` = **RM 13.83M** |
| **Expansions & Penalty Exposure** | `3.02` expansions/path, `0.17` penalty-yrs/path | Mean across `2,000` Monte Carlo paths |

---

## Repository Structure

```text
├── app.py                  # Streamlit interactive decision-support application
├── simulate_shd_ros.py     # Standalone reproducible Monte Carlo & 300-DPI figure script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation and reproducibility guide
```

---

## Installation & Usage

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib plotly streamlit
```

### 2. Run the Interactive Streamlit App

```bash
streamlit run app.py
```

Or use the hosted application directly at [https://shd-ros.streamlit.app/](https://shd-ros.streamlit.app/).

### 3. Reproduce Manuscript Statistics & 300-DPI Figures

```bash
python simulate_shd_ros.py
```

This generates the `2,000`-run summary metrics (`seed = 42`) and publication-grade `300 DPI` figures:
- `fig1_enpv_dist.png` — Lifecycle ENPV distribution comparison
- `fig2_inflow_paths.png` — Jump-diffusion inflow trajectories and ensemble mean
- `fig3_triggers.png` — Dynamic active capacity, `85%` trigger threshold, and sample inflow path

---

## Citation & Archive

If you use this code or framework in your research or planning work, please cite the Zenodo archive and associated manuscript:

- **Zenodo DOI:** `10.5281/zenodo.23655268` ([https://doi.org/10.5281/zenodo.23655268](https://doi.org/10.5281/zenodo.23655268))
