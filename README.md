# Stochastic Hydraulic Design and Real Options Sizing (SHD-ROS)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SHD-ROS** is a quantitative engineering framework and interactive web application designed to evaluate municipal infrastructure investments under deep uncertainty. By integrating stochastic hydraulic sizing with Real Options Analysis (ROA), the framework models capital cost volatility using geometric Brownian motion ($It\hat{o}$ diffusion processes) to optimize phased construction timing, expansion thresholds, and budgetary risk mitigation.

---

## Key Features

- **Stochastic Cost & Demand Modeling:** Simulates multi-year capital expenditure (CapEx) volatility and hydraulic loading uncertainties using Monte Carlo simulations.
- **Real Options Valuation:** Implements binomial lattice and dynamic programming models to price managerial flexibility (the option to expand, defer, or contract network capacity).
- **Hydraulic Network Optimization:** Evaluates friction losses, peak flow factors, and diameter selection criteria for municipal sewer/water mains under probabilistic constraints.
- **Interactive Streamlit Dashboard:** Provides real-time visualization of stochastic asset trajectories, risk-adjusted net present value (rNPV) profiles, and optimal decision boundaries.

---

## Mathematical Framework

The evolution of project investment costs $C_t$ is modeled via the stochastic differential equation (SDE):

$$dC_t = \mu C_t dt + \sigma C_t dW_t$$

where $\mu$ represents the drift rate of infrastructure cost inflation, $\sigma$ is the volatility parameter, and $dW_t$ is the increment of a standard Brownian motion. Real options value $F(C_t, t)$ is derived by solving the corresponding Bellman equation and contingent claim formulations.

---

## Project Structure

```text
SHD-ROS/
│
├── app.py                  # Main Streamlit application interface
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
├── core/                   # Quantitative and hydraulic engines
│   ├── __init__.py
│   ├── hydraulics.py       # Hydraulic sizing and headloss equations
│   ├── stochastic.py       # Itô diffusion and Monte Carlo simulations
│   └── roa_engine.py       # Real Options Analysis valuation models
├── utils/                  # Helper functions and plotting modules
│   └── visualization.py    # Plotly/Matplotlib charting scripts
└── data/                   # Default parameter profiles and lookup tables
