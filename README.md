# Marine Safety Monitoring & ML Forecaster Service

A physics-informed meteorological and oceanographic forecasting pipeline and inference microservice for Camp Freedive PH in Mabini / Anilao, Batangas.

---

## Overview

This service provides multi-horizon environmental forecasts and automated marine safety evaluations for open-water freediving operations. It evaluates wave conditions, surface currents, wind speeds, and barometric pressure trends to deliver Go/No-Go safety clearances and planning outlooks.

---

## Pipeline Structure

The ML pipeline is structured in two stages:

1. **Stage 1: Multi-Variable Environmental Regressors**
   - **Wave Regressor**: Predicts significant wave height ($H_s$), peak wave period ($T_p$), swell height, and wind-wave height.
   - **Wind Regressor**: Predicts wind speed, wind gusts, 3-hour barometric pressure tendency ($\Delta P_{3h}$), and wind direction via $\sin/\cos$ vector decomposition.
   - **Ocean Current Regressor**: Predicts zonal ($u$) and meridional ($v$) surface current velocities via vector regression.

2. **Stage 2: Safety Classification & Thresholds**
   - **Stacked Safety Classifier**: Fuses Stage-1 predictions into a 5-tier safety classification (`Very Safe`, `Safe`, `Moderate`, `High Risk`, `Critical Risk`) optimized with an asymmetric cost matrix that penalizes false negatives ($5\times$ to $20\times$).
   - **Deterministic Physical Thresholds**: Overrides model output with `Critical Risk` if wave height $\ge 2.2\text{ m}$, wind speed $\ge 10.5\text{ m/s}$, current speed $\ge 0.75\text{ m/s}$, or if active PAGASA/PCG storm signals exist.
   - **Operational Horizon Policy**: 
     - $T-1\text{h}$ (Tactical Clearance): Authoritative dockside Go/No-Go verdict.
     - $T-6\text{h}/24\text{h}$ (Provisional Trend Outlook): Discrete badge suppressed to prevent false reassurance; surfaces raw physics and P90 tail risk bounds.
     - $T-48\text{h}+$ (Extended Trend Outlook): Long-range planning outlook with physical backstops.

---

## Key Domain Physics & Design Decisions

- **Vector Regression for Currents**: Models orthogonal $u$ and $v$ vectors directly to ensure mass conservation and derive physical current speed and direction.
- **Cyclic Angular Decomposition**: Encodes wind and current angles using $\sin(\theta)$ and $\cos(\theta)$ to avoid $0^\circ / 360^\circ$ boundary discontinuities.
- **Diurnal Solar Tide Filter ($S_2$)**: Isolates the daily astronomical solar tide pressure oscillation (~2.5 hPa swing at 10 AM / 10 PM) from squall-induced drops by requiring concurrent wind gust ($\ge 10\text{ m/s}$) or heavy rain ($\ge 15\text{ mm/h}$) triggers.
- **Fast ONNX Serving**: Models are exported to ONNX format and loaded via ONNX Runtime for sub-millisecond inference ($<5\text{ ms}$ SLA target).

---

## Tech Stack

- **Language**: Python 3.10+
- **Serving**: FastAPI, Uvicorn, Pydantic v2
- **Inference**: ONNX Runtime (`onnxruntime`)
- **Machine Learning**: XGBoost, Scikit-Learn, Optuna
- **Data & Atmospheric Processing**: NumPy, Pandas, Xarray, NetCDF4, PyArrow

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Virtual environment (`venv` or `conda`)

### Setup & Execution

1. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI service** (Port 8001):
   ```bash
   uvicorn src.serve.main:app --host 127.0.0.1 --port 8001 --reload
   ```

4. **Health check**:
   ```bash
   curl http://127.0.0.1:8001/health
   ```

---

## Testing & Validation

Run the test suite:

```bash
# Verify physical safety thresholds and horizon policies (30 checks)
python src/serve/safety_thresholds.py

# Run live endpoint and ONNX inference validation
python src/serve/test_service.py

# Run baseline comparison benchmarks (Persistence vs Climatology)
python src/models/baselines.py
```

---

## Documentation Standards

The codebase follows the 4-Pillar Documentation Framework:
- **Explain the "Why"**: Details domain physics, marine safety limits, and architectural trade-offs.
- **Document Classes & Functions**: Standard PEP-257 docstrings with parameter types and return descriptions.
- **Zero Stale References**: Accurate identifiers with zero obsolete references.
- **Targeted Roadmap Notes**: High-value `TODO` notes outlining future scalability directions.
