# IoT Predictive Maintenance — Railway Switch Fleet

End-to-end IoT predictive maintenance system that monitors a **fleet** of railway point machines using **three ML models** to predict failures before they happen.

## Architecture — Microservices with Docker Compose

```
docker-compose up -d
┌──────────────────────────────────────────────────────────────┐
│                      Docker Compose                          │
│                                                              │
│  ┌────────────────┐    MQTT (port 1883)   ┌──────────────┐  │
│  │  Simulator      │ ──────────────────► │  ThingsBoard  │  │
│  │  (Python)       │  v1/devices/me/     │  CE           │  │
│  │  5 switches     │  telemetry          │               │  │
│  │  per-device     │                     │  Built-in:    │  │
│  │  MQTT auth      │                     │  MQTT Broker  │  │
│  └────────────────┘                      │  Rule Engine  │  │
│                                          │  PostgreSQL   │  │
│  ┌────────────────┐    REST API          │  Dashboards   │  │
│  │  ML Predictor   │ ◄─── GET telemetry  │  Device Mgmt  │  │
│  │  (Python)       │ ──── POST predict → │               │  │
│  │                  │  /api/v1/{token}/   │  :8080 Web UI │  │
│  │  Random Forest  │  telemetry          │  :1883 MQTT   │  │
│  │  XGBoost        │                     └──────────────┘  │
│  │  LSTM           │                                        │
│  └────────────────┘                                        │
└──────────────────────────────────────────────────────────────┘
```

### How data flows

1. **Simulator → ThingsBoard** via MQTT. Each switch authenticates with its own access token. ThingsBoard's built-in MQTT broker receives and stores telemetry.
2. **ML Predictor → ThingsBoard** via REST API. Predictor polls each device's latest telemetry (`GET /api/v1/{token}/telemetry`), runs RF + XGBoost + LSTM inference, and posts predictions back (`POST /api/v1/{token}/telemetry`).
3. **Rule Engine** (inside ThingsBoard) computes `health_index` and `status_message` from the raw telemetry.
4. **Dashboard** displays everything — raw sensors, ML predictions, and rule engine health scores.

### Why this design?

| Decision | Why |
|---|---|
| **ThingsBoard's built-in MQTT** | No separate broker to manage. ThingsBoard handles auth, storage, and routing natively. |
| **ML Predictor uses REST API** | In ThingsBoard, MQTT is per-device (one connection = one device). REST API lets the predictor read/write any device. |
| **Each service = one container** | Independent scaling, deployment, and failure isolation |
| **3 services, not 5** | Removed unnecessary Mosquitto and bridge. Simpler = easier to debug, explain, and demo. |

## ML Models

### Model 1: Random Forest Classifier — *"What type of failure is this?"*

| | |
|---|---|
| **Algorithm** | Random Forest (200 trees, balanced classes) |
| **Input** | 5 sensor features + 3 engineered features |
| **Output** | Failure mode: `healthy`, `mechanical_friction`, `blockage`, `electrical`, `bearing_wear` |
| **Why RF** | Handles multi-class well, gives feature importance (explainable to maintenance engineers), robust to noise |

**Engineered features** (cross-sensor correlations the model can't learn alone):
- `current × time` — high in mechanical friction (both climb)
- `vibration × current` — spikes in bearing wear
- `power_draw` (current × voltage) — drops during electrical faults

### Model 2: XGBoost Regressor — *"How many cycles until failure?"*

| | |
|---|---|
| **Algorithm** | XGBoost (300 rounds, depth 8, learning rate 0.05) |
| **Input** | 8 features + degradation_progress |
| **Output** | Remaining Useful Life in cycles |
| **Why XGBoost** | Gradient boosting excels at regression with mixed feature types, handles non-linear degradation curves |

### Model 3: LSTM Neural Network — *"What will the sensors read in 5 cycles?"*

| | |
|---|---|
| **Algorithm** | Encoder-Decoder LSTM (64→32 encode, 32 decode) |
| **Input** | Sliding window of last 10 readings (10 × 5 features) |
| **Output** | Forecasted next 5 readings (5 × 5 features) |
| **Why LSTM** | Captures temporal dependencies in sequential sensor data — a rising trend in current predicts continued rise |

**Architecture:**
```
Input(10, 5) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2)
            → RepeatVector(5) → LSTM(32) → Dropout(0.2)
            → TimeDistributed(Dense(5)) → Output(5, 5)
```

## Quick Start

### Full System (Docker Compose — recommended)

```bash
# 1. Train ML models locally first
pip install numpy pandas scikit-learn xgboost joblib
python ml/generate_dataset.py --switches 50 --cycles 80
python ml/train_models.py --skip-lstm

# 2. Start everything
docker-compose up -d

# 3. Watch the fleet
docker-compose logs -f simulator ml-predictor tb-bridge
```

Then follow **[SETUP_GUIDE.md](SETUP_GUIDE.md)** to configure ThingsBoard devices and dashboards.

- **ThingsBoard UI:** http://localhost:8080 (login: `tenant@thingsboard.org` / `tenant`)
- **MQTT spy:** `mosquitto_sub -h localhost -t "iot/#" -v`

### Offline Mode (no Docker needed)

```bash
pip install -r requirements.txt
python ml/generate_dataset.py && python ml/train_models.py --skip-lstm
python simulator.py --offline         # Terminal 1
python ml/predictor.py --standalone   # Terminal 2
```

## Dataset & Feature Engineering

The dataset generator (`ml/generate_dataset.py`) runs switches through their full lifecycle offline:

```
50 switches × 80 cycles = 4000 samples
├── 10 healthy switches (800 samples, label: "healthy")
├── 10 mechanical_friction (800 samples, degradation from cycle ~15)
├── 10 blockage (800 samples, degradation from cycle ~20)
├── 10 electrical (800 samples, degradation from cycle ~12)
└── 10 bearing_wear (800 samples, degradation from cycle ~18)
```

**Raw features** (from sensors):
- `motor_current` — Amps drawn by switch motor
- `transition_time` — Milliseconds to complete throw
- `vibration_peak` — g-force at peak motion
- `supply_voltage` — Rail power supply voltage
- `motor_temperature` — Motor housing temperature

**Engineered features** (computed cross-correlations):
- `current_x_time` — catches mechanical friction (both rise together)
- `vibration_x_current` — catches bearing wear (vibration + load)
- `power_draw` — catches electrical faults (voltage × current anomalies)

**Labels**:
- `failure_mode` — classification target (5 classes)
- `remaining_useful_life` — regression target (cycles until failure)
- `degradation_progress` — 0.0 (healthy) to 1.0 (failed)

## Failure Injection

Two injection methods that **compose** (stack via `max()`):

1. **Scenario-based** — pre-defined in `switch_config.json`, deterministic S-curve degradation
2. **RPC-based** — injected from dashboard button or MQTT command at runtime

| Mode | Motor Current | Transition Time | Signature |
|---|---|---|---|
| `mechanical_friction` | ↑↑ to ~9A | ↑↑ to ~6300ms | Both climb together |
| `blockage` | ↑ slight | ↑↑↑ to ~7800ms | Time spikes, current mild |
| `electrical` | ↑↑↑ erratic | ↑ moderate | Current dominant |
| `bearing_wear` | ↑ moderate | ↑ moderate | Balanced, slow onset |

## Project Structure

```
IOT_predictive_maintenance/
│
│  ── Docker / Microservices ──
├── docker-compose.yml              # Orchestrates all 4 services
├── mosquitto.conf                  # Mosquitto MQTT broker config
├── services/
│   ├── Dockerfile.simulator        # Container for fleet simulator
│   ├── Dockerfile.ml               # Container for ML predictor
│   ├── Dockerfile.bridge           # Container for ThingsBoard bridge
│   └── tb_bridge.py                # MQTT → ThingsBoard HTTP bridge service
│
│  ── Application ──
├── simulator.py                    # Fleet simulator (microservice 1)
├── switch_config.json              # Fleet definition (5 switches)
├── rule_engine_script.js           # ThingsBoard rule engine script
│
│  ── Machine Learning ──
├── ml/
│   ├── generate_dataset.py         # Offline dataset generator
│   ├── train_models.py             # Training pipeline (RF + XGBoost + LSTM)
│   ├── predictor.py                # ML predictor (microservice 2)
│   ├── data/
│   │   └── training_data.csv
│   └── models/
│       ├── failure_classifier_rf.joblib
│       ├── rul_predictor_xgb.joblib
│       └── ...
│
│  ── Dashboard ──
├── widget/
│   ├── switch_health_widget.html   # Fleet dashboard widget
│   └── switch_health_controller.js # Widget AngularJS controller
│
│  ── Documentation ──
├── requirements.txt                # All Python dependencies
├── requirements-simulator.txt      # Slim deps for simulator container
├── requirements-ml.txt             # Slim deps for ML container
├── SETUP_GUIDE.md                  # Step-by-step ThingsBoard setup
├── INTERVIEW_PREP.md               # Shortcomings, improvements, Q&A
└── README.md
```

## Interview Talking Points

### The ML story (how to explain it)

> "We have three layers of intelligence. First, a **rule engine** with physics-based thresholds — because a maintenance engineer needs to see 'transition_time exceeded 4500ms' to trust the system. Second, a **Random Forest** that classifies *which* failure mode is developing — this matters because mechanical friction needs lubrication while a blockage needs debris clearance. Third, an **XGBoost model** that predicts *when* it will fail — Remaining Useful Life — so maintenance can be scheduled between train passes. And an **LSTM** that forecasts sensor trajectories to catch acceleration in degradation."

### Why these specific algorithms

- **Random Forest over SVM** — feature importance is readable; maintenance teams need explainability
- **XGBoost over Linear Regression** — degradation is non-linear (S-curve); linear models underfit
- **LSTM over ARIMA** — multivariate (5 sensors at once); ARIMA is univariate and can't capture cross-sensor dependencies

### Questions to prepare for

- *"Why not just one model?"* — Different tasks: classification ≠ regression ≠ forecasting. A single model can't answer "what", "when", and "what next" simultaneously
- *"How do you retrain?"* — Scheduled retraining pipeline: collect confirmed failures, add to training set, retrain monthly. Monitor for data drift with KL divergence on feature distributions
- *"What about false positives?"* — The RF gives confidence scores. Set a threshold (e.g., >0.7) before alerting. Track precision/recall on production alerts and adjust
- *"How would you scale this?"* — Predictor runs as a microservice. One instance per 1000 devices. Models are stateless (except LSTM window which is just a deque). Horizontal scaling is trivial
- *"What's the cold start problem?"* — New switches have no history for LSTM. Fall back to RF + XGBoost (which work on single readings) until window fills up
