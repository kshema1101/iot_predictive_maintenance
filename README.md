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

