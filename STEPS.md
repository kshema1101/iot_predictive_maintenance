# How to Run — Step by Step

Total time: ~20 minutes (most of it waiting for ThingsBoard to boot)

---

## Prerequisites

- **Docker Desktop** installed and running
- **Python 3.10+** with pip
- **~2GB free RAM** (ThingsBoard needs ~1.5GB)
- **macOS users:** `brew install libomp` (required for XGBoost)

```bash
docker --version
python3 --version
```

---

## Step 1: Install Python Dependencies

```bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
pip install numpy pandas scikit-learn xgboost joblib
```

---

## Step 2: Generate Training Data

```bash
python ml/generate_dataset.py --switches 100 --cycles 120
```

**What this does:** Runs the simulator offline to generate 12,000 labeled telemetry samples across 5 failure modes (healthy, mechanical_friction, blockage, electrical, bearing_wear).

**Output:** `ml/data/training_data.csv`

---

## Step 3: Train ML Models

```bash
python ml/train_models.py --skip-lstm
```

**What this does:** Trains two models:
1. **Random Forest / XGBoost Classifier** — predicts failure mode (picks the best via hyperparameter tuning)
2. **XGBoost Regressor** — predicts Remaining Useful Life (RUL)

**Output:** Model files in `ml/models/` (`.joblib` files)

**Optional:** Train all 3 models including LSTM (requires TensorFlow):
```bash
pip install tensorflow
python ml/train_models.py
```

**Optional:** Skip hyperparameter tuning for faster training:
```bash
python ml/train_models.py --skip-lstm --no-tune
```

Verify models exist:
```bash
ls ml/models/
# Should show: failure_classifier_rf.joblib  rul_predictor_xgb.joblib  etc.
```

---

## Step 4: Start ThingsBoard

```bash
docker-compose up -d thingsboard
```

Wait ~90 seconds for ThingsBoard to fully boot:
```bash
docker-compose logs -f thingsboard
# Wait until you see "Started ThingsBoard in XXX seconds"
# Press Ctrl+C to stop watching
```

Verify it's healthy:
```bash
docker-compose ps
# thingsboard should show "healthy"
```

---

## Step 5: Configure ThingsBoard (manual, one-time)

Open **http://localhost:8080** and log in:
- **Email:** `tenant@thingsboard.org`
- **Password:** `tenant`

### 5.1 Create Device Profile

1. Left menu → **Device profiles** → click **+**
2. Name: `Railway Point Machine`
3. Transport type: MQTT (default)
4. Click **Add**

### 5.2 Create 5 Devices

For each row, go to **Devices** → **+** → **Add new device** → set Name and Profile → **Next: Credentials** → set Access Token → **Add**

| Device Name             | Access Token         | Label                |
|-------------------------|----------------------|----------------------|
| Railway Switch SW-001   | `RAILWAY_SWITCH_01`  | Platform 3 Junction  |
| Railway Switch SW-002   | `RAILWAY_SWITCH_02`  | North Yard Entry     |
| Railway Switch SW-003   | `RAILWAY_SWITCH_03`  | Depot Throat         |
| Railway Switch SW-004   | `RAILWAY_SWITCH_04`  | South Junction       |
| Railway Switch SW-005   | `RAILWAY_SWITCH_05`  | East Crossover       |

**The access tokens MUST match `switch_config.json` exactly.**

---

## Step 6: Start Simulator and ML Predictor

```bash
docker-compose up -d simulator ml-predictor
```

Watch the logs:
```bash
docker-compose logs -f simulator ml-predictor
```

**Simulator output:** Fleet cycle data for all 5 switches every 5 seconds.
**ML Predictor output:** Failure mode predictions and RUL estimates per switch.

Verify in ThingsBoard: **Devices** → click any device → **Latest telemetry** — you should see sensor data + ML predictions (`ml_failure_mode`, `ml_rul_cycles`, `ml_failure_confidence`).

---

## Step 7: Set Up Rule Engine (Health Scoring)

1. Left menu → **Rule chains** → open **Root Rule Chain**
2. From the left palette, drag a **Transformation → Script** node onto the canvas
3. Name: `Health Scorer`
4. Paste the contents of `rule_engine_script.js` (Language: TBEL)
5. Click **Add**
6. Drag a new **Action → Save Timeseries** node onto the canvas, name it `Save Enriched Data`
7. Wire the flow:
   - Remove the existing `Message Type Switch` → `Save Timeseries` connection (the "Post telemetry" link)
   - Connect: `Message Type Switch` →(Post telemetry)→ `Health Scorer`
   - Connect: `Health Scorer` →(Success)→ `Save Timeseries` (original node)
8. Click **Apply changes** (checkmark, top right)

Verify: **Devices** → any device → **Latest telemetry** should now show `health_index`, `status_message`, `severity_level`, `maintenance_required`.

---

## Step 8: Create Dashboard

1. Left menu → **Dashboards** → **+** → Create new dashboard → Name: `Railway Switch Fleet` → **Add**
2. Open the dashboard → click **pencil icon** (edit mode)
3. Click **Entity aliases** (chain icon, top right) → **Add alias**:
   - Alias: `All Switches`, Filter type: `Device type`, Device type: `Railway Point Machine`
   - Alias: `SW-002`, Filter type: `Single entity`, Device: `Railway Switch SW-002`

**Widget 1 — Fleet Table:**
- Add widget → Tables → Entities table
- Datasource: `All Switches` alias
- Columns: `entityName`, `health_index`, `status_message`, `motor_current`, `transition_time`, `ml_failure_mode`, `ml_rul_cycles`

**Widget 2 — Time-Series Chart:**
- Add widget → Charts → Timeseries Line Chart
- Datasource: `SW-002` alias
- Keys: `motor_current`, `transition_time`, `health_index`

**Widget 3 — Health Gauge:**
- Add widget → Analogue gauges → Radial gauge
- Datasource: `SW-002` alias
- Key: `health_index` (Min: 0, Max: 100)

4. Click **checkmark** to save

---

## What to Watch

- **SW-001 & SW-005** stay healthy (no failure scenario)
- **SW-002** degrades around cycle 10 (mechanical friction)
- **SW-003** degrades around cycle 20 (blockage)
- **SW-004** degrades around cycle 15 (electrical)

The ML predictor detects failures before the rule engine thresholds trigger — that's predictive maintenance.

---

## Useful Commands

```bash
docker-compose up -d                    # Start everything
docker-compose down                     # Stop everything
docker-compose down -v                  # Stop + delete all data (fresh start)
docker-compose up -d --build            # Rebuild after code changes
docker-compose ps                       # Check service health
docker-compose logs -f                  # Follow all logs
docker-compose logs -f simulator        # Follow specific service
docker-compose logs -f ml-predictor     # Follow ML predictor
```

---

## File Reference

| File | Purpose |
|------|---------|
| `simulator.py` | Fleet simulator — 5 railway switches publishing telemetry via MQTT |
| `switch_config.json` | Switch definitions — IDs, tokens, failure scenarios |
| `ml/generate_dataset.py` | Generates synthetic training data from the simulator |
| `ml/train_models.py` | Trains RF/XGBoost classifiers + XGBoost RUL regressor |
| `ml/predictor.py` | Live ML inference — polls ThingsBoard, posts predictions back |
| `rule_engine_script.js` | ThingsBoard TBEL script for health scoring |
| `docker-compose.yml` | Orchestrates ThingsBoard, Simulator, ML Predictor |
| `services/Dockerfile.simulator` | Docker image for the simulator |
| `services/Dockerfile.ml` | Docker image for the ML predictor |
| `widget/switch_health_widget.html` | Custom ThingsBoard widget (optional) |
| `widget/switch_health_controller.js` | Widget controller (optional) |
| `requirements.txt` | Full Python dependencies |
| `requirements-ml.txt` | ML predictor container dependencies |
| `requirements-simulator.txt` | Simulator container dependencies |

---
