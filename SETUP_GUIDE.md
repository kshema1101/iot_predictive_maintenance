# Setup Guide — Step by Step

This guide walks you through starting the entire system and configuring ThingsBoard from scratch.

**Total time: ~20 minutes** (most of it is waiting for ThingsBoard to boot)

---

## Step 0: Prerequisites

Make sure you have:
- **Docker Desktop** installed and running ([download](https://www.docker.com/products/docker-desktop/))
- **Python 3.10+** (for local ML training)
- **~2GB free RAM** (ThingsBoard needs ~1.5GB)

Verify Docker is working:
```bash
docker --version
docker-compose --version
```

---

## Step 1: Train the ML Models (local, before Docker)

The ML models need to exist before the predictor container starts.

```bash
# Install Python dependencies locally
pip install numpy pandas scikit-learn xgboost joblib

# Generate training data
python ml/generate_dataset.py --switches 50 --cycles 80

# Train Random Forest + XGBoost (skip LSTM to save time)
python ml/train_models.py --skip-lstm
```

You should see output like:
```
Loaded 4000 samples
MODEL 1: Random Forest — Failure Classification
  ...classification report...
MODEL 2: XGBoost — Remaining Useful Life (RUL)
  MAE = X.XX cycles
  R² = 0.XXXX
ALL MODELS TRAINED SUCCESSFULLY
```

Verify models exist:
```bash
ls ml/models/
# Should show: failure_classifier_rf.joblib  rul_predictor_xgb.joblib  etc.
```

---

## Step 2: Start ThingsBoard

```bash
# Start ThingsBoard only first (it takes ~90 seconds to boot)
docker-compose up -d thingsboard
```

**Wait for ThingsBoard to fully start** (~60-90 seconds on first run):
```bash
# Watch the logs until you see "Started ThingsBoard"
docker-compose logs -f thingsboard
```

Look for this line:
```
Started ThingsBoard in XXX seconds
```

Press `Ctrl+C` to stop watching logs.

Verify it's healthy:
```bash
docker-compose ps
```
`thingsboard` should show `healthy`.

---

## Step 3: Log in to ThingsBoard

1. Open your browser: **http://localhost:8080**
2. Log in with the **Tenant Administrator** account:
   - **Email:** `tenant@thingsboard.org`
   - **Password:** `tenant`

You should see the ThingsBoard dashboard home page.

---

## Step 4: Create Devices in ThingsBoard

Each switch needs a device in ThingsBoard with a matching access token.

### 4.1 Create a Device Profile (do this once)

1. Left menu → **Device profiles**
2. Click **+** (Add device profile)
3. Fill in:
   - **Name:** `Railway Point Machine`
   - **Rule chain:** `Root Rule Chain` (default)
   - **Transport type:** `MQTT` (default)
4. Click **Add**

### 4.2 Create the 5 Switch Devices

Repeat this for each switch. I'll show SW-001 as an example:

1. Left menu → **Devices**
2. Click **+** (Add new device)
3. Fill in:
   - **Name:** `Railway Switch SW-001`
   - **Device profile:** `Railway Point Machine`
   - **Label:** `Platform 3 Junction`
4. Click **Next: Credentials**
5. Set **Credentials type:** `Access Token`
6. Set **Access token:** `RAILWAY_SWITCH_01`  ← **THIS MUST MATCH switch_config.json**
7. Click **Add**

**Repeat for all 5 switches:**

| Device Name             | Access Token         | Label                          |
|-------------------------|----------------------|--------------------------------|
| Railway Switch SW-001   | `RAILWAY_SWITCH_01`  | Platform 3 Junction            |
| Railway Switch SW-002   | `RAILWAY_SWITCH_02`  | North Yard Entry               |
| Railway Switch SW-003   | `RAILWAY_SWITCH_03`  | Depot Throat                   |
| Railway Switch SW-004   | `RAILWAY_SWITCH_04`  | South Junction                 |
| Railway Switch SW-005   | `RAILWAY_SWITCH_05`  | East Crossover                 |

### 4.3 Verify Devices

Go to **Devices** → you should see all 5 devices listed with status "Inactive" (they haven't sent data yet).

---

## Step 5: Start the Application Services

Now start the simulator and ML predictor:

```bash
docker-compose up -d simulator ml-predictor
```

Watch the services:
```bash
docker-compose logs -f simulator ml-predictor
```

You should see:
- **Simulator:** Fleet cycle output with telemetry for all 5 switches, publishing directly to ThingsBoard's built-in MQTT broker on port 1883
- **ML Predictor:** Polling ThingsBoard REST API, running inference, posting predictions back

After ~10 seconds, go back to **ThingsBoard → Devices**. The devices should now show **Active** (green dot).

---

## Step 6: Verify Data is Flowing into ThingsBoard

1. Click on any device (e.g., `Railway Switch SW-002`)
2. Click the **Latest telemetry** tab
3. You should see live data updating every 5 seconds:
   - `motor_current`
   - `transition_time`
   - `vibration_peak`
   - `supply_voltage`
   - `motor_temperature`
   - `ml_failure_mode` (from the ML predictor)
   - `ml_rul_cycles`
   - `ml_failure_confidence`
   - `health_index` (if rule engine is configured)

If you see data here, **the entire pipeline is working**.

---

## Step 7: Set Up the Rule Engine

1. Left menu → **Rule chains**
2. Click on **Root Rule Chain** to open the editor
3. You'll see the default flow: `Input` → `Message Type Switch` → `Save Telemetry`

### Add the Health Scoring Script Node:

1. Drag a **Script** node from the left palette onto the canvas
2. Double-click it to configure:
   - **Name:** `Health Scorer`
   - **Script language:** `TBEL` (or JavaScript, depending on version)
3. Paste the contents of `rule_engine_script.js` into the script body
4. Click **Add**
5. **Connect it:** Draw a line from `Save Telemetry` → `Health Scorer`
6. Draw a line from `Health Scorer` → `Save Telemetry` (to save the enriched data back)
7. Click **Apply changes** (checkmark icon, top-right)

### Verify Rule Engine:

Go back to a device → Latest telemetry. You should now also see:
- `health_index` (0-100)
- `status_message` ("Optimal", "Degrading", etc.)
- `maintenance_required` (true/false)

---

## Step 8: Create a Dashboard

### 8.1 Create the Dashboard

1. Left menu → **Dashboards**
2. Click **+** → **Create new dashboard**
3. **Name:** `Railway Switch Fleet`
4. Click **Add**
5. Click on the new dashboard to open it
6. Click the **pencil icon** (edit mode) in the bottom-right

### 8.2 Add a Telemetry Table Widget

1. Click **Add widget** → **Tables** → **Entities table**
2. **Datasource:** Click **+ Add**, select **Entity type: Device**
3. **Entity name filter:** leave blank (shows all devices) or filter by device profile `Railway Point Machine`
4. **Columns:** Add these data keys:
   - `motor_current`
   - `transition_time`
   - `vibration_peak`
   - `health_index`
   - `ml_failure_mode`
   - `ml_rul_cycles`
5. Click **Add**

### 8.3 Add Time-Series Charts

1. Click **Add widget** → **Charts** → **Time series chart**
2. **Datasource:** Select device `Railway Switch SW-002`
3. **Data keys:** `motor_current`, `transition_time`
4. Click **Add**
5. Repeat for other devices or metrics

### 8.4 Add the Custom Widget (optional, for bonus points)

1. Left menu → **Widget Library** → **+** → **Create new widget bundle**
2. **Name:** `Railway Fleet`
3. Open the bundle → **+ Add widget type** → **Static**
4. **HTML tab:** Paste contents of `widget/switch_health_widget.html`
5. **JavaScript tab:** Paste contents of `widget/switch_health_controller.js`
6. Save and add to your dashboard

### 8.5 Save the Dashboard

Click the **checkmark icon** (save) in the bottom-right.

---

## Step 9: Watch the Demo

At this point you have the full system running:

```bash
# See all service logs
docker-compose logs -f

# Or just the interesting parts
docker-compose logs -f simulator ml-predictor
```

**What to watch for:**
- SW-001 and SW-005 stay **healthy** (green) — no failure scenario
- SW-002 starts degrading around cycle 10 (mechanical friction)
- SW-003 starts degrading around cycle 20 (blockage)  
- SW-004 starts degrading around cycle 15 (electrical)

The ML predictor should detect these failures **before** the rule engine thresholds trigger.

---

## Useful Commands

```bash
# Start everything
docker-compose up -d

# Stop everything  
docker-compose down

# Stop and delete all data (fresh start)
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build simulator ml-predictor

# View logs for specific service
docker-compose logs -f simulator
docker-compose logs -f ml-predictor
docker-compose logs -f thingsboard

# Check service health
docker-compose ps

# Open a shell in a container
docker exec -it simulator sh
docker exec -it ml-predictor sh

# Spy on MQTT messages going to ThingsBoard (from your Mac)
# Install: brew install mosquitto
mosquitto_sub -h localhost -p 1883 -t "#" -u "RAILWAY_SWITCH_01" -v
```

---

## Troubleshooting

### ThingsBoard won't start
- It needs ~90 seconds on first boot (database initialization)
- Check memory: `docker stats` — it needs ~1.5GB
- Check logs: `docker-compose logs thingsboard`

### Devices show "Inactive"
- Check the simulator is running: `docker-compose logs simulator`
- Verify tokens match: the access tokens in `switch_config.json` must match the tokens you set in ThingsBoard Step 4

### No telemetry data on devices
- Check simulator logs: `docker-compose logs -f simulator`
- Make sure you created devices with the **exact** access tokens from the table in Step 4
- Test MQTT manually: `mosquitto_pub -h localhost -p 1883 -t "v1/devices/me/telemetry" -u "RAILWAY_SWITCH_01" -m '{"test": 42}'`

### ML predictions not appearing
- Did you train models first? (Step 1)
- Check: `ls ml/models/` — should have `.joblib` files
- Check predictor logs: `docker-compose logs -f ml-predictor`

### Port conflicts
- Port 1883 (MQTT) or 8080 (web) already in use?
- Stop local Mosquitto: `brew services stop mosquitto`
- Change ports in docker-compose.yml: `"9090:9090"` instead of `"8080:9090"`
