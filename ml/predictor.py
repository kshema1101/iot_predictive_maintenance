"""
Live ML Predictor for Predictive Maintenance

Three operating modes:
  --standalone              No MQTT. Runs simulator internally, prints predictions.
  (default)                 Generic MQTT (Mosquitto). Subscribes to iot/switches/+/telemetry
  --thingsboard-api         Polls ThingsBoard REST API, pushes predictions back via HTTP.

The ThingsBoard API mode is the correct pattern for microservice deployment:
the predictor polls each device's latest telemetry from ThingsBoard's REST API,
runs ML inference, and posts predictions back as new telemetry keys on the same device.

Usage:
  python ml/predictor.py --standalone
  python ml/predictor.py --host localhost --port 1883
  python ml/predictor.py --thingsboard-api --tb-url http://localhost:8080
"""

import json
import logging
import argparse
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import deque

import numpy as np
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ML-Predictor] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Predictor")

MODEL_DIR = Path(__file__).resolve().parent / "models"

FEATURE_COLS = [
    "motor_current", "transition_time", "vibration_peak",
    "supply_voltage", "motor_temperature",
    "current_x_time", "vibration_x_current", "power_draw",
]

SEQUENCE_FEATURES = [
    "motor_current", "transition_time", "vibration_peak",
    "supply_voltage", "motor_temperature",
]

GENERIC_TELEMETRY_TOPIC = "iot/switches/+/telemetry"
GENERIC_PREDICTION_TOPIC = "iot/predictions/{switch_id}"
TB_TELEMETRY_TOPIC = "v1/devices/me/telemetry"
PREDICTOR_TOKEN = "ML_PREDICTOR_SERVICE"
WINDOW_SIZE = 10
FORECAST_HORIZON = 5

# ThingsBoard device tokens (must match what you configured in TB)
DEVICE_TOKENS = {
    "SW-001": "RAILWAY_SWITCH_01",
    "SW-002": "RAILWAY_SWITCH_02",
    "SW-003": "RAILWAY_SWITCH_03",
    "SW-004": "RAILWAY_SWITCH_04",
    "SW-005": "RAILWAY_SWITCH_05",
}


class PredictiveMaintenanceEngine:
    """Loads trained models and runs inference on live telemetry."""

    def __init__(self):
        self.rf_model = None
        self.rf_scaler = None
        self.rf_label_encoder = None
        self.xgb_model = None
        self.xgb_scaler = None
        self.lstm_model = None
        self.lstm_scaler = None

        self.history: dict[str, deque] = {}
        self._load_models()

    def _load_models(self):
        rf_path = MODEL_DIR / "failure_classifier_rf.joblib"
        if rf_path.exists():
            self.rf_model = joblib.load(rf_path)
            self.rf_scaler = joblib.load(MODEL_DIR / "failure_scaler.joblib")
            self.rf_label_encoder = joblib.load(MODEL_DIR / "failure_label_encoder.joblib")
            log.info("Loaded Random Forest classifier")
        else:
            log.warning("Random Forest model not found at %s", rf_path)

        xgb_path = MODEL_DIR / "rul_predictor_xgb.joblib"
        if xgb_path.exists():
            self.xgb_model = joblib.load(xgb_path)
            self.xgb_scaler = joblib.load(MODEL_DIR / "rul_scaler.joblib")
            log.info("Loaded XGBoost RUL predictor")
        else:
            log.warning("XGBoost model not found at %s", xgb_path)

        lstm_path = MODEL_DIR / "lstm_forecaster.keras"
        if lstm_path.exists():
            try:
                from tensorflow import keras
                self.lstm_model = keras.models.load_model(str(lstm_path))
                self.lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
                log.info("Loaded LSTM forecaster")
            except ImportError:
                log.warning("TensorFlow not installed — LSTM predictions disabled")
        else:
            log.warning("LSTM model not found at %s", lstm_path)

    def _engineer_features(self, telemetry: dict) -> dict:
        mc = telemetry.get("motor_current", 0)
        tt = telemetry.get("transition_time", 0)
        vp = telemetry.get("vibration_peak", 0)
        sv = telemetry.get("supply_voltage", 24)
        return {
            "current_x_time": round(mc * tt, 2),
            "vibration_x_current": round(vp * mc, 3),
            "power_draw": round(mc * sv, 2),
        }

    def predict(self, telemetry: dict) -> dict:
        result = {}
        engineered = self._engineer_features(telemetry)

        if self.rf_model:
            features = [telemetry.get(f, 0) for f in FEATURE_COLS[:5]]
            features += [engineered["current_x_time"],
                         engineered["vibration_x_current"],
                         engineered["power_draw"]]
            X = np.array(features).reshape(1, -1)
            X_scaled = self.rf_scaler.transform(X)
            pred_class = self.rf_model.predict(X_scaled)[0]
            pred_proba = self.rf_model.predict_proba(X_scaled)[0]
            confidence = float(np.max(pred_proba))
            failure_mode = self.rf_label_encoder.inverse_transform([pred_class])[0]
            result["ml_failure_mode"] = failure_mode
            result["ml_failure_confidence"] = round(confidence, 3)
            for cls, prob in zip(self.rf_label_encoder.classes_, pred_proba):
                result[f"ml_prob_{cls}"] = round(float(prob), 3)

        if self.xgb_model:
            features = [telemetry.get(f, 0) for f in FEATURE_COLS[:5]]
            features += [engineered["current_x_time"],
                         engineered["vibration_x_current"],
                         engineered["power_draw"]]
            features.append(telemetry.get("degradation_progress", 0))
            X = np.array(features).reshape(1, -1)
            X_scaled = self.xgb_scaler.transform(X)
            rul = max(0, float(self.xgb_model.predict(X_scaled)[0]))
            result["ml_rul_cycles"] = round(rul, 1)

        switch_id = telemetry.get("switch_id", "unknown")
        seq_values = [telemetry.get(f, 0) for f in SEQUENCE_FEATURES]
        if switch_id not in self.history:
            self.history[switch_id] = deque(maxlen=WINDOW_SIZE)
        self.history[switch_id].append(seq_values)

        if self.lstm_model and len(self.history[switch_id]) == WINDOW_SIZE:
            window = np.array(list(self.history[switch_id]))
            n_feat = window.shape[1]
            window_scaled = self.lstm_scaler.transform(
                window.reshape(-1, n_feat)
            ).reshape(1, WINDOW_SIZE, n_feat)
            forecast_scaled = self.lstm_model.predict(window_scaled, verbose=0)
            forecast = self.lstm_scaler.inverse_transform(
                forecast_scaled.reshape(-1, n_feat)
            ).reshape(FORECAST_HORIZON, n_feat)
            for step in range(FORECAST_HORIZON):
                for j, feat in enumerate(SEQUENCE_FEATURES):
                    result[f"ml_forecast_{feat}_t{step+1}"] = round(float(forecast[step, j]), 2)
            tt_idx = SEQUENCE_FEATURES.index("transition_time")
            result["ml_forecast_transition_t5"] = round(float(forecast[-1, tt_idx]), 1)

        return result

    def predict_and_log(self, telemetry: dict) -> dict:
        preds = self.predict(telemetry)
        switch_id = telemetry.get("switch_id", "?")
        mode = preds.get("ml_failure_mode", "?")
        conf = preds.get("ml_failure_confidence", 0)
        rul = preds.get("ml_rul_cycles", "?")
        fc_tt = preds.get("ml_forecast_transition_t5", "?")

        color = "\033[92m" if mode == "healthy" else "\033[93m" if conf < 0.7 else "\033[91m"
        reset = "\033[0m"
        print(
            f"  {color}ML{reset} [{switch_id}] "
            f"predicted={color}{mode}{reset} ({conf:.0%})  "
            f"RUL={rul} cycles  "
            f"forecast_tt@t+5={fc_tt}ms"
        )
        return preds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Mode 1: ThingsBoard REST API (microservice pattern)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TB_TENANT_USER = "tenant@thingsboard.org"
TB_TENANT_PASS = "tenant"


class ThingsBoardClient:
    """Handles JWT auth and REST calls to the ThingsBoard tenant API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.jwt_token: str | None = None
        self.device_ids: dict[str, str] = {}

    def _request(self, url: str, method: str = "GET",
                 payload: dict | None = None,
                 headers: dict | None = None,
                 _retry: bool = True) -> dict | None:
        hdrs = {"Content-Type": "application/json"}
        if self.jwt_token:
            hdrs["X-Authorization"] = f"Bearer {self.jwt_token}"
        if headers:
            hdrs.update(headers)
        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            if e.code == 401 and _retry:
                log.info("JWT token expired, re-authenticating...")
                if self.login():
                    return self._request(url, method, payload, headers, _retry=False)
            log.error("TB API %s %s → %d: %s", method, url, e.code, e.read().decode()[:200])
        except Exception as e:
            log.debug("TB API %s %s → %s", method, url, e)
        return None

    def login(self) -> bool:
        result = self._request(
            f"{self.base_url}/api/auth/login",
            method="POST",
            payload={"username": TB_TENANT_USER, "password": TB_TENANT_PASS},
        )
        if result and "token" in result:
            self.jwt_token = result["token"]
            log.info("Authenticated with ThingsBoard tenant API")
            return True
        log.error("Failed to authenticate with ThingsBoard")
        return False

    def resolve_device_ids(self, tokens: dict[str, str]) -> dict[str, str]:
        """Map switch_id → ThingsBoard device UUID by querying devices list."""
        result = self._request(
            f"{self.base_url}/api/tenant/devices?pageSize=100&page=0"
        )
        if not result or "data" not in result:
            log.error("Cannot fetch devices list")
            return {}

        token_to_switch = {v: k for k, v in tokens.items()}
        for device in result["data"]:
            dev_id = device["id"]["id"]
            cred = self._request(
                f"{self.base_url}/api/device/{dev_id}/credentials"
            )
            if cred:
                cred_id = cred.get("credentialsId", "")
                if cred_id in token_to_switch:
                    switch_id = token_to_switch[cred_id]
                    self.device_ids[switch_id] = dev_id
                    log.info("Resolved %s → device %s", switch_id, dev_id)

        return self.device_ids

    def get_latest_telemetry(self, device_uuid: str, keys: str) -> dict | None:
        return self._request(
            f"{self.base_url}/api/plugins/telemetry/DEVICE/{device_uuid}/values/timeseries?keys={keys}"
        )

    def post_device_telemetry(self, token: str, payload: dict) -> bool:
        """POST predictions via device HTTP API (no JWT needed)."""
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/v1/{token}/telemetry",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            log.error("Post telemetry error for %s: %s", token, e)
            return False


def run_thingsboard_api_predictor(tb_url: str, interval: int = 5):
    """
    Poll ThingsBoard for each device's latest telemetry via tenant REST API,
    run ML inference, and push predictions back via device HTTP API.

    Read path:  GET /api/plugins/telemetry/DEVICE/{id}/values/timeseries (JWT auth)
    Write path: POST /api/v1/{token}/telemetry (device token auth)
    """
    engine = PredictiveMaintenanceEngine()
    tb = ThingsBoardClient(tb_url)

    log.info("ThingsBoard API mode — polling %s every %ds", tb_url, interval)
    log.info("Devices configured: %s", list(DEVICE_TOKENS.keys()))

    # Wait for ThingsBoard to be ready
    for attempt in range(1, 31):
        try:
            req = urllib.request.Request(f"{tb_url}/login")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    log.info("ThingsBoard is ready")
                    break
        except Exception:
            pass
        log.info("Waiting for ThingsBoard... (attempt %d/30)", attempt)
        time.sleep(5)
    else:
        log.error("ThingsBoard not reachable at %s", tb_url)
        return

    if not tb.login():
        return

    device_ids = tb.resolve_device_ids(DEVICE_TOKENS)
    if not device_ids:
        log.error("No devices resolved — check that devices exist in ThingsBoard with matching tokens")
        return

    telemetry_keys = "motor_current,transition_time,vibration_peak,supply_voltage,motor_temperature,switch_id,cycle_count"

    while True:
        print(f"\n── ML Prediction Cycle ──")
        for switch_id, token in DEVICE_TOKENS.items():
            device_uuid = device_ids.get(switch_id)
            if not device_uuid:
                continue

            raw = tb.get_latest_telemetry(device_uuid, telemetry_keys)
            if not raw:
                log.debug("No data yet for %s", switch_id)
                continue

            telemetry = {"switch_id": switch_id}
            if isinstance(raw, dict):
                for key, values in raw.items():
                    if isinstance(values, list) and len(values) > 0:
                        telemetry[key] = values[0].get("value", values[0])
                    else:
                        telemetry[key] = values

            for key in ["motor_current", "transition_time", "vibration_peak",
                        "supply_voltage", "motor_temperature"]:
                if key in telemetry:
                    try:
                        telemetry[key] = float(telemetry[key])
                    except (ValueError, TypeError):
                        pass

            if "motor_current" not in telemetry:
                continue

            predictions = engine.predict_and_log(telemetry)

            if predictions:
                tb.post_device_telemetry(token, predictions)

        time.sleep(interval)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Mode 2: Generic MQTT (Mosquitto)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_mqtt_predictor(host: str, port: int, token: str):
    """Subscribe to Mosquitto, run ML inference, publish predictions."""
    import paho.mqtt.client as mqtt

    engine = PredictiveMaintenanceEngine()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ml-predictor")
    if token:
        client.username_pw_set(token)

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            log.info("Connected to Mosquitto — subscribing to: %s", GENERIC_TELEMETRY_TOPIC)
            client.subscribe(GENERIC_TELEMETRY_TOPIC)
        else:
            log.error("MQTT connection failed (rc=%d)", rc)

    def on_message(client, userdata, msg):
        try:
            telemetry = json.loads(msg.payload.decode())
            if "motor_current" not in telemetry:
                return
            predictions = engine.predict_and_log(telemetry)
            if predictions:
                switch_id = telemetry.get("switch_id", "unknown")
                pub_topic = GENERIC_PREDICTION_TOPIC.format(switch_id=switch_id)
                client.publish(pub_topic, json.dumps(predictions))
        except Exception as e:
            log.error("Prediction error: %s", e)

    client.on_connect = on_connect
    client.on_message = on_message

    log.info("Connecting to Mosquitto at %s:%d ...", host, port)
    try:
        client.connect(host, port, keepalive=60)
        client.loop_forever()
    except ConnectionRefusedError:
        log.error("Cannot connect to MQTT broker at %s:%d", host, port)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        client.disconnect()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Mode 3: Standalone (no MQTT, no ThingsBoard)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_standalone_demo():
    """Run predictions on simulated data without any broker."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from simulator import PointMachine, DegradationProfile

    engine = PredictiveMaintenanceEngine()
    switches = [
        PointMachine("DEMO-HEALTHY", "", "", 0),
        PointMachine("DEMO-FRICTION", "", "", 0,
                     degradation=DegradationProfile("mechanical_friction", 5, 30)),
        PointMachine("DEMO-BLOCKAGE", "", "", 0,
                     degradation=DegradationProfile("blockage", 8, 25)),
    ]

    print("\n" + "=" * 70)
    print("  STANDALONE ML DEMO — No MQTT / ThingsBoard required")
    print("=" * 70)

    for cycle in range(1, 51):
        print(f"\n── Cycle {cycle} ──")
        for sw in switches:
            telemetry = sw.generate_telemetry()
            engine.predict_and_log(telemetry)
        time.sleep(0.5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="ML Predictive Maintenance Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --standalone           No broker needed. Simulates data internally.
  (default)              Generic MQTT (Mosquitto) subscriber.
  --thingsboard-api      Polls ThingsBoard REST API. Best for Docker deployment.
        """,
    )
    parser.add_argument("--host", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--token", default="", help="MQTT access token")
    parser.add_argument("--thingsboard-api", action="store_true",
                        help="Use ThingsBoard REST API instead of MQTT")
    parser.add_argument("--tb-url", default="http://localhost:8080",
                        help="ThingsBoard base URL (for --thingsboard-api mode)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Polling interval in seconds (for --thingsboard-api)")
    parser.add_argument("--standalone", action="store_true",
                        help="Run without any broker (demo mode)")
    args = parser.parse_args()

    if args.standalone:
        run_standalone_demo()
    elif args.thingsboard_api:
        run_thingsboard_api_predictor(args.tb_url, args.interval)
    else:
        run_mqtt_predictor(args.host, args.port, args.token)


if __name__ == "__main__":
    main()
