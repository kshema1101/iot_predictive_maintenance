"""
ThingsBoard Bridge — Event-Driven Microservice

Subscribes to the Mosquitto event bus and forwards telemetry + ML predictions
to ThingsBoard via its HTTP Telemetry API.

This decouples the simulator and ML predictor from ThingsBoard entirely.
They publish to generic MQTT topics; this bridge translates and forwards.

Event flow:
  Simulator → Mosquitto (iot/switches/+/telemetry)   → Bridge → ThingsBoard HTTP API
  ML        → Mosquitto (iot/predictions/+)           → Bridge → ThingsBoard HTTP API
  Simulator → Mosquitto (iot/switches/+/attributes)   → Bridge → ThingsBoard HTTP API

Environment variables:
  MQTT_HOST          Mosquitto host (default: mosquitto)
  MQTT_PORT          Mosquitto port (default: 1883)
  TB_HOST            ThingsBoard host (default: thingsboard)
  TB_PORT            ThingsBoard HTTP port (default: 9090)
  TB_TOKEN_MAP       JSON mapping switch_id → ThingsBoard access token
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error

import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [TB-Bridge] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TBBridge")

MQTT_HOST = os.environ.get("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TB_HOST = os.environ.get("TB_HOST", "thingsboard")
TB_PORT = os.environ.get("TB_PORT", "9090")

TOKEN_MAP_RAW = os.environ.get("TB_TOKEN_MAP", "{}")
TOKEN_MAP: dict[str, str] = json.loads(TOKEN_MAP_RAW)

TB_TELEMETRY_URL = f"http://{TB_HOST}:{TB_PORT}/api/v1/{{token}}/telemetry"
TB_ATTRIBUTES_URL = f"http://{TB_HOST}:{TB_PORT}/api/v1/{{token}}/attributes"

SUBSCRIBE_TOPICS = [
    ("iot/switches/+/telemetry", 1),
    ("iot/switches/+/attributes", 1),
    ("iot/predictions/+", 1),
]


def post_to_thingsboard(url: str, payload: dict) -> bool:
    """Send JSON payload to ThingsBoard HTTP API."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        log.error("TB HTTP %d: %s → %s", e.code, url, e.read().decode()[:200])
        return False
    except Exception as e:
        log.error("TB connection error: %s", e)
        return False


def extract_switch_id(topic: str) -> str | None:
    """Extract switch_id from topic like iot/switches/SW-001/telemetry."""
    parts = topic.split("/")
    if len(parts) >= 3:
        return parts[2]
    return None


def get_token(switch_id: str) -> str | None:
    """Look up ThingsBoard access token for a switch_id."""
    if switch_id in TOKEN_MAP:
        return TOKEN_MAP[switch_id]

    # Also try payload-level switch_id (from predictions topic)
    for key, token in TOKEN_MAP.items():
        if key.lower() == switch_id.lower():
            return token

    return None


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("Connected to Mosquitto at %s:%d", MQTT_HOST, MQTT_PORT)
        for topic, qos in SUBSCRIBE_TOPICS:
            client.subscribe(topic, qos)
            log.info("Subscribed to: %s", topic)
    else:
        log.error("MQTT connection failed (rc=%d)", rc)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic

        # Determine switch_id from topic or payload
        switch_id = extract_switch_id(topic) or payload.get("switch_id")
        if not switch_id:
            return

        token = get_token(switch_id)
        if not token:
            log.debug("No TB token for switch_id=%s, skipping", switch_id)
            return

        # Route to correct ThingsBoard API endpoint
        if "/attributes" in topic:
            url = TB_ATTRIBUTES_URL.format(token=token)
            endpoint = "attributes"
        else:
            url = TB_TELEMETRY_URL.format(token=token)
            endpoint = "telemetry"

        success = post_to_thingsboard(url, payload)
        if success:
            log.debug("→ TB [%s] %s (%d keys)", switch_id, endpoint, len(payload))
        else:
            log.warning("✗ TB [%s] %s failed", switch_id, endpoint)

    except json.JSONDecodeError:
        log.warning("Invalid JSON on topic %s", msg.topic)
    except Exception as e:
        log.error("Bridge error: %s", e)


def main():
    log.info("ThingsBoard Bridge starting...")
    log.info("Mosquitto: %s:%d", MQTT_HOST, MQTT_PORT)
    log.info("ThingsBoard: %s:%s", TB_HOST, TB_PORT)
    log.info("Token map: %d devices configured", len(TOKEN_MAP))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="tb-bridge")
    client.on_connect = on_connect
    client.on_message = on_message

    # Retry connection (ThingsBoard and Mosquitto may not be ready yet)
    for attempt in range(1, 31):
        try:
            client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            break
        except Exception:
            log.info("Waiting for Mosquitto... (attempt %d/30)", attempt)
            time.sleep(2)
    else:
        log.error("Could not connect to Mosquitto after 30 attempts")
        return

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        log.info("Bridge shutting down")
        client.disconnect()


if __name__ == "__main__":
    main()
