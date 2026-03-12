"""
IoT Predictive Maintenance Simulator — Railway Switch Fleet

Simulates a fleet of railway point machines publishing telemetry via MQTT.
Failure injection is scenario-driven and deterministic: degradation is a
pure function of elapsed cycles, not a background thread mutating shared state.

Three operating modes:
  --offline           No MQTT broker needed. Runs fleet + prints telemetry locally.
  (default)           Generic MQTT (Mosquitto). Topic: iot/switches/<id>/telemetry
  --thingsboard       ThingsBoard MQTT API. Topic: v1/devices/me/telemetry

Usage:
  python simulator.py --offline                # No broker, just terminal output
  python simulator.py                          # Mosquitto on localhost:1883
  python simulator.py --thingsboard            # ThingsBoard mode
  python simulator.py --switches 10            # Auto-generate 10-switch fleet
"""

import json
import math
import time
import random
import logging
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# ── MQTT Topic Profiles ─────────────────────────────────────────
TOPIC_PROFILES = {
    "generic": {
        "telemetry":    "iot/switches/{switch_id}/telemetry",
        "attributes":   "iot/switches/{switch_id}/attributes",
        "rpc_response": "iot/switches/{switch_id}/rpc/response/",
        "rpc_request":  "iot/switches/{switch_id}/rpc/request/+",
    },
    "thingsboard": {
        "telemetry":    "v1/devices/me/telemetry",
        "attributes":   "v1/devices/me/attributes",
        "rpc_response": "v1/devices/me/rpc/response/",
        "rpc_request":  "v1/devices/me/rpc/request/+",
    },
}

# ── Normal Operating Baselines ───────────────────────────────────
BASELINE = {
    "motor_current":    (3.0, 5.0),
    "transition_time":  (2000, 3000),
    "vibration_peak":   (0.5, 1.2),
    "supply_voltage":   (22.0, 26.0),
    "motor_temperature": (15.0, 45.0),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Degradation Profiles — Pure functions, no threads, no mutation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DegradationProfile:
    """
    Defines how a switch degrades over time. The factors are computed
    from elapsed_cycles alone — making them deterministic and testable.

    friction_factor multiplies motor_current.
    blockage_factor multiplies transition_time.
    """
    mode: str
    start_after_cycles: int = 10
    duration_cycles: int = 30

    def compute(self, elapsed_cycles: int) -> dict:
        """Return degradation multipliers for the given cycle count."""
        if elapsed_cycles < self.start_after_cycles:
            return {"friction": 1.0, "blockage": 1.0, "phase": "healthy"}

        cycles_into_failure = elapsed_cycles - self.start_after_cycles
        progress = min(cycles_into_failure / max(self.duration_cycles, 1), 1.0)

        # S-curve (logistic) for realistic gradual-then-accelerating degradation
        steepness = 8
        midpoint = 0.5
        s_progress = 1.0 / (1.0 + math.exp(-steepness * (progress - midpoint)))

        if self.mode == "mechanical_friction":
            friction = 1.0 + s_progress * 0.9     # current: up to ~1.9x → ~8A peak
            blockage = 1.0 + s_progress * 1.1     # time:    up to ~2.1x → ~6300ms peak
        elif self.mode == "blockage":
            friction = 1.0 + s_progress * 0.3
            blockage = 1.0 + s_progress * 1.6     # severe slowdown
        elif self.mode == "electrical":
            friction = 1.0 + s_progress * 1.3     # erratic current spikes
            blockage = 1.0 + s_progress * 0.4
        elif self.mode == "bearing_wear":
            friction = 1.0 + s_progress * 0.6
            blockage = 1.0 + s_progress * 0.7
        else:
            friction = 1.0
            blockage = 1.0

        if progress >= 1.0:
            phase = "failed"
        elif progress > 0.5:
            phase = "critical"
        elif progress > 0.0:
            phase = "degrading"
        else:
            phase = "healthy"

        return {"friction": friction, "blockage": blockage, "phase": phase}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Single Switch — owns its MQTT client and telemetry state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PointMachine:
    """One railway point machine with its own MQTT connection."""

    def __init__(self, switch_id: str, token: str, host: str, port: int,
                 location: str = "", model: str = "S700K",
                 manufacturer: str = "Siemens Mobility",
                 degradation: Optional[DegradationProfile] = None,
                 topic_profile: str = "generic",
                 offline: bool = False):
        self.switch_id = switch_id
        self.token = token
        self.host = host
        self.port = port
        self.location = location
        self.model = model
        self.manufacturer = manufacturer
        self.degradation = degradation
        self.offline = offline

        # Resolve MQTT topics
        topics = TOPIC_PROFILES.get(topic_profile, TOPIC_PROFILES["generic"])
        self.topic_telemetry = topics["telemetry"].format(switch_id=switch_id)
        self.topic_attributes = topics["attributes"].format(switch_id=switch_id)
        self.topic_rpc_response = topics["rpc_response"].format(switch_id=switch_id)
        self.topic_rpc_request = topics["rpc_request"].format(switch_id=switch_id)

        self.log = logging.getLogger(switch_id)
        self.cycle_count = 0
        self.connected = False

        # RPC-injectable failure (on top of scenario-based)
        self._rpc_degradation: Optional[DegradationProfile] = None
        self._rpc_inject_cycle: int = 0

        self.client = None
        if not offline and MQTT_AVAILABLE:
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"sim-{switch_id}",
            )
            if token:
                self.client.username_pw_set(token)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message

    # ── MQTT lifecycle ────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.connected = True
            self.log.info("Connected to MQTT broker")
            client.subscribe(self.topic_rpc_request)
            self._publish_attributes()
        else:
            self.log.error("MQTT connect failed (rc=%d)", rc)

    def connect(self):
        if self.offline or not self.client:
            self.connected = False
            return
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            self.log.error("Cannot connect: %s", e)

    def disconnect(self):
        if self.client and self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.log.info("Disconnected")

    # ── RPC handling ──────────────────────────────────────────────

    def _on_message(self, client, userdata, msg):
        try:
            request_id = msg.topic.split("/")[-1]
            payload = json.loads(msg.payload.decode())
            method = payload.get("method", "")
            params = payload.get("params", {})
            self.log.info("RPC ← %s(%s)", method, params)

            response = self._handle_rpc(method, params)
            client.publish(self.topic_rpc_response + request_id, json.dumps(response))
        except Exception as e:
            self.log.error("RPC error: %s", e)

    def _handle_rpc(self, method: str, params: dict) -> dict:
        if method == "clearError":
            self._rpc_degradation = None
            self.log.info("RPC reset — cleared injected failure")
            return {"success": True, "switch_id": self.switch_id}

        if method == "injectFailure":
            mode = params.get("mode", "mechanical_friction")
            duration = params.get("duration_cycles", 25)
            self._rpc_degradation = DegradationProfile(
                mode=mode, start_after_cycles=0, duration_cycles=duration,
            )
            self._rpc_inject_cycle = self.cycle_count
            self.log.warning("RPC inject — mode=%s duration=%d cycles", mode, duration)
            return {"success": True, "switch_id": self.switch_id, "mode": mode}

        if method == "getStatus":
            return {
                "switch_id": self.switch_id,
                "cycle_count": self.cycle_count,
                "scenario_failure": self.degradation.mode if self.degradation else None,
                "rpc_failure": self._rpc_degradation.mode if self._rpc_degradation else None,
            }

        return {"error": f"Unknown method: {method}"}

    # ── Telemetry generation ──────────────────────────────────────

    def _get_effective_degradation(self) -> dict:
        """Combine scenario-based and RPC-injected degradation."""
        result = {"friction": 1.0, "blockage": 1.0, "phase": "healthy"}

        if self.degradation:
            d = self.degradation.compute(self.cycle_count)
            result["friction"] = max(result["friction"], d["friction"])
            result["blockage"] = max(result["blockage"], d["blockage"])
            if d["phase"] != "healthy":
                result["phase"] = d["phase"]

        if self._rpc_degradation:
            rpc_elapsed = self.cycle_count - self._rpc_inject_cycle
            d = self._rpc_degradation.compute(rpc_elapsed)
            result["friction"] = max(result["friction"], d["friction"])
            result["blockage"] = max(result["blockage"], d["blockage"])
            if d["phase"] != "healthy":
                result["phase"] = d["phase"]

        return result

    def generate_telemetry(self) -> dict:
        self.cycle_count += 1
        deg = self._get_effective_degradation()

        # Add realistic sensor noise (±2%)
        noise = lambda: random.uniform(0.98, 1.02)

        motor_current = round(
            random.uniform(*BASELINE["motor_current"]) * deg["friction"] * noise(), 2
        )
        transition_time = round(
            random.uniform(*BASELINE["transition_time"]) * deg["blockage"] * noise(), 1
        )
        vibration_peak = round(
            random.uniform(*BASELINE["vibration_peak"])
            * ((deg["friction"] + deg["blockage"]) / 2) * noise(), 3
        )
        supply_voltage = round(
            random.uniform(*BASELINE["supply_voltage"]) - (motor_current - 4.0) * 0.3, 2
        )
        motor_temperature = round(
            random.uniform(*BASELINE["motor_temperature"]) + (motor_current - 4.0) * 5.0, 1
        )

        return {
            "switch_id": self.switch_id,
            "motor_current": motor_current,
            "transition_time": transition_time,
            "vibration_peak": vibration_peak,
            "supply_voltage": supply_voltage,
            "motor_temperature": motor_temperature,
            "switch_completed": transition_time < 4500,
            "cycle_count": self.cycle_count,
            "degradation_phase": deg["phase"],
            "failure_active": deg["phase"] not in ("healthy",),
            "ts": int(datetime.now().timestamp() * 1000),
        }

    def publish_telemetry(self) -> dict:
        payload = self.generate_telemetry()
        if self.connected and self.client:
            self.client.publish(self.topic_telemetry, json.dumps(payload))
        return payload

    # ── Attributes ────────────────────────────────────────────────

    def _publish_attributes(self):
        attrs = {
            "device_type": "Railway Point Machine",
            "switch_id": self.switch_id,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "firmware_version": "3.2.1",
            "location": self.location,
            "installation_date": "2023-06-15",
            "max_rated_current": 10.0,
            "max_transition_time": 8000,
        }
        if self.client:
            self.client.publish(self.topic_attributes, json.dumps(attrs))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fleet Orchestrator — manages multiple switches
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FleetSimulator:
    """Orchestrates telemetry publishing for a fleet of point machines."""

    def __init__(self, switches: list[PointMachine], interval: int = 5):
        self.switches = {sw.switch_id: sw for sw in switches}
        self.interval = interval
        self.log = logging.getLogger("Fleet")

    def connect_all(self):
        for sw in self.switches.values():
            sw.connect()
            time.sleep(0.1)  # stagger connections to avoid broker overload
        self.log.info("Fleet connected: %d switches", len(self.switches))

    def disconnect_all(self):
        for sw in self.switches.values():
            sw.disconnect()
        self.log.info("Fleet disconnected")

    def run(self):
        """Main loop — publishes telemetry for all switches each interval."""
        self.connect_all()
        self.log.info(
            "Publishing telemetry every %ds for %d switches",
            self.interval, len(self.switches),
        )

        cycle = 0
        try:
            while True:
                cycle += 1
                print(f"\n{'═' * 70}")
                print(f"  FLEET CYCLE #{cycle}  —  {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'═' * 70}")

                for sw in self.switches.values():
                    payload = sw.publish_telemetry()
                    status_icon = {
                        "healthy": "●",
                        "degrading": "▲",
                        "critical": "■",
                        "failed": "✖",
                    }.get(payload["degradation_phase"], "?")

                    color_code = {
                        "healthy": "\033[92m",
                        "degrading": "\033[93m",
                        "critical": "\033[91m",
                        "failed": "\033[31m",
                    }.get(payload["degradation_phase"], "")
                    reset = "\033[0m"

                    print(
                        f"  {color_code}{status_icon}{reset} [{payload['switch_id']}] "
                        f"current={payload['motor_current']:.1f}A  "
                        f"time={payload['transition_time']:.0f}ms  "
                        f"vib={payload['vibration_peak']:.2f}g  "
                        f"temp={payload['motor_temperature']:.0f}°C  "
                        f"{color_code}{payload['degradation_phase']}{reset}"
                    )

                time.sleep(self.interval)

        except KeyboardInterrupt:
            self.log.info("Shutting down fleet...")
        finally:
            self.disconnect_all()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Config Loading & Auto-Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FAILURE_MODES = ["mechanical_friction", "blockage", "electrical", "bearing_wear"]
LOCATIONS = [
    "Platform 1 Junction", "North Yard Entry", "Depot Throat",
    "South Crossover", "East Bypass", "West Siding",
    "Freight Terminal", "Passenger Loop", "Maintenance Spur",
    "Express Divert", "Slow Line Merge", "Signal Gantry Junction",
]
MODELS = [
    ("S700K", "Siemens Mobility"),
    ("HW1000", "Alstom"),
    ("L826H", "Voestalpine"),
    ("Clamplock", "Network Rail"),
]


def load_from_config(path: str, host_override: str = None,
                     port_override: int = None,
                     topic_profile: str = "generic",
                     offline: bool = False) -> FleetSimulator:
    """Build fleet from a JSON config file."""
    cfg = json.loads(Path(path).read_text())
    broker = cfg.get("thingsboard", cfg.get("broker", {}))
    host = host_override or broker.get("host", "localhost")
    port = port_override or broker.get("port", 1883)
    interval = cfg.get("publish_interval_sec", 5)

    machines = []
    for sw_cfg in cfg["switches"]:
        deg = None
        if sw_cfg.get("failure_scenario"):
            fs = sw_cfg["failure_scenario"]
            deg = DegradationProfile(
                mode=fs["mode"],
                start_after_cycles=fs.get("start_after_cycles", 10),
                duration_cycles=fs.get("duration_cycles", 30),
            )

        machines.append(PointMachine(
            switch_id=sw_cfg["id"],
            token=sw_cfg.get("token", ""),
            host=host,
            port=port,
            location=sw_cfg.get("location", ""),
            model=sw_cfg.get("model", "S700K"),
            manufacturer=sw_cfg.get("manufacturer", "Siemens Mobility"),
            degradation=deg,
            topic_profile=topic_profile,
            offline=offline,
        ))

    return FleetSimulator(machines, interval)


def generate_fleet(count: int, host: str, port: int,
                   failure_ratio: float = 0.4,
                   topic_profile: str = "generic",
                   offline: bool = False) -> FleetSimulator:
    """Auto-generate a fleet of N switches with random failure assignments."""
    machines = []
    failing_count = max(1, int(count * failure_ratio))
    failing_indices = set(random.sample(range(count), failing_count))

    for i in range(count):
        sw_id = f"SW-{i + 1:03d}"
        model, mfr = random.choice(MODELS)

        deg = None
        if i in failing_indices:
            deg = DegradationProfile(
                mode=random.choice(FAILURE_MODES),
                start_after_cycles=random.randint(5, 25),
                duration_cycles=random.randint(15, 40),
            )

        machines.append(PointMachine(
            switch_id=sw_id,
            token=f"RAILWAY_SWITCH_{i + 1:02d}",
            host=host,
            port=port,
            location=random.choice(LOCATIONS),
            model=model,
            manufacturer=mfr,
            degradation=deg,
            topic_profile=topic_profile,
            offline=offline,
        ))

    return FleetSimulator(machines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Railway Point Machine Fleet Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulator.py --offline                          # No broker needed
  python simulator.py                                    # Mosquitto on localhost
  python simulator.py --thingsboard                      # ThingsBoard mode
  python simulator.py --switches 10                      # auto-generate 10 switches
  python simulator.py --switches 20 --failure-ratio 0.6  # 60%% will degrade
        """,
    )
    parser.add_argument("--config", default="switch_config.json",
                        help="Path to fleet config JSON (default: switch_config.json)")
    parser.add_argument("--host", default=None, help="Override MQTT broker host")
    parser.add_argument("--port", type=int, default=None, help="Override MQTT port")
    parser.add_argument("--switches", type=int, default=None,
                        help="Auto-generate N switches instead of using config file")
    parser.add_argument("--failure-ratio", type=float, default=0.4,
                        help="Fraction of auto-generated switches that will degrade (default: 0.4)")
    parser.add_argument("--offline", action="store_true",
                        help="Run without any MQTT broker (terminal output only)")
    parser.add_argument("--thingsboard", action="store_true",
                        help="Use ThingsBoard MQTT topic format (default: generic Mosquitto topics)")
    args = parser.parse_args()

    topic_profile = "thingsboard" if args.thingsboard else "generic"

    if not args.offline and not MQTT_AVAILABLE:
        logging.warning("paho-mqtt not installed. Running in offline mode.")
        logging.info("Install with: pip install paho-mqtt")
        args.offline = True

    if args.switches:
        host = args.host or "localhost"
        port = args.port or 1883
        fleet = generate_fleet(
            args.switches, host, port, args.failure_ratio,
            topic_profile=topic_profile, offline=args.offline,
        )
        logging.getLogger("Fleet").info(
            "Auto-generated %d switches (%.0f%% failing)%s",
            args.switches, args.failure_ratio * 100,
            " [OFFLINE]" if args.offline else "",
        )
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            logging.error("Config file not found: %s", config_path)
            logging.info("Use --switches N to auto-generate a fleet instead")
            return
        fleet = load_from_config(
            str(config_path), args.host, args.port,
            topic_profile=topic_profile, offline=args.offline,
        )

    fleet.run()


if __name__ == "__main__":
    main()
