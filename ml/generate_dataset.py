"""
Dataset Generator for Predictive Maintenance ML Models

Uses the simulator's DegradationProfile to generate thousands of labeled
telemetry samples across all failure modes. Each sample is tagged with:
  - failure_mode (classification label)
  - remaining_useful_life (regression target)
  - degradation_progress (0.0 = healthy, 1.0 = failed)

This runs offline — no MQTT needed. It imports directly from the simulator.

Usage:
  python ml/generate_dataset.py                     # default 50 switches, 80 cycles each
  python ml/generate_dataset.py --switches 200 --cycles 120
"""

import sys
import csv
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulator import PointMachine, DegradationProfile, FAILURE_MODES, BASELINE

OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def generate_switch_history(switch_id: str, mode: str | None,
                            start_after: int, duration: int,
                            total_cycles: int) -> list[dict]:
    """Run a switch through its full lifecycle and collect labeled telemetry."""
    deg = None
    if mode:
        deg = DegradationProfile(
            mode=mode,
            start_after_cycles=start_after,
            duration_cycles=duration,
        )

    machine = PointMachine(
        switch_id=switch_id, token="", host="", port=0, degradation=deg,
    )

    rows = []
    for _ in range(total_cycles):
        telemetry = machine.generate_telemetry()

        if deg:
            d = deg.compute(machine.cycle_count)
            cycles_into_failure = max(0, machine.cycle_count - start_after)
            progress = min(cycles_into_failure / max(duration, 1), 1.0)
            rul = max(0, (start_after + duration) - machine.cycle_count)
        else:
            d = {"friction": 1.0, "blockage": 1.0, "phase": "healthy"}
            progress = 0.0
            rul = total_cycles - machine.cycle_count

        rows.append({
            "switch_id": switch_id,
            "cycle": machine.cycle_count,
            "motor_current": telemetry["motor_current"],
            "transition_time": telemetry["transition_time"],
            "vibration_peak": telemetry["vibration_peak"],
            "supply_voltage": telemetry["supply_voltage"],
            "motor_temperature": telemetry["motor_temperature"],
            # Engineered features
            "current_x_time": round(telemetry["motor_current"] * telemetry["transition_time"], 2),
            "vibration_x_current": round(telemetry["vibration_peak"] * telemetry["motor_current"], 3),
            "power_draw": round(telemetry["motor_current"] * telemetry["supply_voltage"], 2),
            # Labels
            "failure_mode": mode or "healthy",
            "degradation_progress": round(progress, 4),
            "remaining_useful_life": rul,
            "phase": d["phase"],
        })

    return rows


def generate_full_dataset(num_switches: int = 50, cycles_per_switch: int = 80) -> list[dict]:
    """Generate a balanced dataset across all failure modes + healthy switches."""
    all_rows = []
    healthy_count = max(1, num_switches // 5)
    failing_count = num_switches - healthy_count

    modes_cycle = FAILURE_MODES * ((failing_count // len(FAILURE_MODES)) + 1)

    # Healthy switches
    for i in range(healthy_count):
        rows = generate_switch_history(
            switch_id=f"GEN-H-{i+1:03d}",
            mode=None,
            start_after=0, duration=0,
            total_cycles=cycles_per_switch,
        )
        all_rows.extend(rows)

    # Failing switches (varied onset and duration for diversity)
    for i in range(failing_count):
        mode = modes_cycle[i]
        start = random.randint(5, cycles_per_switch // 3)
        duration = random.randint(cycles_per_switch // 4, cycles_per_switch // 2)

        rows = generate_switch_history(
            switch_id=f"GEN-F-{i+1:03d}",
            mode=mode,
            start_after=start, duration=duration,
            total_cycles=cycles_per_switch,
        )
        all_rows.extend(rows)

    random.shuffle(all_rows)
    return all_rows


def save_csv(rows: list[dict], filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples → {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate ML training dataset")
    parser.add_argument("--switches", type=int, default=50, help="Number of virtual switches")
    parser.add_argument("--cycles", type=int, default=80, help="Cycles per switch")
    parser.add_argument("--output", default="training_data.csv", help="Output filename")
    args = parser.parse_args()

    print(f"Generating dataset: {args.switches} switches × {args.cycles} cycles...")
    rows = generate_full_dataset(args.switches, args.cycles)

    mode_counts = {}
    for r in rows:
        mode_counts[r["failure_mode"]] = mode_counts.get(r["failure_mode"], 0) + 1

    print("\nClass distribution:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode:25s} → {count:6d} samples")

    save_csv(rows, args.output)


if __name__ == "__main__":
    main()
