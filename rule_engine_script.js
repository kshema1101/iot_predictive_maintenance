/**
 * ThingsBoard Rule Engine – Transformation Script Node (TBEL)
 * 
 * Processes incoming telemetry from any railway switch in the fleet.
 * Computes health_index, flags maintenance, and classifies severity.
 *
 * Installation:
 *   1. Rule Chains -> Root Rule Chain
 *   2. Add "Transformation Script" node
 *   3. Wire: Message Type Switch ->(Post telemetry)-> Health Scorer ->(Success)-> Save Timeseries
 *   4. Paste this code into the script body (Language: TBEL)
 */

var mc = msg.motor_current;

if (mc == null) {
    return {msg: msg, metadata: metadata, msgType: msgType};
}

var transitionTime  = parseFloat(msg.transition_time);
var motorCurrent    = parseFloat(msg.motor_current);
var vibrationPeak   = parseFloat(msg.vibration_peak);
var motorTemp       = parseFloat(msg.motor_temperature);
var supplyVoltage   = parseFloat(msg.supply_voltage);
var cycleCount      = parseFloat(msg.cycle_count);

if (transitionTime == null) { transitionTime = 0; }
if (motorCurrent == null) { motorCurrent = 0; }
if (vibrationPeak == null) { vibrationPeak = 0; }
if (motorTemp == null) { motorTemp = 0; }
if (supplyVoltage == null) { supplyVoltage = 24; }
if (cycleCount == null) { cycleCount = 1; }

var healthIndex = 100.0;

if (transitionTime > 3000) {
    healthIndex = healthIndex - ((transitionTime - 3000) / 500).intValue() * 10;
}
if (motorCurrent > 5.0) {
    healthIndex = healthIndex - ((motorCurrent - 5.0) * 8).intValue();
}
if (vibrationPeak > 1.2) {
    healthIndex = healthIndex - ((vibrationPeak - 1.2) / 0.5 * 5).intValue();
}
if (motorTemp > 50) {
    healthIndex = healthIndex - ((motorTemp - 50) / 10 * 5).intValue();
}
if (supplyVoltage < 20.0) {
    healthIndex = healthIndex - ((20.0 - supplyVoltage) * 10).intValue();
}

if (healthIndex < 0) { healthIndex = 0; }
if (healthIndex > 100) { healthIndex = 100; }

var maintenanceRequired = transitionTime > 4500 || motorCurrent > 7.0 || vibrationPeak > 2.0 || motorTemp > 65.0 || supplyVoltage < 20.0;

var statusMessage = "Optimal";
var severityLevel = "NORMAL";

if (healthIndex <= 20) {
    statusMessage = "Critical Failure - Inoperable";
    severityLevel = "CRITICAL";
} else if (healthIndex <= 40) {
    statusMessage = "Critical - Immediate Action";
    severityLevel = "MAJOR";
} else if (healthIndex <= 60) {
    statusMessage = "Poor - Maintenance Required";
    severityLevel = "MINOR";
} else if (healthIndex <= 80) {
    statusMessage = "Degrading - Schedule Inspection";
    severityLevel = "WARNING";
}

msg.health_index = healthIndex;
msg.status_message = statusMessage;
msg.severity_level = severityLevel;
msg.maintenance_required = maintenanceRequired;

return {msg: msg, metadata: metadata, msgType: msgType};
