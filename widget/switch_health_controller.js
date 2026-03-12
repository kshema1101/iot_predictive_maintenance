/**
 * ThingsBoard Custom Widget Controller — Fleet Switch Health
 *
 * Handles multiple devices: each entity in the widget's datasource
 * becomes a card in the fleet grid. Supports RPC per device.
 *
 * Data keys required:
 *   health_index, motor_current, transition_time, vibration_peak,
 *   motor_temperature, status_message, maintenance_required,
 *   remaining_cycles, switch_id
 */

self.onInit = function () {
    var scope = self.ctx.$scope;

    scope.devices = [];
    scope.summary = { healthy: 0, degrading: 0, critical: 0 };

    var MAX_ARC = 235.6 * 0.75;  // 270° arc for mini gauge (r=50)

    // ── Helpers ──────────────────────────────────────────────────

    function getGaugeColor(h) {
        if (h > 80) return "#3fb950";
        if (h > 50) return "#d29922";
        return "#f85149";
    }

    function getStatusClass(h) {
        if (h > 80) return "optimal";
        if (h > 50) return "degrading";
        return "critical";
    }

    function computeGaugeDash(h) {
        var arc = (h / 100) * MAX_ARC;
        var gap = 235.6 - arc;
        return arc.toFixed(2) + " " + gap.toFixed(2);
    }

    // ── Build / update device list from datasource ──────────────

    function updateDevices(data) {
        if (!data || data.length === 0) return;

        // Group series by datasource index (one per entity/device)
        var deviceMap = {};

        data.forEach(function (series) {
            if (!series.data || series.data.length === 0) return;

            var dsIndex = series.datasource ? series.datasource.entityId : series.dsIndex;
            if (!dsIndex && dsIndex !== 0) dsIndex = "default";

            if (!deviceMap[dsIndex]) {
                deviceMap[dsIndex] = {
                    id: "—",
                    entityId: series.datasource ? series.datasource.entityId : null,
                    healthIndex: 100,
                    motorCurrent: "0.00",
                    transitionTime: "0.0",
                    vibrationPeak: "0.000",
                    motorTemp: "0.0",
                    statusMessage: "Waiting...",
                    maintenanceRequired: false,
                    remainingCycles: "—",
                    gaugeColor: "#3fb950",
                    gaugeDash: "0 235.6",
                    statusClass: "optimal"
                };
            }

            var dev = deviceMap[dsIndex];
            var val = series.data[series.data.length - 1][1];

            switch (series.dataKey.name) {
                case "health_index":       dev.healthIndex = Math.round(parseFloat(val)); break;
                case "motor_current":      dev.motorCurrent = parseFloat(val).toFixed(1); break;
                case "transition_time":    dev.transitionTime = parseFloat(val).toFixed(0); break;
                case "vibration_peak":     dev.vibrationPeak = parseFloat(val).toFixed(2); break;
                case "motor_temperature":  dev.motorTemp = parseFloat(val).toFixed(0); break;
                case "status_message":     dev.statusMessage = val; break;
                case "maintenance_required":
                    dev.maintenanceRequired = (val === true || val === "true");
                    break;
                case "remaining_cycles":   dev.remainingCycles = parseInt(val) || "—"; break;
                case "switch_id":          dev.id = val; break;
            }
        });

        // Compute visual props and build array
        var devArray = [];
        var summary = { healthy: 0, degrading: 0, critical: 0 };

        Object.keys(deviceMap).forEach(function (key) {
            var dev = deviceMap[key];
            var h = dev.healthIndex;

            dev.gaugeColor = getGaugeColor(h);
            dev.gaugeDash = computeGaugeDash(h);
            dev.statusClass = getStatusClass(h);

            if (h > 80) summary.healthy++;
            else if (h > 50) summary.degrading++;
            else summary.critical++;

            // Use device name from datasource if switch_id wasn't sent
            if (dev.id === "—" && deviceMap[key].entityId) {
                dev.id = key.substring(0, 8);
            }

            devArray.push(dev);
        });

        // Sort: critical first, then degrading, then healthy
        devArray.sort(function (a, b) {
            return a.healthIndex - b.healthIndex;
        });

        scope.devices = devArray;
        scope.summary = summary;
        self.ctx.detectChanges();
    }

    // ── RPC commands (per device) ────────────────────────────────

    scope.clearError = function (dev) {
        sendRpc(dev, "clearError", {});
    };

    scope.injectFailure = function (dev) {
        sendRpc(dev, "injectFailure", { mode: "mechanical_friction", duration_cycles: 25 });
    };

    function sendRpc(dev, method, params) {
        if (!self.ctx.controlApi) {
            console.warn("RPC not available");
            return;
        }

        var entityId = dev.entityId;
        if (entityId) {
            self.ctx.controlApi.sendOneWayCommand(method, params, null, entityId).then(
                function () { dev.statusMessage = method + " sent"; self.ctx.detectChanges(); },
                function (err) { console.error("RPC failed:", err); }
            );
        } else {
            self.ctx.controlApi.sendOneWayCommand(method, params).then(
                function () { dev.statusMessage = method + " sent"; self.ctx.detectChanges(); },
                function (err) { console.error("RPC failed:", err); }
            );
        }
    }

    // ── Data binding ─────────────────────────────────────────────

    self.onDataUpdated = function () {
        updateDevices(self.ctx.data);
    };

    updateDevices(self.ctx.data);
};

self.onDataUpdated = function () {
    if (self.ctx && self.ctx.$scope) {
        self.onInit && self.onDataUpdated && self.onDataUpdated();
    }
};

self.onDestroy = function () {};
