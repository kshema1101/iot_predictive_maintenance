FROM python:3.11-slim

WORKDIR /app

COPY requirements-ml.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./ml/
COPY simulator.py .
COPY switch_config.json .

# The ML predictor enriches telemetry via ThingsBoard REST API.
# It polls each device's latest telemetry, runs ML inference,
# and pushes predictions back as additional telemetry keys.
CMD ["python", "-u", "ml/predictor.py", "--thingsboard-api", "--tb-url", "http://thingsboard:9090"]
