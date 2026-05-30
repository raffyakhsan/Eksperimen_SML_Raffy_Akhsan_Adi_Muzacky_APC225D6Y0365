"""
prometheus_exporter.py
========================
Custom Prometheus Exporter untuk monitoring model Machine Learning (Titanic).
Mengekspos berbagai metrics yang dapat dimonitor oleh Prometheus dan divisualisasikan di Grafana.

Metrics yang diekspos (10+ untuk Advanced):
1.  http_requests_total          - Total HTTP requests (counter)
2.  http_request_duration_seconds - Durasi request (histogram)
3.  http_request_errors_total    - Total request errors (counter)
4.  system_cpu_usage             - CPU usage (gauge)
5.  system_ram_usage             - RAM usage dalam MB (gauge)
6.  system_ram_usage_percent     - RAM usage dalam persen (gauge)
7.  model_prediction_total       - Total prediksi (counter)
8.  model_prediction_latency     - Latency prediksi dalam ms (gauge)
9.  model_prediction_confidence  - Confidence rata-rata prediksi (gauge)
10. model_survived_count         - Total prediksi Survived=1 (counter)
11. model_not_survived_count     - Total prediksi Not Survived=0 (counter)
12. model_last_prediction_time   - Timestamp prediksi terakhir (gauge)

Penggunaan:
    python prometheus_exporter.py
    # Akses metrics: http://localhost:8000/metrics
    # Akses predict: POST http://localhost:8000/predict
"""

import time
import os
import json
import random
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import psutil
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server, REGISTRY
)
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# PROMETHEUS METRICS DEFINITIONS
# ============================================================

# Metric 1: Total HTTP requests
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests received',
    ['method', 'endpoint', 'status_code']
)

# Metric 2: HTTP request duration
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Metric 3: HTTP errors
http_request_errors_total = Counter(
    'http_request_errors_total',
    'Total number of HTTP errors',
    ['method', 'endpoint', 'error_type']
)

# Metric 4: CPU usage
system_cpu_usage = Gauge(
    'system_cpu_usage',
    'System CPU usage percentage',
    ['instance', 'job']
)

# Metric 5: RAM usage in MB
system_ram_usage = Gauge(
    'system_ram_usage',
    'System RAM usage in megabytes',
    ['instance', 'job']
)

# Metric 6: RAM usage percentage
system_ram_usage_percent = Gauge(
    'system_ram_usage_percent',
    'System RAM usage percentage',
    ['instance', 'job']
)

# Metric 7: Total model predictions
model_prediction_total = Counter(
    'model_prediction_total',
    'Total number of model predictions made',
    ['model_name', 'version']
)

# Metric 8: Prediction latency
model_prediction_latency = Gauge(
    'model_prediction_latency',
    'Model prediction latency in milliseconds',
    ['model_name']
)

# Metric 9: Average prediction confidence
model_prediction_confidence = Gauge(
    'model_prediction_confidence',
    'Average prediction confidence score',
    ['model_name', 'prediction_class']
)

# Metric 10: Survived predictions count
model_survived_count = Counter(
    'model_survived_count',
    'Total predictions where Survived = 1',
    ['model_name']
)

# Metric 11: Not survived predictions count
model_not_survived_count = Counter(
    'model_not_survived_count',
    'Total predictions where Survived = 0',
    ['model_name']
)

# Metric 12: Last prediction timestamp
model_last_prediction_time = Gauge(
    'model_last_prediction_time',
    'Unix timestamp of the last prediction',
    ['model_name']
)

# Additional metric: API uptime
api_uptime_seconds = Gauge(
    'api_uptime_seconds',
    'API server uptime in seconds'
)

# Track start time for uptime
_start_time = time.time()

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Titanic ML Model API with Prometheus Monitoring",
    description="REST API untuk serving model Titanic dengan Prometheus metrics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_NAME = "titanic-random-forest"
MODEL_VERSION = "1"


class PredictRequest(BaseModel):
    """Schema input untuk prediksi."""
    Pclass: int = 3
    Sex: int = 1           # 0=female, 1=male
    Age: float = 0.0       # sudah dinormalisasi
    SibSp: float = 0.0
    Parch: float = 0.0
    Fare: float = 0.0
    Embarked: int = 2      # 0=C, 1=Q, 2=S
    FamilySize: float = 0.0
    IsAlone: int = 1
    AgeGroup: int = 3
    FareBin: int = 1


class PredictResponse(BaseModel):
    """Schema output untuk prediksi."""
    prediction: int
    prediction_label: str
    confidence: float
    model_name: str
    model_version: str
    inference_time_ms: float
    timestamp: str


def update_system_metrics():
    """Update system metrics secara periodik."""
    instance_label = "127.0.0.1:8000"
    job_label = "ml_model_exporter"
    
    while True:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.labels(
                instance=instance_label,
                job=job_label
            ).set(cpu_percent)
            
            # RAM usage
            memory = psutil.virtual_memory()
            ram_mb = memory.used / (1024 * 1024)
            system_ram_usage.labels(
                instance=instance_label,
                job=job_label
            ).set(ram_mb)
            
            system_ram_usage_percent.labels(
                instance=instance_label,
                job=job_label
            ).set(memory.percent)
            
            # Uptime
            api_uptime_seconds.set(time.time() - _start_time)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
        
        time.sleep(10)


def dummy_model_predict(features: dict):
    """
    Dummy model prediction (simulasi saat model artifact tidak tersedia).
    Dalam produksi, ganti ini dengan mlflow.pyfunc.load_model().
    """
    # Simulasi prediksi berdasarkan fitur utama
    survival_prob = 0.5
    
    # Logika sederhana berdasarkan fitur
    if features.get('Sex', 1) == 0:  # female
        survival_prob += 0.25
    if features.get('Pclass', 3) == 1:  # first class
        survival_prob += 0.15
    elif features.get('Pclass', 3) == 3:  # third class
        survival_prob -= 0.15
    if features.get('Age', 0) < -0.5:  # anak-anak (usia rendah setelah normalisasi)
        survival_prob += 0.10
    
    # Clip ke [0.1, 0.9]
    survival_prob = max(0.1, min(0.9, survival_prob))
    
    # Tambah sedikit noise untuk simulasi
    survival_prob += random.uniform(-0.05, 0.05)
    survival_prob = max(0.05, min(0.95, survival_prob))
    
    prediction = 1 if survival_prob > 0.5 else 0
    return prediction, survival_prob


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware untuk tracking HTTP metrics."""
    start_time = time.time()
    endpoint = request.url.path
    method = request.method
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        status_code = str(response.status_code)
        
        # Update metrics
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        http_request_errors_total.labels(
            method=method,
            endpoint=endpoint,
            error_type=type(e).__name__
        ).inc()
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - _start_time, 2),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    memory = psutil.virtual_memory()
    return {
        "status": "healthy",
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": memory.percent,
        "uptime_seconds": round(time.time() - _start_time, 2)
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Endpoint untuk prediksi survival Titanic."""
    start_time = time.time()
    
    try:
        features = request.dict()
        
        # Lakukan prediksi
        prediction, confidence = dummy_model_predict(features)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Update prediction metrics
        model_prediction_total.labels(
            model_name=MODEL_NAME,
            version=MODEL_VERSION
        ).inc()
        
        model_prediction_latency.labels(
            model_name=MODEL_NAME
        ).set(inference_time_ms)
        
        model_prediction_confidence.labels(
            model_name=MODEL_NAME,
            prediction_class=str(prediction)
        ).set(confidence)
        
        model_last_prediction_time.labels(
            model_name=MODEL_NAME
        ).set(time.time())
        
        if prediction == 1:
            model_survived_count.labels(model_name=MODEL_NAME).inc()
        else:
            model_not_survived_count.labels(model_name=MODEL_NAME).inc()
        
        prediction_label = "Survived" if prediction == 1 else "Not Survived"
        
        return PredictResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=round(confidence, 4),
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inference_time_ms=round(inference_time_ms, 4),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/simulate")
async def simulate_traffic(n: int = 100):
    """Simulasi traffic untuk testing metrics (gunakan di development)."""
    import asyncio
    results = {"survived": 0, "not_survived": 0, "total": 0}
    
    for _ in range(min(n, 500)):  # max 500 simulasi
        # Random features
        req = PredictRequest(
            Pclass=random.randint(1, 3),
            Sex=random.randint(0, 1),
            Age=random.uniform(-2, 2),
            SibSp=random.uniform(-1, 2),
            Parch=random.uniform(-1, 2),
            Fare=random.uniform(-1, 3),
            Embarked=random.randint(0, 2),
            FamilySize=random.uniform(-1, 3),
            IsAlone=random.randint(0, 1),
            AgeGroup=random.randint(0, 4),
            FareBin=random.randint(0, 3)
        )
        
        prediction, confidence = dummy_model_predict(req.dict())
        
        # Update metrics langsung
        model_prediction_total.labels(model_name=MODEL_NAME, version=MODEL_VERSION).inc()
        model_prediction_latency.labels(model_name=MODEL_NAME).set(random.uniform(1, 50))
        model_last_prediction_time.labels(model_name=MODEL_NAME).set(time.time())
        
        if prediction == 1:
            model_survived_count.labels(model_name=MODEL_NAME).inc()
            results["survived"] += 1
        else:
            model_not_survived_count.labels(model_name=MODEL_NAME).inc()
            results["not_survived"] += 1
        results["total"] += 1
    
    return {
        "message": f"Simulated {results['total']} predictions",
        "results": results
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Menjalankan server FastAPI."""
    logger.info(f"Memulai ML Model API Server...")
    logger.info(f"Metrics endpoint: http://localhost:{port}/metrics")
    logger.info(f"Predict endpoint: POST http://localhost:{port}/predict")
    logger.info(f"Simulate traffic: GET http://localhost:{port}/simulate?n=100")
    
    # Jalankan thread update system metrics
    system_thread = threading.Thread(target=update_system_metrics, daemon=True)
    system_thread.start()
    logger.info("System metrics update thread started")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prometheus Exporter untuk ML Model')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    
    start_server(host=args.host, port=args.port)
