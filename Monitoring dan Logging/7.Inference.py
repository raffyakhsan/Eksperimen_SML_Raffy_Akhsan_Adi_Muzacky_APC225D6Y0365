"""
inference.py (7.Inference.py)
==============================
Script untuk melakukan inferensi menggunakan model MLflow yang sudah di-deploy.
Mendukung single prediction, batch prediction, dan benchmark testing.

Penggunaan:
    # Single prediction
    python 7.Inference.py --mode single

    # Batch prediction dari CSV
    python 7.Inference.py --mode batch --input titanic_preprocessing.csv

    # Benchmark (stress test untuk monitoring)
    python 7.Inference.py --mode benchmark --n 1000

    # Via API endpoint
    python 7.Inference.py --mode api --url http://localhost:8000/predict
"""

import requests
import json
import time
import argparse
import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Default API URL
DEFAULT_API_URL = "http://localhost:8000/predict"
DEFAULT_MLFLOW_URI = "http://127.0.0.1:5002"


def predict_via_api(features: Dict, api_url: str = DEFAULT_API_URL) -> Dict:
    """
    Melakukan prediksi melalui REST API endpoint.
    
    Args:
        features: Dictionary berisi fitur input
        api_url: URL endpoint prediksi
        
    Returns:
        Dictionary berisi hasil prediksi
    """
    try:
        start_time = time.time()
        response = requests.post(
            api_url,
            json=features,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        latency_ms = (time.time() - start_time) * 1000
        
        result = response.json()
        result["client_latency_ms"] = round(latency_ms, 3)
        return result
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Tidak dapat terhubung ke {api_url}. Pastikan server berjalan!")
        return {"error": "Connection refused", "api_url": api_url}
    except Exception as e:
        logger.error(f"Error saat prediksi: {e}")
        return {"error": str(e)}


def predict_via_mlflow(features_df: pd.DataFrame, model_uri: str) -> List:
    """
    Melakukan prediksi menggunakan MLflow model langsung.
    
    Args:
        features_df: DataFrame berisi fitur input
        model_uri: URI model MLflow (mis: 'models:/titanic-random-forest/latest')
        
    Returns:
        List berisi hasil prediksi
    """
    try:
        import mlflow.pyfunc
        logger.info(f"Memuat model dari: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        predictions = model.predict(features_df)
        logger.info(f"Prediksi berhasil: {len(predictions)} sampel")
        return predictions.tolist()
    except Exception as e:
        logger.error(f"Error memuat/menggunakan MLflow model: {e}")
        return []


def single_prediction_demo(api_url: str = DEFAULT_API_URL):
    """Demo single prediction dengan contoh data."""
    logger.info("=== DEMO SINGLE PREDICTION ===")
    
    # Contoh 1: Penumpang wanita kelas 1 (kemungkinan selamat tinggi)
    passenger_1 = {
        "Pclass": 1,
        "Sex": 0,         # female
        "Age": -0.5,      # usia muda (sudah dinormalisasi)
        "SibSp": 0.1,
        "Parch": -0.5,
        "Fare": 1.2,      # tiket mahal
        "Embarked": 0,    # Cherbourg
        "FamilySize": 0.2,
        "IsAlone": 0,
        "AgeGroup": 3,
        "FareBin": 3
    }
    
    # Contoh 2: Penumpang pria kelas 3 (kemungkinan selamat rendah)
    passenger_2 = {
        "Pclass": 3,
        "Sex": 1,         # male
        "Age": 0.3,
        "SibSp": -0.5,
        "Parch": -0.5,
        "Fare": -0.8,     # tiket murah
        "Embarked": 2,    # Southampton
        "FamilySize": -0.5,
        "IsAlone": 1,
        "AgeGroup": 3,
        "FareBin": 0
    }
    
    passengers = [
        ("Wanita Kelas 1 (Expected: Survived)", passenger_1),
        ("Pria Kelas 3 (Expected: Not Survived)", passenger_2)
    ]
    
    for desc, features in passengers:
        print(f"\nPenumpang: {desc}")
        result = predict_via_api(features, api_url)
        
        if "error" not in result:
            print(f"  Prediksi    : {result.get('prediction_label', 'N/A')}")
            print(f"  Confidence  : {result.get('confidence', 0):.2%}")
            print(f"  Latency     : {result.get('inference_time_ms', 0):.2f} ms")
            print(f"  Timestamp   : {result.get('timestamp', 'N/A')}")
        else:
            print(f"  Error: {result.get('error')}")
    
    return result


def batch_prediction(csv_path: str, api_url: str = DEFAULT_API_URL,
                     max_rows: int = 50) -> pd.DataFrame:
    """
    Melakukan prediksi batch dari file CSV.
    
    Args:
        csv_path: Path ke file CSV dengan fitur yang sudah diproses
        api_url: URL endpoint prediksi
        max_rows: Maksimum baris yang diproses
        
    Returns:
        DataFrame berisi prediksi
    """
    logger.info(f"=== BATCH PREDICTION dari {csv_path} ===")
    
    try:
        df = pd.read_csv(csv_path)
        if 'Survived' in df.columns:
            y_true = df['Survived'].values
            X = df.drop('Survived', axis=1)
        else:
            y_true = None
            X = df
        
        X = X.head(max_rows)
        logger.info(f"Memproses {len(X)} sampel...")
        
        predictions = []
        confidences = []
        latencies = []
        
        for idx, row in X.iterrows():
            features = row.to_dict()
            result = predict_via_api(features, api_url)
            
            if "error" not in result:
                predictions.append(result.get('prediction', -1))
                confidences.append(result.get('confidence', 0))
                latencies.append(result.get('inference_time_ms', 0))
            else:
                predictions.append(-1)
                confidences.append(0)
                latencies.append(0)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{len(X)}")
        
        results_df = X.copy()
        results_df['predicted_survival'] = predictions
        results_df['confidence'] = confidences
        results_df['latency_ms'] = latencies
        
        if y_true is not None:
            results_df['actual_survival'] = y_true[:max_rows]
            valid_preds = [(p, a) for p, a in zip(predictions, y_true[:max_rows]) if p != -1]
            if valid_preds:
                correct = sum(1 for p, a in valid_preds if p == a)
                accuracy = correct / len(valid_preds)
                logger.info(f"Batch Accuracy: {accuracy:.4f} ({correct}/{len(valid_preds)})")
        
        # Simpan hasil
        output_path = "batch_predictions.csv"
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Batch prediction selesai! Hasil disimpan: {output_path}")
        logger.info(f"Rata-rata latency: {np.mean(latencies):.2f} ms")
        logger.info(f"Rata-rata confidence: {np.mean(confidences):.4f}")
        
        return results_df
    
    except FileNotFoundError:
        logger.error(f"File tidak ditemukan: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error batch prediction: {e}")
        return pd.DataFrame()


def benchmark_test(api_url: str = DEFAULT_API_URL, n_requests: int = 1000):
    """
    Benchmark dan stress test untuk monitoring.
    
    Args:
        api_url: URL endpoint prediksi
        n_requests: Jumlah request untuk dikirim
    """
    logger.info(f"=== BENCHMARK TEST ({n_requests} requests) ===")
    
    results = {
        "total": 0, "success": 0, "error": 0,
        "survived": 0, "not_survived": 0,
        "latencies": []
    }
    
    start_time = time.time()
    
    for i in range(n_requests):
        features = {
            "Pclass": random.randint(1, 3),
            "Sex": random.randint(0, 1),
            "Age": random.uniform(-2.0, 2.0),
            "SibSp": random.uniform(-1.0, 2.0),
            "Parch": random.uniform(-1.0, 2.0),
            "Fare": random.uniform(-1.0, 3.0),
            "Embarked": random.randint(0, 2),
            "FamilySize": random.uniform(-1.0, 3.0),
            "IsAlone": random.randint(0, 1),
            "AgeGroup": random.randint(0, 4),
            "FareBin": random.randint(0, 3)
        }
        
        result = predict_via_api(features, api_url)
        results["total"] += 1
        
        if "error" not in result:
            results["success"] += 1
            results["latencies"].append(result.get("inference_time_ms", 0))
            if result.get("prediction") == 1:
                results["survived"] += 1
            else:
                results["not_survived"] += 1
        else:
            results["error"] += 1
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rps = (i + 1) / elapsed
            logger.info(f"Progress: {i+1}/{n_requests} | RPS: {rps:.1f}")
        
        # Jeda kecil untuk tidak overload
        time.sleep(0.01)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*55}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*55}")
    print(f"Total Requests : {results['total']}")
    print(f"Success        : {results['success']}")
    print(f"Error          : {results['error']}")
    print(f"Success Rate   : {results['success']/results['total']*100:.1f}%")
    print(f"Survived Preds : {results['survived']}")
    print(f"Not Survived   : {results['not_survived']}")
    
    if results["latencies"]:
        latencies = results["latencies"]
        print(f"\nLATENCY STATISTICS:")
        print(f"Min            : {min(latencies):.2f} ms")
        print(f"Max            : {max(latencies):.2f} ms")
        print(f"Mean           : {np.mean(latencies):.2f} ms")
        print(f"P50 (Median)   : {np.percentile(latencies, 50):.2f} ms")
        print(f"P95            : {np.percentile(latencies, 95):.2f} ms")
        print(f"P99            : {np.percentile(latencies, 99):.2f} ms")
    
    print(f"\nThroughput     : {results['total']/total_time:.1f} requests/sec")
    print(f"Total Time     : {total_time:.2f} seconds")
    print(f"{'='*55}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script untuk model Titanic'
    )
    parser.add_argument('--mode', choices=['single', 'batch', 'benchmark', 'api'],
                        default='single', help='Mode inferensi')
    parser.add_argument('--url', type=str, default=DEFAULT_API_URL,
                        help='API endpoint URL')
    parser.add_argument('--input', type=str, default='titanic_preprocessing.csv',
                        help='Input CSV untuk batch mode')
    parser.add_argument('--n', type=int, default=100,
                        help='Jumlah request untuk benchmark mode')
    parser.add_argument('--model-uri', type=str, default=None,
                        help='MLflow model URI (untuk mode mlflow)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print(f"\n{'='*55}")
    print(f"TITANIC ML MODEL INFERENCE")
    print(f"{'='*55}")
    print(f"Mode    : {args.mode}")
    print(f"API URL : {args.url}")
    print(f"Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")
    
    if args.mode == 'single' or args.mode == 'api':
        single_prediction_demo(api_url=args.url)
    
    elif args.mode == 'batch':
        results = batch_prediction(
            csv_path=args.input,
            api_url=args.url,
            max_rows=50
        )
        if not results.empty:
            print(f"\nBatch Results Preview:")
            print(results[['predicted_survival', 'confidence', 'latency_ms']].head(10))
    
    elif args.mode == 'benchmark':
        benchmark_test(api_url=args.url, n_requests=args.n)
