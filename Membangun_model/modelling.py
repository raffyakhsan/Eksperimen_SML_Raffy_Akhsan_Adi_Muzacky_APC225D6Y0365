"""
modelling.py
=============
Training model Machine Learning menggunakan MLflow Tracking.
Menggunakan autolog untuk logging otomatis semua metrics dan artifacts.

Penggunaan:
    python modelling.py [--dataset titanic_preprocessing.csv]
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_preprocessed_data(dataset_path: str):
    """Memuat dataset yang sudah diproses."""
    if not os.path.exists(dataset_path):
        logger.warning(f"File {dataset_path} tidak ditemukan. Membuat dataset dummy...")
        # Buat dataset dummy untuk testing
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=891, n_features=11, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(11)])
        df['Survived'] = y
        df.to_csv(dataset_path, index=False)
        logger.info("Dataset dummy dibuat.")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset dimuat: {df.shape}")
    return df


def train_model(dataset_path: str = 'titanic_preprocessing.csv'):
    """
    Melatih model RandomForestClassifier dengan MLflow autolog.
    
    Args:
        dataset_path: Path ke dataset yang sudah diproses
    """
    # Set MLflow tracking URI ke localhost
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("Titanic Survival Prediction")
    
    # Load data
    df = load_preprocessed_data(dataset_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Aktifkan MLflow autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        logger.info("Memulai training dengan MLflow autolog...")
        
        # Inisialisasi dan training model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model berhasil dilatih!")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        print(f"\n{'='*50}")
        print(f"TRAINING SELESAI")
        print(f"{'='*50}")
        print(f"Model: RandomForestClassifier")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"MLflow Tracking URI: http://127.0.0.1:5000/")
        print(f"{'='*50}")
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train ML model dengan MLflow')
    parser.add_argument('--dataset', type=str, default='titanic_preprocessing.csv',
                        help='Path ke dataset preprocessing')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(dataset_path=args.dataset)
