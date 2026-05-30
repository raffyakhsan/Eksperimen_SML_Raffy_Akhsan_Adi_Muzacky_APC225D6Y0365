"""
modelling.py (MLProject version)
==================================
Training model ML dengan MLflow untuk dijalankan via GitHub Actions CI/CD.
File ini diadaptasi dari eksperimen notebook untuk kebutuhan MLflow Projects.

Penggunaan:
    mlflow run . --experiment-name "Titanic Survival Prediction"
    python modelling.py --dataset titanic_preprocessing.csv
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
import os
import json
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(dataset_path: str):
    """Memuat dataset preprocessing."""
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset tidak ditemukan: {dataset_path}")
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Data siap. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, list(X.columns)


def make_confusion_matrix_plot(y_true, y_pred) -> str:
    """Buat dan simpan confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not Survived', 'Survived'])
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    path = 'training_confusion_matrix.png'
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def make_roc_plot(y_true, y_prob) -> str:
    """Buat dan simpan ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color='darkorange', label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], lw=2, linestyle='--', color='navy')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    plt.tight_layout()
    path = 'roc_curve.png'
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def train(args):
    """Main training function."""
    # Setup MLflow
    if os.environ.get('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
    mlflow.set_experiment("Titanic Survival Prediction")
    
    X_train, X_test, y_train, y_test, feature_names = load_data(args.dataset)
    
    # Aktifkan autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_CI_Pipeline"):
        # Training
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Log metrics tambahan (manual)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", auc)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())
        
        # Log artifacts
        cm_path = make_confusion_matrix_plot(y_test, y_pred)
        mlflow.log_artifact(cm_path)
        
        roc_path = make_roc_plot(y_test, y_prob)
        mlflow.log_artifact(roc_path)
        
        # Log classification report
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = "metric_info.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_path)
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        
        # Simpan run_id ke file untuk digunakan workflow GitHub Actions
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)
        
        print(f"\n{'='*55}")
        print(f"TRAINING CI PIPELINE SELESAI")
        print(f"{'='*55}")
        print(f"Run ID     : {run_id}")
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"ROC-AUC    : {auc:.4f}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"{'='*55}")
        
        # Cleanup
        for p in [cm_path, roc_path, report_path]:
            if os.path.exists(p): os.remove(p)
    
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='titanic_preprocessing.csv')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--min-samples-split', type=int, default=4)
    parser.add_argument('--min-samples-leaf', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
