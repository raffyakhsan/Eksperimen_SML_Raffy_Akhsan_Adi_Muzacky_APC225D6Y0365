"""
modelling_tuning.py
====================
Training model dengan Hyperparameter Tuning menggunakan MLflow manual logging.
Mendukung penyimpanan ke DagsHub untuk Advanced criteria.

Penggunaan:
    # Basic (local MLflow):
    python modelling_tuning.py --dataset titanic_preprocessing.csv
    
    # Advanced (DagsHub):
    python modelling_tuning.py --dataset titanic_preprocessing.csv --use-dagshub \
        --dagshub-user raffyakhsan --dagshub-repo SML-Titanic
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
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.preprocessing import label_binarize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(dataset_path: str):
    """Memuat dan mempersiapkan dataset."""
    if not os.path.exists(dataset_path):
        logger.warning(f"File tidak ditemukan: {dataset_path}. Membuat dataset dummy...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=891, n_features=11, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(11)])
        df['Survived'] = y
        df.to_csv(dataset_path, index=False)
    
    df = pd.read_csv(dataset_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Data dimuat. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def save_confusion_matrix(y_true, y_pred, run_name: str) -> str:
    """Membuat dan menyimpan confusion matrix sebagai artefak."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Survived', 'Survived'])
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f'Confusion Matrix - {run_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'confusion_matrix_{run_name.replace(" ", "_")}.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def save_roc_curve(y_true, y_prob, run_name: str) -> str:
    """Membuat dan menyimpan ROC Curve sebagai artefak."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {run_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    path = f'roc_curve_{run_name.replace(" ", "_")}.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def save_feature_importance(model, feature_names: list, run_name: str) -> str:
    """Membuat dan menyimpan feature importance plot sebagai artefak."""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance_df['feature'], importance_df['importance'],
                   color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Feature Importance - {run_name}', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, importance_df['importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    path = f'feature_importance_{run_name.replace(" ", "_")}.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def save_learning_curve_data(model, X_train, y_train, run_name: str) -> str:
    """Membuat learning curve analysis sebagai artefak."""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color='blue')
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color='red')
    ax.set_xlabel('Training Size', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_title(f'Learning Curve - {run_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f'learning_curve_{run_name.replace(" ", "_")}.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def train_with_tuning(dataset_path: str, use_dagshub: bool = False,
                       dagshub_user: str = None, dagshub_repo: str = None):
    """
    Training dengan hyperparameter tuning dan manual MLflow logging.
    Mendukung DagsHub untuk Advanced criteria.
    """
    
    # Setup MLflow (DagsHub atau local)
    if use_dagshub and dagshub_user and dagshub_repo:
        import dagshub
        dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
        logger.info(f"DagsHub initialized: {dagshub_user}/{dagshub_repo}")
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        logger.info("Menggunakan MLflow lokal: http://127.0.0.1:5000/")
    
    mlflow.set_experiment("Titanic Survival Prediction - Tuning")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data(dataset_path)
    
    # Definisi kandidat model dan hyperparameter grid
    models_config = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7, None],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5],
                "subsample": [0.7, 0.8, 1.0]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["lbfgs", "liblinear"],
                "penalty": ["l2"]
            }
        }
    }
    
    best_models = {}
    
    for model_name, config in models_config.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*50}")
        
        # Grid Search Cross Validation
        grid_search = GridSearchCV(
            config["model"],
            config["param_grid"],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Evaluasi pada test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Hitung semua metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "cv_mean_accuracy": grid_search.best_score_,
            "cv_std_accuracy": grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        }
        
        run_name = f"{model_name}_BestParams"
        
        with mlflow.start_run(run_name=run_name):
            # Manual logging - Parameters
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("tuning_method", "GridSearchCV")
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", len(feature_names))
            
            # Manual logging - Metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            logger.info(f"Best params: {best_params}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # === ARTEFAK 1: Confusion Matrix ===
            cm_path = save_confusion_matrix(y_test, y_pred, run_name)
            mlflow.log_artifact(cm_path)
            logger.info(f"Artefak: Confusion Matrix -> {cm_path}")
            
            # === ARTEFAK 2: ROC Curve ===
            roc_path = save_roc_curve(y_test, y_prob, run_name)
            mlflow.log_artifact(roc_path)
            logger.info(f"Artefak: ROC Curve -> {roc_path}")
            
            # === ARTEFAK 3: Feature Importance (untuk tree-based) ===
            fi_path = save_feature_importance(best_model, feature_names, run_name)
            if fi_path:
                mlflow.log_artifact(fi_path)
                logger.info(f"Artefak: Feature Importance -> {fi_path}")
            
            # === ARTEFAK 4: Learning Curve ===
            lc_path = save_learning_curve_data(best_model, X_train, y_train, run_name)
            mlflow.log_artifact(lc_path)
            logger.info(f"Artefak: Learning Curve -> {lc_path}")
            
            # === ARTEFAK 5: Classification Report (JSON) ===
            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = f'classification_report_{run_name.replace(" ", "_")}.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            mlflow.log_artifact(report_path)
            logger.info(f"Artefak: Classification Report -> {report_path}")
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model",
                                      registered_model_name=f"titanic_{model_name.lower()}")
            
            # Simpan run ID
            run_id = mlflow.active_run().info.run_id
            best_models[model_name] = {
                "model": best_model,
                "metrics": metrics,
                "run_id": run_id
            }
        
        # Cleanup temporary files
        for path in [cm_path, roc_path, report_path]:
            if os.path.exists(path):
                os.remove(path)
        if fi_path and os.path.exists(fi_path):
            os.remove(fi_path)
        if os.path.exists(lc_path):
            os.remove(lc_path)
    
    # Tampilkan perbandingan semua model
    print(f"\n{'='*60}")
    print("PERBANDINGAN SEMUA MODEL")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Accuracy':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 60)
    for name, info in best_models.items():
        m = info['metrics']
        print(f"{name:<25} {m['accuracy']:<12.4f} {m['f1_score']:<12.4f} {m['roc_auc']:<12.4f}")
    
    # Best model
    best_name = max(best_models.keys(), key=lambda k: best_models[k]['metrics']['accuracy'])
    print(f"\nModel Terbaik: {best_name} (Accuracy: {best_models[best_name]['metrics']['accuracy']:.4f})")
    print(f"{'='*60}")
    
    return best_models


def parse_args():
    parser = argparse.ArgumentParser(description='Training dengan tuning dan manual MLflow logging')
    parser.add_argument('--dataset', type=str, default='titanic_preprocessing.csv')
    parser.add_argument('--use-dagshub', action='store_true', help='Gunakan DagsHub untuk advanced')
    parser.add_argument('--dagshub-user', type=str, default=None, help='DagsHub username')
    parser.add_argument('--dagshub-repo', type=str, default=None, help='DagsHub repo name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_with_tuning(
        dataset_path=args.dataset,
        use_dagshub=args.use_dagshub,
        dagshub_user=args.dagshub_user,
        dagshub_repo=args.dagshub_repo
    )
