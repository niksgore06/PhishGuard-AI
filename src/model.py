"""
Machine Learning Engine for PhishGuard AI:
Model training, cross-model benchmarking, evaluation visualization, and inference wrapper.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.features import URLFeatureExtractor
from src.utils import normalize_url, is_valid_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PhishGuard.Model")


class ModelManager:
    """
    Handles training, evaluating, plotting, and persisting scikit-learn models for PhishGuard AI.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.best_model_name: Optional[str] = None
        self.best_model: Any = None
        self.results: Dict[str, Dict[str, Any]] = {}

    def get_candidate_models(self) -> Dict[str, Any]:
        """
        Instantiate candidate classification algorithms.
        """
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=14,
                min_samples_split=10,
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_split=6,
                random_state=42,
                n_jobs=-1
            ),
            "HistGradient Boosting": HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=10,
                random_state=42
            )
        }

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all candidate models and record comparative metrics.
        """
        candidates = self.get_candidate_models()
        self.results = {}
        best_f1 = -1.0

        for name, model in candidates.items():
            logger.info(f"Training {name} on {len(X_train)} samples...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_test, y_prob))
            except Exception:
                auc = 0.5
            cm = confusion_matrix(y_test, y_pred).tolist()

            logger.info(
                f"{name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
                f"Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}"
            )

            self.results[name] = {
                "model": model,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc,
                "confusion_matrix": cm,
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            }

            # Select best model based on F1-Score
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name
                self.best_model = model

        logger.info(f"Best selected model: {self.best_model_name} (F1 = {best_f1:.4f})")
        return self.results

    def generate_and_save_plots(self, X_train: pd.DataFrame):
        """
        Generate high-resolution visual plots for model comparison, confusion matrices, and feature importance.
        """
        sns.set_theme(style="whitegrid", palette="muted")

        # 1. Model Comparison Bar Chart
        metrics_df = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": data["accuracy"],
                "Precision": data["precision"],
                "Recall": data["recall"],
                "F1-Score": data["f1_score"]
            }
            for name, data in self.results.items()
        ])
        
        melted_df = pd.melt(metrics_df, id_vars=["Model"], var_name="Metric", value_name="Score")
        plt.figure(figsize=(10, 5.5))
        chart = sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="Blues_d")
        plt.title("PhishGuard AI — Model Performance Benchmark", fontsize=14, fontweight="bold", pad=15)
        plt.ylim(0.7, 1.02)
        plt.ylabel("Score (0.0 - 1.0)", fontsize=11)
        plt.xlabel("Candidate Model", fontsize=11)
        plt.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        comp_plot_path = os.path.join(self.models_dir, "model_comparison.png")
        plt.savefig(comp_plot_path, dpi=200)
        plt.close()
        logger.info(f"Saved model comparison plot to {comp_plot_path}")

        # 2. Confusion Matrices Subplots
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        axes = axes.flatten()
        class_labels = ["Legitimate", "Phishing"]

        for idx, (name, data) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = np.array(data["confusion_matrix"])
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_labels, yticklabels=class_labels,
                annot_kws={"size": 13, "weight": "bold"}
            )
            is_best = " (Winner)" if name == self.best_model_name else ""
            ax.set_title(f"{name}{is_best}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted Label", fontsize=10)
            ax.set_ylabel("True Label", fontsize=10)

        plt.suptitle("PhishGuard AI — Confusion Matrices", fontsize=15, fontweight="bold", y=1.00)
        plt.tight_layout()
        cm_plot_path = os.path.join(self.models_dir, "confusion_matrices.png")
        plt.savefig(cm_plot_path, dpi=200)
        plt.close()
        logger.info(f"Saved confusion matrices to {cm_plot_path}")

        # 3. Feature Importance (from Random Forest)
        rf_model = self.results.get("Random Forest", {}).get("model")
        if rf_model and hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            feature_names = X_train.columns
            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(15)

            plt.figure(figsize=(9, 6))
            sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
            plt.title("PhishGuard AI — Top 15 Indicative URL Features (Random Forest)", fontsize=13, fontweight="bold", pad=12)
            plt.xlabel("Mean Impurity Reduction (Gini Importance)", fontsize=10)
            plt.ylabel("Extracted Feature", fontsize=10)
            plt.tight_layout()
            fi_plot_path = os.path.join(self.models_dir, "feature_importance.png")
            plt.savefig(fi_plot_path, dpi=200)
            plt.close()
            logger.info(f"Saved feature importance plot to {fi_plot_path}")

    def save_artifacts(self, feature_names: List[str]):
        """
        Serialize the best model, feature list, and metadata.
        """
        if not self.best_model:
            raise ValueError("No trained model to save.")

        model_path = os.path.join(self.models_dir, "phishguard_best_model.joblib")
        joblib.dump(self.best_model, model_path)
        logger.info(f"Saved winning model ({self.best_model_name}) to {model_path}")

        features_path = os.path.join(self.models_dir, "feature_names.joblib")
        joblib.dump(feature_names, features_path)
        logger.info(f"Saved feature schema ({len(feature_names)} features) to {features_path}")

        metadata = {
            "best_model_name": self.best_model_name,
            "created_at": datetime.now().isoformat(),
            "feature_names": feature_names,
            "evaluation_metrics": {
                name: {
                    "accuracy": data["accuracy"],
                    "precision": data["precision"],
                    "recall": data["recall"],
                    "f1_score": data["f1_score"],
                    "roc_auc": data["roc_auc"],
                    "confusion_matrix": data["confusion_matrix"]
                }
                for name, data in self.results.items()
            }
        }
        metadata_path = os.path.join(self.models_dir, "model_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to {metadata_path}")


class PhishGuardPredictor:
    """
    Live inference engine loaded by Streamlit and production callers.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "phishguard_best_model.joblib")
        self.features_path = os.path.join(models_dir, "feature_names.joblib")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Run train_model.py first.")

        self.model = joblib.load(self.model_path)
        self.feature_names = joblib.load(self.features_path)
        self.extractor = URLFeatureExtractor()

        self.metadata = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict(self, raw_url: str) -> Dict[str, Any]:
        """
        Run end-to-end inference on a single URL string.
        """
        if not is_valid_url(raw_url):
            return {
                "valid": False,
                "error": "Invalid URL format. Please enter a valid web address or hostname.",
                "url": raw_url
            }

        normalized = normalize_url(raw_url)
        features_dict = self.extractor.extract_features(normalized)
        features_df = pd.DataFrame([features_dict], columns=self.feature_names)

        pred = int(self.model.predict(features_df)[0])
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features_df)[0]
            prob_legit = float(probs[0])
            prob_phish = float(probs[1])
        else:
            prob_phish = 1.0 if pred == 1 else 0.0
            prob_legit = 1.0 - prob_phish

        risk_indicators = self.extractor.explain_risk_indicators(features_dict)

        return {
            "valid": True,
            "url": raw_url,
            "normalized_url": normalized,
            "prediction": "Phishing" if pred == 1 else "Legitimate",
            "is_phishing": pred,
            "phishing_probability": round(prob_phish, 4),
            "legitimate_probability": round(prob_legit, 4),
            "confidence_score": round(max(prob_phish, prob_legit) * 100, 2),
            "features": features_dict,
            "risk_indicators": risk_indicators
        }
