"""
PhishGuard AI — Complete Machine Learning Training Pipeline.
Runs data download, feature extraction, multi-model evaluation, visualization, and model serialization.
"""

import os
import sys
import argparse
import logging
import time
import pandas as pd

from src.data_loader import DataLoader
from src.features import URLFeatureExtractor
from src.model import ModelManager, PhishGuardPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PhishGuard.Train")


def run_pipeline(sample_size: int = 30000, test_size: float = 0.2, random_state: int = 42):
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("   PHISHGUARD AI — ML TRAINING & BENCHMARKING PIPELINE")
    logger.info("=" * 70)

    # 1. Ingestion & Preprocessing
    logger.info("\n--- STEP 1: Ingesting Real Phishing & Legitimate URL Dataset ---")
    data_loader = DataLoader(data_dir="data")
    df = data_loader.load_and_preprocess(sample_size=sample_size, random_state=random_state)
    logger.info(f"Loaded {len(df)} total samples.")

    # 2. Feature Engineering
    logger.info("\n--- STEP 2: Extracting Security & Lexical URL Features ---")
    extractor = URLFeatureExtractor()
    feat_start = time.time()
    X = extractor.transform(df["url"])
    y = df["is_phishing"]
    feat_time = time.time() - feat_start
    logger.info(f"Extracted {X.shape[1]} features across {len(X)} URLs in {feat_time:.2f}s.")

    # Save processed features cache for notebook EDA
    processed_cache_path = os.path.join("data", "processed", "features_extracted.csv")
    full_processed_df = pd.concat([df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    full_processed_df.to_csv(processed_cache_path, index=False)
    logger.info(f"Cached extracted feature matrix to {processed_cache_path}")

    # 3. Train/Test Split
    logger.info(f"\n--- STEP 3: Stratified Split (test_size={test_size}, random_state={random_state}) ---")
    X_train, X_test, y_train, y_test = data_loader.get_train_test_data(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 4. Multi-Model Training & Evaluation
    logger.info("\n--- STEP 4: Training & Evaluating Classification Models ---")
    model_mgr = ModelManager(models_dir="models")
    results = model_mgr.train_and_evaluate(X_train, X_test, y_train, y_test)

    # 5. Visualizations
    logger.info("\n--- STEP 5: Generating Benchmark & Evaluation Charts ---")
    model_mgr.generate_and_save_plots(X_train)

    # 6. Model Serialization
    logger.info("\n--- STEP 6: Serializing Best Model & Feature Schema ---")
    model_mgr.save_artifacts(feature_names=extractor.FEATURE_NAMES)

    # 7. Verification Inferences
    logger.info("\n--- STEP 7: Testing Live Predictor on Real-World URLs ---")
    predictor = PhishGuardPredictor(models_dir="models")

    test_urls = [
        "https://www.google.com",
        "https://github.com/microsoft/vscode",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "http://192.168.1.105/paypal-login/update-account.php",
        "http://secure-login.chase.com.verify-billing-update.xyz/auth?user=victim",
        "http://appleid.apple.com.verify-device-alert.info/signin"
    ]

    logger.info("-" * 70)
    for u in test_urls:
        res = predictor.predict(u)
        logger.info(f"URL: {u}")
        logger.info(
            f" => Prediction: [{res['prediction']}] | "
            f"Confidence: {res['confidence_score']}% | "
            f"Phishing Prob: {res['phishing_probability']} | "
            f"Flags: {len(res['risk_indicators'])}"
        )
    logger.info("-" * 70)

    total_time = time.time() - start_time
    logger.info(f"\nPipeline completed successfully in {total_time:.2f} seconds!")
    logger.info(f"Winning Model: {model_mgr.best_model_name}")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PhishGuard AI classification models.")
    parser.add_argument("--sample-size", type=int, default=30000, help="Number of balanced samples to train on.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed.")
    args = parser.parse_args()

    run_pipeline(sample_size=args.sample_size, test_size=args.test_size, random_state=args.random_state)
