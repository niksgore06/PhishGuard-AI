"""
Data Ingestion and Preprocessing Pipeline for PhishGuard AI.
Downloads, caches, validates, debiases, and samples the official benchmark dataset.
"""

import os
import io
import zipfile
import logging
from typing import Tuple, Optional
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import normalize_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PhishGuard.DataLoader")

# Official UCI Machine Learning Repository PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"


class DataLoader:
    """
    Manages downloading, extracting, and loading labeled phishing and legitimate URLs.
    Includes data-cleaning heuristics to prevent root-domain collection bias.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_dataset(self, timeout: int = 180) -> str:
        """
        Download the official UCI PhiUSIIL Phishing URL Dataset zip archive and extract CSV.
        """
        csv_candidates = [
            os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir)
            if f.endswith(".csv") and "phiusiil" in f.lower()
        ]
        if csv_candidates:
            logger.info(f"Using existing cached dataset: {csv_candidates[0]}")
            return csv_candidates[0]

        zip_path = os.path.join(self.raw_dir, "phiusiil_dataset.zip")
        logger.info(f"Downloading UCI PhiUSIIL Phishing URL Dataset from {UCI_ZIP_URL}...")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PhishGuard-AI-Downloader/1.0"
        }
        response = requests.get(UCI_ZIP_URL, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        logger.info(f"Downloaded zip archive to {zip_path}")

        logger.info("Extracting CSV file from archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        # Locate extracted CSV
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(".csv") and not file.startswith("."):
                    found_path = os.path.join(root, file)
                    logger.info(f"Found extracted dataset: {found_path}")
                    return found_path

        raise FileNotFoundError("Could not find a valid .csv inside the extracted archive.")

    def load_and_preprocess(
        self,
        csv_path: Optional[str] = None,
        sample_size: int = 30000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load the dataset, validate schema, standardize labels, and create a balanced sample.
        
        UCI PhiUSIIL schema notes:
        - 'URL': Raw URL string
        - 'label': Original label (1 = legitimate, 0 = phishing in UCI PhiUSIIL)
        We standardize:
        - 'url': Normalized, canonical URL string
        - 'is_phishing': 1 for Phishing, 0 for Legitimate (standard cybersecurity convention)
        """
        if not csv_path or not os.path.exists(csv_path):
            csv_path = self.download_dataset()

        logger.info(f"Reading dataset from {csv_path}...")
        try:
            df = pd.read_csv(csv_path, usecols=lambda col: col.strip().lower() in ["url", "label"])
        except Exception:
            df = pd.read_csv(csv_path)

        df.columns = [c.strip().lower() for c in df.columns]

        if "url" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Expected 'url' and 'label' columns, got: {list(df.columns)}")

        df = df.dropna(subset=["url", "label"]).copy()
        df["url"] = df["url"].astype(str).str.strip()
        df = df[df["url"].str.len() > 3]

        # Standardize labels: 1 = Phishing (positive class), 0 = Legitimate (negative class)
        df["is_phishing"] = df["label"].apply(lambda v: 1 if int(v) == 0 else 0)

        logger.info(f"Total dataset records: {len(df)} | Phishing: {(df['is_phishing'] == 1).sum()} | Legitimate: {(df['is_phishing'] == 0).sum()}")

        # Stratified sampling with real-world distribution debiasing
        if sample_size and sample_size < len(df):
            logger.info(f"Extracting stratified balanced sample of {sample_size} records (random_state={random_state})...")
            half_n = sample_size // 2
            legit_df = df[df["is_phishing"] == 0].sample(n=min(half_n, (df["is_phishing"] == 0).sum()), random_state=random_state).copy()
            phish_df = df[df["is_phishing"] == 1].sample(n=min(half_n, (df["is_phishing"] == 1).sum()), random_state=random_state).copy()

            # Realistic path distribution augmentation on legitimate subset to break Alexa homepage bias
            benign_paths = [
                "", "/about", "/contact", "/index.html", "/products/item", "/wiki/Computer_science",
                "/search?q=query", "/docs/guide", "/help/faq", "/categories/tech",
                "/user/settings", "/watch?v=12345", "/blog/post", "/news/article",
                "/legal/privacy", "/solutions/enterprise", "/en/home", "/download/setup"
            ]
            legit_urls = []
            for idx, raw_u in enumerate(legit_df["url"]):
                p = benign_paths[idx % len(benign_paths)]
                legit_urls.append(normalize_url(raw_u.rstrip("/") + p))
            legit_df["url"] = legit_urls

            phish_df["url"] = [normalize_url(u) for u in phish_df["url"]]

            df = pd.concat([legit_df, phish_df], ignore_index=True)
            df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            logger.info(f"Sampled distribution: Phishing={(df['is_phishing'] == 1).sum()}, Legitimate={(df['is_phishing'] == 0).sum()}")

        sample_path = os.path.join(self.processed_dir, "phishing_urls_sample.csv")
        df[["url", "is_phishing"]].to_csv(sample_path, index=False)
        logger.info(f"Saved processed URL sample to {sample_path}")

        return df[["url", "is_phishing"]]

    def get_train_test_data(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split feature matrix and labels into stratified training and testing sets.
        """
        return train_test_split(
            features_df,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
