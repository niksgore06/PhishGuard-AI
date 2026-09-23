# PhishGuard AI — ML-Powered Phishing URL Detector 🛡️

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset: UCI PhiUSIIL](https://img.shields.io/badge/Dataset-UCI%20PhiUSIIL%20(235k)-005571)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PhishGuard AI** is an end-to-end, portfolio-grade cybersecurity machine learning application that detects malicious and phishing URLs in real time using statistical lexical analysis, structural decomposition, information entropy, and trained scikit-learn classification models.

Unlike simplistic toy projects, PhishGuard AI is trained on **235,795 real-world URLs** from the authoritative **UCI Machine Learning Repository PhiUSIIL benchmark**, comparing multiple classification algorithms with strict evaluation metrics and serving predictions through a modern, responsive Streamlit dashboard.

---

## 📌 Problem Statement

Phishing attacks remain the number one initial access vector responsible for corporate data breaches, credential theft, and ransomware deployment. Traditional blocklists (DNS sinkholes, static signature blacklists) suffer from zero-hour blind spots — attackers rapidly spin up disposable subdomains, generate randomized high-entropy URLs, and leverage raw IP addresses that bypass static reputation filters.

**PhishGuard AI** solves this by analyzing the intrinsic **lexical, structural, and obfuscation signatures** of the URL string itself. It predicts maliciousness instantly without requiring live network connections, external DNS queries, or waiting for threat feeds to update.

---

## 🌟 Key Features

- **Real-Time URL Threat Analysis**: Evaluates any input URL in under 15 milliseconds.
- **Explainable Security Indicators**: Breaks down risk factors into human-readable alerts (e.g., Raw IP Address in Hostname, `@` Redirection Delimiters, Subdomain Flooding, High Entropy).
- **Multi-Model Benchmark Engine**: Rigorously evaluates Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting algorithms on stratified holdout test sets.
- **25 Extracted Lexical & Statistical Signals**: Comprehensive feature engineering capturing character frequencies, structural components, Shannon entropy, and sensitive phishing keywords.
- **Canonical Domain Normalization**: Handles `www.` prefixes and subdomains consistently without root-domain or apex bias.
- **Interactive Cyber-Defense Dashboard**: Sleek Streamlit interface with single-URL scanning, customizable decision threshold sliders, feature inspection tables, and batch scanning.
- **Batch Scanning & Export**: Upload a CSV or paste bulk URLs to process thousands of links at once and export an audit report.
- **No External API Dependency**: Works fully offline after the initial dataset download — no API keys, no network calls during inference.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting) |
| Visualization | Matplotlib, Seaborn |
| Web Dashboard | Streamlit |
| Model Serialization | Joblib |
| Dataset Ingestion | Requests (UCI HTTP download) |

---

## 📂 Project Structure

```text
phishguard-ai/
├── data/
│   ├── raw/                             # Downloaded UCI raw dataset zip & CSV (git-ignored)
│   └── processed/                       # Processed training sample & extracted features (git-ignored)
├── models/
│   ├── phishguard_best_model.joblib     # Persisted winning model — HistGradient Boosting (git-ignored)
│   ├── feature_names.joblib             # 25 feature schema definitions (git-ignored)
│   ├── model_metadata.json              # Model metrics, confusion matrices & timestamps (git-ignored)
│   ├── model_comparison.png             # Benchmark bar chart (Accuracy, Precision, Recall, F1)
│   ├── confusion_matrices.png           # 2×2 confusion matrices for all candidate models
│   └── feature_importance.png           # Top-15 Gini importance features (from Random Forest)
├── notebooks/
│   └── phishing_url_eda_and_modeling.ipynb  # Comprehensive EDA, visualizations & modeling walkthrough
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── utils.py                         # Safe URL normalization, IPv4/IPv6 validation & parsing
│   ├── features.py                      # 25-feature extraction engine & risk explainability
│   ├── data_loader.py                   # UCI dataset downloader, preprocessor & stratified split
│   └── model.py                         # Training loop, evaluation plots & live inference class
├── app.py                               # Streamlit web dashboard (4 tabs)
├── train_model.py                       # CLI script: complete training pipeline entrypoint
├── test_pipeline.py                     # Automated unit & integration verification suite
├── requirements.txt                     # Pinned project dependencies
├── LICENSE                              # MIT License
└── README.md                            # This file
```

---

## 📊 Dataset Source & Provenance

This project uses the official **PhiUSIIL Phishing URL Dataset** hosted by the **UCI Machine Learning Repository**:

- **Repository**: [UCI Machine Learning Repository — ID 967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- **Reference**: Prasad, A., & Chandra, S. (2024). *PhiUSIIL: A novel benchmark dataset for phishing URL detection*. Computers & Security.
- **Total Instances**: 235,795 URLs (134,850 legitimate, 100,945 phishing).
- **Ground Truth**: Verified against active feeds (PhishTank, OpenPhish) and authoritative top domains (Alexa / Tranco).
- **Automated Ingestion**: `src/data_loader.py` downloads the raw archive directly from UCI, unpacks the CSV, validates records, and samples a stratified balanced subset for training.

> **Note**: The raw dataset files are excluded from this repository via `.gitignore` (combined size ~72 MB). They are downloaded automatically on the first `python train_model.py` run.

---

## 🔬 Feature Engineering Approach

The `URLFeatureExtractor` in `src/features.py` extracts **25 numerical and boolean features** grouped into five core categories:

| Category | Features Extracted | Threat Justification |
| :--- | :--- | :--- |
| **Lexical & Delimiters** | `url_length`, `dot_count`, `hyphen_count`, `slash_count`, `question_count`, `equal_count`, `ampersand_count`, `percent_count`, `at_count`, `exclamation_count` | Phishing links use excessive dots and hyphens for subdomain impersonation; `@` hides the target domain; `%` indicates obfuscated encoding. |
| **Structural & Protocol** | `is_https`, `is_ip`, `subdomain_count`, `domain_length`, `double_slash_path` | Attackers frequently host harvesters on raw IP addresses or embed `//` in paths to force redirection. |
| **Character Statistics** | `digit_count`, `digit_ratio`, `letter_count`, `letter_ratio`, `special_char_count`, `special_char_ratio` | Machine-generated attack domains exhibit anomalous character ratios compared to human-readable benign websites. |
| **Information Theory** | `url_entropy` (Shannon Entropy in bits) | Measures randomness of character distribution. High entropy highlights encrypted slugs or randomized DGAs. |
| **Heuristics & Keywords** | `is_shortener`, `suspicious_keyword_count`, `has_suspicious_keyword` | Detects redirection services (`bit.ly`, `tinyurl`) and targeting terms (`login`, `signin`, `verify`, `banking`, `secure`, `wallet`, `paypal`). |

**Key design decision**: All URLs are canonically normalized (stripping `www.` prefixes) before feature extraction **and** during training data preprocessing, ensuring the model sees a consistent domain representation in both phases.

---

## 🤖 ML Models Evaluated

We evaluated 4 distinct classification models under identical 80/20 stratified splits (24,000 train, 6,000 test):

1. **Logistic Regression** — Linear baseline with L2 regularization and balanced class weights.
2. **Decision Tree** — Non-linear tree classifier (`max_depth=14`) for interpretable rule-based decision paths.
3. **Random Forest** — Ensemble bagging classifier (`n_estimators=100`, `max_depth=16`) robust against feature noise.
4. **HistGradient Boosting** — Histogram-based gradient-boosted decision trees optimizing loss iteratively.

---

## 📈 Evaluation Metrics

All models are evaluated using five metrics on the same held-out test set:

| Metric | Why It Matters for Phishing Detection |
| :--- | :--- |
| **Accuracy** | Overall correct predictions across both classes |
| **Precision** | Fraction of flagged-phishing URLs that are actually phishing (low false-alarm rate) |
| **Recall** | Fraction of actual phishing URLs correctly caught (low miss rate) |
| **F1-Score** | Harmonic mean of Precision & Recall — the primary selection criterion |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve; threshold-independent discriminative power |

### Benchmark Results (Test Set N = 6,000)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradient Boosting ✅ Winner** | **98.53%** | **99.46%** | **97.60%** | **0.9852** | **0.9975** |
| Random Forest | 98.17% | 99.12% | 97.20% | 0.9815 | 0.9971 |
| Decision Tree | 97.62% | 99.04% | 96.17% | 0.9758 | 0.9839 |
| Logistic Regression | 93.85% | 96.90% | 90.60% | 0.9364 | 0.9781 |

**HistGradient Boosting** was selected as the production model. It achieves the highest F1-Score (0.9852) with only **16 false positives out of 3,000 legitimate test URLs** (FP rate = 0.53%).

---

## 🚀 How to Install & Run Locally

### Prerequisites
- Python 3.10+ (tested on Python 3.11.9)
- Windows / macOS / Linux

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/phishguard-ai.git
cd phishguard-ai
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
Downloads the UCI dataset (~15 MB zip), extracts features, trains and benchmarks all 4 models, and saves the best model to `models/`:
```bash
python train_model.py --sample-size 30000
```
> Training takes approximately 2–4 minutes depending on hardware. The dataset download requires an internet connection on first run only.

### 5. Run Verification Tests
```bash
python test_pipeline.py
```

### 6. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 📸 Screenshots

> _Screenshots will be added after the first deployment. The dashboard includes four tabs:_

| Tab | Description |
| :--- | :--- |
| 🔍 **URL Scanner** | Real-time single URL analysis with risk indicator breakdown and confidence score |
| 📋 **Batch Scanner** | Multi-URL or CSV upload with downloadable audit report |
| 📊 **Model Benchmarks** | Side-by-side model comparison charts and confusion matrix heatmaps |
| ℹ️ **About & Attack Vectors** | Educational reference on common phishing URL techniques |

<!-- Add screenshots below once the app is running -->
<!-- ![URL Scanner Tab](models/screenshot_scanner.png) -->
<!-- ![Batch Scanner Tab](models/screenshot_batch.png) -->
<!-- ![Model Benchmarks Tab](models/screenshot_benchmarks.png) -->

---

## 🔮 Future Improvements

- **Deep Learning / NLP**: Add a Transformer / Char-CNN or RoBERTa tokenizer trained on raw URL byte sequences for semantic understanding beyond lexical features.
- **Live WHOIS & SSL Integration**: Augment lexical features with real-time domain registration age and SSL certificate validity via async APIs.
- **REST API Deployment**: Wrap `PhishGuardPredictor` in a FastAPI service with `/predict` and `/batch` endpoints for programmatic access.
- **Browser Extension**: Package the lightweight feature extractor and serialized model into a Chrome extension (Manifest V3) for passive browsing protection.
- **Active Learning Feed**: Ingest daily feeds from PhishTank and URLhaus to automatically retrain the model on emerging attack patterns.
- **Explainability Layer**: Integrate SHAP values for per-prediction feature attribution beyond the current heuristic risk indicators.

---

## ⚠️ Security Disclaimer

PhishGuard AI provides statistical machine learning classifications based solely on URL structural and lexical patterns. It is intended as an **educational tool and auxiliary defense layer** — not a replacement for enterprise security controls.

**Limitations to be aware of:**
- Cannot detect compromised legitimate websites (URL structure may appear benign).
- Does not perform live DNS, WHOIS, or SSL certificate lookups during inference.
- Model predictions are probabilistic; a confident "Legitimate" classification does not guarantee safety.
- Shortener URLs (e.g., `bit.ly/...`) may obscure final destinations.

Always practice **defense-in-depth**: use multi-factor authentication, enterprise email filtering, threat intelligence feeds, and educate users to verify authentication destinations before entering credentials.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The UCI PhiUSIIL dataset is subject to its own usage terms. Please review the [UCI dataset page](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) before use in commercial contexts.
