"""
PhishGuard AI — Streamlit Web Application.
Interactive cybersecurity dashboard for real-time phishing URL detection,
feature inspection, batch scanning, and machine learning analytics.
"""

import os
import json
import io
import pandas as pd
import streamlit as st

from src.model import PhishGuardPredictor
from src.features import URLFeatureExtractor
from src.utils import is_valid_url

# Page configuration
st.set_page_config(
    page_title="PhishGuard AI — Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber-Defense Theming & Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .status-badge-phish {
        background-color: #ef4444;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }
    .status-badge-safe {
        background-color: #10b981;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    .flag-card-critical {
        border-left: 5px solid #ef4444;
        background-color: rgba(239, 68, 68, 0.1);
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
    }
    .flag-card-high {
        border-left: 5px solid #f97316;
        background-color: rgba(249, 115, 22, 0.1);
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
    }
    .flag-card-medium {
        border-left: 5px solid #eab308;
        background-color: rgba(234, 179, 8, 0.1);
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
    }
    .flag-card-low {
        border-left: 5px solid #3b82f6;
        background-color: rgba(59, 130, 246, 0.1);
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
    }
    .flag-card-safe {
        border-left: 5px solid #10b981;
        background-color: rgba(16, 185, 129, 0.1);
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
    }
    .disclaimer-box {
        background-color: rgba(100, 116, 139, 0.15);
        border: 1px dashed #64748b;
        border-radius: 8px;
        padding: 14px;
        font-size: 0.88rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load model predictor singleton."""
    return PhishGuardPredictor(models_dir="models")


def main():
    # Header Section
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.markdown("<h1 style='text-align: center; margin: 0;'>🛡️</h1>", unsafe_allow_html=True)
    with col_title:
        st.title("PhishGuard AI — Phishing URL Detector")
        st.caption("Industrial Machine-Learning Defense System for URL Threat Analysis & Threat Intelligence")

    # Load model
    try:
        predictor = load_predictor()
        metadata = predictor.metadata
        best_model_name = metadata.get("best_model_name", "Random Forest")
        eval_metrics = metadata.get("evaluation_metrics", {}).get(best_model_name, {})
    except Exception as e:
        st.error(f"⚠️ Model artifacts not found: {e}")
        st.info("Please execute the training pipeline first: `python train_model.py`")
        return

    # Sidebar Navigation & Model Info
    with st.sidebar:
        st.header("⚙️ System Status")
        st.success(f"**Model Active:** {best_model_name}")
        st.markdown(f"""
        - **Dataset:** UCI PhiUSIIL (235k benchmark)
        - **Feature Space:** {len(predictor.feature_names)} Extracted Signals
        - **Model Accuracy:** {eval_metrics.get('accuracy', 0.96):.1%}
        - **F1-Score:** {eval_metrics.get('f1_score', 0.96):.1%}
        """)

        st.markdown("---")
        st.subheader("🎯 Decision Threshold")
        threshold = st.slider(
            "Phishing Probability Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Threshold above which a URL is flagged as Phishing."
        )

        st.markdown("---")
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Security Disclaimer</strong><br>
            PhishGuard AI produces predictive classifications based on statistical and lexical URL characteristics. It does not replace full-perimeter defense, DNS filtering, or human judgment.
        </div>
        """, unsafe_allow_html=True)

    # Main Tabs
    tab_scan, tab_batch, tab_analytics, tab_about = st.tabs([
        "🔍 Single URL Scanner",
        "📂 Batch File Scanner",
        "📊 Model Benchmarks & Analytics",
        "ℹ️ About & Attack Vectors"
    ])

    # ---------------- TAB 1: SINGLE URL SCANNER ----------------
    with tab_scan:
        st.subheader("Analyze a Web Address")
        
        # Quick Sample Buttons
        st.markdown("**Quick Test Samples:**")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        sample_url = ""
        if col_s1.button("🟢 Google (Legitimate)", use_container_width=True):
            sample_url = "https://www.google.com"
        if col_s2.button("🟢 Wikipedia (Legitimate)", use_container_width=True):
            sample_url = "https://en.wikipedia.org/wiki/Computer_security"
        if col_s3.button("🔴 IP Spoofing (Phish)", use_container_width=True):
            sample_url = "http://192.168.1.100/paypal-update/account-login.php"
        if col_s4.button("🔴 Subdomain Trick (Phish)", use_container_width=True):
            sample_url = "http://chase.com.security-verify-banking.xyz/signin?user=auth"

        url_input = st.text_input(
            "Enter URL to inspect:",
            value=sample_url,
            placeholder="e.g. https://login.microsoftonline.com or suspicious-link.xyz/verify",
            key="url_input_field"
        )

        scan_btn = st.button("🚀 Scan URL", type="primary", use_container_width=True)

        if scan_btn or url_input:
            if not url_input.strip():
                st.warning("Please enter a URL to scan.")
            elif not is_valid_url(url_input):
                st.error("⚠️ Invalid URL structure. Please enter a valid domain, IP, or web address.")
            else:
                with st.spinner("Analyzing lexical patterns, entropy, and structural anomalies..."):
                    result = predictor.predict(url_input)

                # Custom threshold adjustment
                phish_prob = result["phishing_probability"]
                is_flagged = phish_prob >= threshold
                label = "Phishing" if is_flagged else "Legitimate"

                st.markdown("---")

                # Verdict Banner
                col_verdict, col_prob, col_metrics = st.columns([2, 2, 2])
                with col_verdict:
                    if is_flagged:
                        st.markdown("""
                        <div style='text-align: center; padding: 15px;'>
                            <div class='status-badge-phish'>🚨 PHISHING DETECTED</div>
                            <h3 style='color: #ef4444; margin-top: 10px;'>High Risk Threat</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='text-align: center; padding: 15px;'>
                            <div class='status-badge-safe'>🛡️ LEGITIMATE / SAFE</div>
                            <h3 style='color: #10b981; margin-top: 10px;'>Standard Profile</h3>
                        </div>
                        """, unsafe_allow_html=True)

                with col_prob:
                    st.metric(
                        label="Phishing Probability",
                        value=f"{phish_prob * 100:.1f}%",
                        delta=f"{'+' if is_flagged else '-'}{abs(phish_prob - 0.5) * 100:.1f}% vs baseline",
                        delta_color="inverse" if is_flagged else "normal"
                    )
                    st.progress(phish_prob)

                with col_metrics:
                    st.metric(
                        label="Analyzed Characters",
                        value=result["features"]["url_length"]
                    )
                    st.metric(
                        label="Entropy Score",
                        value=f"{result['features']['url_entropy']:.2f} bits"
                    )

                # Key Risk Indicators Section
                st.subheader("🚩 Threat Indicators & Heuristic Analysis")
                indicators = result["risk_indicators"]
                for ind in indicators:
                    sev = ind["severity"].lower()
                    st.markdown(f"""
                    <div class='flag-card-{sev}'>
                        <strong>[{ind['severity']}] {ind['title']}</strong><br>
                        <span style='color: #cbd5e1;'>{ind['desc']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Extracted Features Table
                st.subheader("🔬 Extracted Feature Vector")
                feat_df = pd.DataFrame([
                    {"Feature": k, "Extracted Value": v}
                    for k, v in result["features"].items()
                ])
                st.dataframe(feat_df, use_container_width=True, height=280)

    # ---------------- TAB 2: BATCH FILE SCANNER ----------------
    with tab_batch:
        st.subheader("Bulk Threat Scanning")
        st.write("Scan lists of URLs from file or direct text input.")

        batch_input_mode = st.radio("Input Method:", ["Paste Multiple URLs", "Upload CSV / Text File"], horizontal=True)
        urls_to_scan = []

        if batch_input_mode == "Paste Multiple URLs":
            raw_text = st.text_area(
                "Paste one URL per line:",
                placeholder="https://google.com\nhttp://malicious-fake-bank.xyz/update\nhttps://github.com",
                height=150
            )
            if raw_text:
                urls_to_scan = [line.strip() for line in raw_text.splitlines() if line.strip()]
        else:
            uploaded_file = st.file_uploader("Upload CSV or TXT file (URL column will be auto-detected)", type=["csv", "txt"])
            if uploaded_file:
                if uploaded_file.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded_file)
                    url_cols = [c for c in df_up.columns if "url" in c.lower()]
                    if url_cols:
                        urls_to_scan = df_up[url_cols[0]].dropna().astype(str).tolist()
                    else:
                        urls_to_scan = df_up.iloc[:, 0].dropna().astype(str).tolist()
                else:
                    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                    urls_to_scan = [line.strip() for line in content.splitlines() if line.strip()]

        if st.button("🔍 Execute Batch Scan", type="primary"):
            if not urls_to_scan:
                st.warning("Please provide at least one URL to scan.")
            else:
                progress_bar = st.progress(0.0)
                batch_records = []
                total = len(urls_to_scan)

                for idx, u in enumerate(urls_to_scan):
                    res = predictor.predict(u)
                    if res.get("valid"):
                        p_prob = res["phishing_probability"]
                        flagged = p_prob >= threshold
                        batch_records.append({
                            "URL": u,
                            "Verdict": "Phishing" if flagged else "Legitimate",
                            "Phishing Probability": f"{p_prob * 100:.1f}%",
                            "Confidence": f"{res['confidence_score']:.1f}%",
                            "Flags Triggered": len(res["risk_indicators"])
                        })
                    else:
                        batch_records.append({
                            "URL": u,
                            "Verdict": "Invalid URL",
                            "Phishing Probability": "N/A",
                            "Confidence": "N/A",
                            "Flags Triggered": 0
                        })
                    progress_bar.progress((idx + 1) / total)

                batch_df = pd.DataFrame(batch_records)
                st.success(f"Successfully processed {total} URLs.")

                # Summary Statistics
                phish_count = (batch_df["Verdict"] == "Phishing").sum()
                legit_count = (batch_df["Verdict"] == "Legitimate").sum()
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Total Scanned", total)
                col_b2.metric("🚨 Flagged as Phishing", phish_count)
                col_b3.metric("🛡️ Legitimate URLs", legit_count)

                st.dataframe(batch_df, use_container_width=True)

                # Download Results as CSV
                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Scan Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="phishguard_batch_report.csv",
                    mime="text/csv"
                )

    # ---------------- TAB 3: MODEL BENCHMARKS & ANALYTICS ----------------
    with tab_analytics:
        st.subheader("Machine Learning Performance & Cross-Model Benchmarking")
        st.write("All models were trained on stratified splits of the official UCI PhiUSIIL Phishing URL Dataset.")

        # Benchmark Metrics Table
        if "evaluation_metrics" in metadata:
            metrics_rows = []
            for m_name, m_stats in metadata["evaluation_metrics"].items():
                metrics_rows.append({
                    "Model": m_name,
                    "Accuracy": f"{m_stats.get('accuracy', 0):.4f}",
                    "Precision": f"{m_stats.get('precision', 0):.4f}",
                    "Recall": f"{m_stats.get('recall', 0):.4f}",
                    "F1-Score": f"{m_stats.get('f1_score', 0):.4f}",
                    "ROC-AUC": f"{m_stats.get('roc_auc', 0):.4f}"
                })
            bench_df = pd.DataFrame(metrics_rows)
            st.dataframe(bench_df, use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            comp_img = "models/model_comparison.png"
            if os.path.exists(comp_img):
                st.image(comp_img, caption="Cross-Model Benchmark Comparison", use_container_width=True)
            else:
                st.info("Model comparison plot will appear after running train_model.py")

        with col_p2:
            cm_img = "models/confusion_matrices.png"
            if os.path.exists(cm_img):
                st.image(cm_img, caption="Validation Set Confusion Matrices", use_container_width=True)
            else:
                st.info("Confusion matrix plot will appear after running train_model.py")

        st.markdown("---")
        st.subheader("Feature Importance Distribution")
        fi_img = "models/feature_importance.png"
        if os.path.exists(fi_img):
            st.image(fi_img, caption="Top URL Predictors Identified by Random Forest Gini Impurity", use_container_width=True)
        else:
            st.info("Feature importance plot will appear after running train_model.py")

    # ---------------- TAB 4: ABOUT & ATTACK VECTORS ----------------
    with tab_about:
        st.subheader("Understanding Phishing URL Tactics")
        st.markdown("""
        Phishing remains the primary vector for enterprise credential theft and security breaches. Threat actors employ distinct lexical and structural deception methods:

        1. **Domain Spoofing & Subdomain Obfuscation**: Attackers register deceptive domains (`chase.com.security-verify.xyz`) where legitimate brand names are placed in subdomains, misleading users who only glance at the left part of a link.
        2. **Raw IP Addresses**: Phishers bypass domain lookup reputation feeds by hosting credential harvesters on direct IP addresses (`http://192.168.1.1/login`).
        3. **Character Obfuscation & High Entropy**: Automated attack frameworks use randomized parameters, high-entropy slugs, and hex percent-encoding (`%20`, `%40`) to evade static regex rules.
        4. **Credential Redirection Delimiters**: Utilizing `@` to cause browsers to treat preceding text as basic auth credentials, routing victims to the subsequent domain.
        5. **URL Shorteners**: Masks the target domain to defeat email gateway filters.

        ### Dataset Grounding
        This system is trained using the **PhiUSIIL Phishing URL Dataset** (University of California Irvine ML Repository, ID 967; *Computers & Security* 2024 by Arvind Prasad & Shalini Chandra).
        """)


if __name__ == "__main__":
    main()
