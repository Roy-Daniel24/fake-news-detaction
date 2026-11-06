"""
app.py
Production-grade multi-page Streamlit Dashboard for Fake News Detection.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ----------------- PAGE CONFIG (must be the first Streamlit call) -----------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import after set_page_config so an import error can be shown cleanly in the UI.
try:
    from utils.text_preprocessor import clean_text
except ImportError as exc:  # pragma: no cover
    st.error(f"Could not import `utils.text_preprocessor.clean_text`: {exc}")
    st.stop()

# ----------------- CONSTANTS -----------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_metrics.json")
DEFAULT_DATASET_PATH = os.path.join("dataset", "news_dataset.csv")

FAKE_LABEL = 1  # 1 = Fake, 0 = Real
LABEL_NAMES = {0: "Real", 1: "Fake"}

# ----------------- CUSTOM CSS -----------------
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2563EB;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
        text-align: center;
        height: 100%;
    }
    .metric-card h3, .metric-card p { color: white; }
    .status-fake {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        color: #991B1B;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .status-real {
        background-color: #F0FDF4;
        border-left: 6px solid #22C55E;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        padding: 0.85rem;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-top: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------- HELPERS -----------------
def _mtime(path: str) -> float:
    """File modification time, used so caches refresh when files are retrained."""
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_resource
def load_trained_artifacts(model_mtime: float, vec_mtime: float):
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


@st.cache_data
def load_metrics(metrics_mtime: float):
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_default_dataset(dataset_mtime: float):
    if os.path.exists(DEFAULT_DATASET_PATH):
        return pd.read_csv(DEFAULT_DATASET_PATH)
    return None


def compute_confidence(model, vec, prediction):
    """Return confidence (0-100) for the predicted class, or None if unavailable."""
    classes = getattr(model, "classes_", np.array([0, 1]))
    class_idx = int(np.where(classes == prediction)[0][0])

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(vec)[0][class_idx] * 100)

    if hasattr(model, "decision_function"):
        decision = float(np.ravel(model.decision_function(vec))[0])
        prob_positive = 1 / (1 + np.exp(-decision))  # sigmoid of margin (approximate)
        # decision > 0 corresponds to classes_[1]
        prob_of_pred = prob_positive if class_idx == 1 else 1 - prob_positive
        return float(prob_of_pred * 100)

    return None


def show_fig(fig):
    """Render a matplotlib figure and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


# ----------------- LOAD ARTIFACTS -----------------
model, vectorizer = load_trained_artifacts(_mtime(MODEL_PATH), _mtime(VECTORIZER_PATH))
metrics_data = load_metrics(_mtime(METRICS_PATH))

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("# 🛡️")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 News Detection", "📊 Analytics & EDA", "📁 Dataset Explorer"],
)

st.sidebar.markdown("---")
st.sidebar.caption("⚡ **Fake News Detection System**")
st.sidebar.caption("Developed with Scikit-Learn & Streamlit")

# ==========================================================
# PAGE 1: HOME
# ==========================================================
if page == "🏠 Home":
    st.markdown('<div class="main-title">🛡️ Fake News Detection System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">An intelligent Natural Language Processing platform designed to '
        "analyze textual claims and discern authentic journalism from misinformation.</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Machine Learning</h3>
            <p>Compares Naive Bayes, Logistic Regression & Passive Aggressive Classifiers.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3>NLP Pipeline</h3>
            <p>Utilizes regex text cleaning, stopword filtration, stemming & n-gram TF-IDF.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h3>Instant Verification</h3>
            <p>Analyzes headlines and full-length articles with confidence scores.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("💡 How It Operates")
    st.markdown(
        """
    1. **Input Submission**: Paste either a brief news headline or a complete investigative article.
    2. **NLP Transformation**: Text undergoes tokenization, noise removal, and TF-IDF matrix projection.
    3. **Model Scoring**: The best-performing classifier (selected by F1-score) evaluates linguistic cues and patterns.
    4. **Prediction & Audit**: Results are labeled as **REAL** or **FAKE** alongside confidence percentages.
    """
    )

# ==========================================================
# PAGE 2: NEWS DETECTION
# ==========================================================
elif page == "🔍 News Detection":
    st.markdown('<div class="main-title">🔍 Real-Time News Verification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enter a news headline or full article text below to verify its authenticity.</div>',
        unsafe_allow_html=True,
    )

    if model is None or vectorizer is None:
        st.error("⚠️ Model artifacts were not found! Please run `python train.py` first to train and save the model.")
    else:
        news_input = st.text_area(
            "Enter News Headline / Article Content:",
            height=200,
            placeholder="Type or paste the news content here...",
        )

        col_btn, _ = st.columns([1, 4])
        with col_btn:
            predict_btn = st.button("🚀 Analyze Authenticity")

        if predict_btn:
            if not news_input.strip():
                st.warning("Please enter text before analyzing.")
            else:
                cleaned = clean_text(news_input)

                if not cleaned:
                    st.warning("The input contains only symbols, numbers, or stopwords. Please provide descriptive text.")
                else:
                    vec = vectorizer.transform([cleaned])
                    prediction = model.predict(vec)[0]
                    confidence = compute_confidence(model, vec, prediction)
                    conf_text = f" (Confidence: {confidence:.2f}%)" if confidence is not None else ""

                    st.markdown("### Verification Result")
                    if prediction == FAKE_LABEL:
                        st.markdown(
                            f'<div class="status-fake">⚠️ FAKE NEWS DETECTED{conf_text}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="status-real">✅ REAL NEWS VERIFIED{conf_text}</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        """
                    <div class="disclaimer-box">
                        <strong>⚠️ Algorithmic Disclaimer:</strong> Machine learning models evaluate linguistic patterns, styling cues, and training data correlations. Predictions are probabilistic and should not replace independent, journalistic fact-checking.
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

# ==========================================================
# PAGE 3: ANALYTICS & EDA
# ==========================================================
elif page == "📊 Analytics & EDA":
    st.markdown('<div class="main-title">📊 Analytics & Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detailed diagnostic comparison across algorithms and exploratory data analysis.</div>',
        unsafe_allow_html=True,
    )

    models_dict = (metrics_data or {}).get("models", {})

    if not models_dict:
        st.info("Metrics not found. Run `python train.py` to populate performance benchmarks.")
    else:
        st.subheader("1. Algorithm Performance Benchmarking")

        metrics_list = [
            {
                "Algorithm": name,
                "Accuracy (%)": m.get("accuracy", 0) * 100,
                "Precision (%)": m.get("precision", 0) * 100,
                "Recall (%)": m.get("recall", 0) * 100,
                "F1-Score (%)": m.get("f1_score", 0) * 100,
            }
            for name, m in models_dict.items()
        ]
        df_metrics = pd.DataFrame(metrics_list)
        numeric_cols = [c for c in df_metrics.columns if c != "Algorithm"]

        st.dataframe(
            df_metrics.style.format({c: "{:.2f}" for c in numeric_cols}).highlight_max(
                axis=0, subset=numeric_cols, color="#D1FAE5"
            ),
            hide_index=True,
        )

        st.markdown("#### Model Metric Comparison")
        fig, ax = plt.subplots(figsize=(9, 4))
        df_melted = df_metrics.melt(id_vars="Algorithm", var_name="Metric", value_name="Score")
        sns.barplot(data=df_melted, x="Algorithm", y="Score", hue="Metric", palette="Blues_r", ax=ax)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score Percentage (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        show_fig(fig)

        st.markdown("---")
        st.subheader("2. Best Model Confusion Matrix")

        best_name = metrics_data.get("best_model_name")
        if best_name not in models_dict:
            st.warning(f"Best model `{best_name}` was not found in the saved metrics.")
        else:
            best = models_dict[best_name]
            col_cm, col_summary = st.columns([1.2, 1])

            with col_cm:
                cm = best.get("confusion_matrix")
                if cm is None:
                    st.info("No confusion matrix stored for this model.")
                else:
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    sns.heatmap(
                        np.array(cm),
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Real (0)", "Fake (1)"],
                        yticklabels=["Real (0)", "Fake (1)"],
                        ax=ax_cm,
                    )
                    ax_cm.set_xlabel("Predicted Label")
                    ax_cm.set_ylabel("True Label")
                    ax_cm.set_title(f"Confusion Matrix: {best_name}")
                    fig_cm.tight_layout()
                    show_fig(fig_cm)

            with col_summary:
                st.markdown(f"**Top Performing Classifier:** `{best_name}`")
                st.markdown(f"- **Accuracy:** `{best.get('accuracy', 0) * 100:.2f}%`")
                st.markdown(f"- **Precision:** `{best.get('precision', 0) * 100:.2f}%`")
                st.markdown(f"- **Recall:** `{best.get('recall', 0) * 100:.2f}%`")
                st.markdown(f"- **F1-Score:** `{best.get('f1_score', 0) * 100:.2f}%`")
                st.caption("F1-score was selected as the deciding benchmark to balance precision and recall.")

# ==========================================================
# PAGE 4: DATASET EXPLORER
# ==========================================================
elif page == "📁 Dataset Explorer":
    st.markdown('<div class="main-title">📁 Dataset Inspection & Profiler</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload a custom CSV dataset or examine the active training data.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload custom CSV news dataset", type=["csv"])

    df_view = None
    if uploaded_file is not None:
        try:
            df_view = pd.read_csv(uploaded_file)
            st.success("Custom CSV successfully loaded for analysis!")
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
    else:
        df_view = load_default_dataset(_mtime(DEFAULT_DATASET_PATH))

    if df_view is not None and not df_view.empty:
        st.dataframe(df_view.head(10))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df_view):,}")
        col2.metric("Features/Columns", len(df_view.columns))
        col3.metric("Missing Fields", f"{int(df_view.isnull().sum().sum()):,}")
        col4.metric("Duplicate Rows", f"{int(df_view.duplicated().sum()):,}")

        st.markdown("---")
        st.subheader("Distribution Visualizations")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if "label" in df_view.columns:
                st.markdown("**Real vs. Fake News Proportions**")
                counts = df_view["label"].value_counts().sort_index()
                names = [LABEL_NAMES.get(i, str(i)) for i in counts.index]
                fig_label, ax_label = plt.subplots(figsize=(6, 4))
                ax_label.bar(names, counts.values, color=["#1E3A8A", "#93C5FD", "#2563EB", "#60A5FA"][: len(counts)])
                ax_label.set_ylabel("Count")
                fig_label.tight_layout()
                show_fig(fig_label)

        with col_g2:
            if "subject" in df_view.columns:
                st.markdown("**Subject / Category Breakdown**")
                fig_subj, ax_subj = plt.subplots(figsize=(6, 4))
                df_view["subject"].value_counts().head(8).sort_values().plot(
                    kind="barh", color="#2563EB", ax=ax_subj
                )
                ax_subj.set_xlabel("Articles")
                fig_subj.tight_layout()
                show_fig(fig_subj)

        if "text" in df_view.columns:
            st.markdown("**Article Character Length Distribution**")
            fig_len, ax_len = plt.subplots(figsize=(10, 3.5))
            df_view["text"].fillna("").astype(str).str.len().plot(
                kind="hist", bins=30, color="#1E3A8A", ax=ax_len
            )
            ax_len.set_xlabel("Character Count")
            fig_len.tight_layout()
            show_fig(fig_len)
    elif df_view is not None:
        st.warning("The dataset is empty.")
    else:
        st.info("No active dataset found. Run `python train.py` to auto-generate a sample dataset or upload a CSV file above.")
