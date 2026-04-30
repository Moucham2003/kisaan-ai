"""
crop_accuracy_report.py  —  Kisaan AI
======================================
Run with:  streamlit run crop_accuracy_report.py
Generates a full accuracy analysis dashboard for the Crop Recommendation model.
"""

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Model Accuracy Report 📊",
    page_icon="📊",
    layout="wide"
)

# ─── Color Palette ─────────────────────────────────────────────────────────────
GREEN  = "#2e7d32"
LIGHT  = "#a5d6a7"
AMBER  = "#f59e0b"
RED    = "#ef4444"
BG     = "#f0fdf4"

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    crop_model   = joblib.load("crop_model.pkl")
    crop_encoder = joblib.load("crop_encoder.pkl")
    return crop_model, crop_encoder

@st.cache_data
def build_dataset():
    """
    Reconstruct a representative test dataset using the known feature
    distributions of the standard Crop Recommendation dataset
    (22 crops × 100 samples = 2 200 rows).
    """
    np.random.seed(42)
    crop_params = {
        "rice":        dict(N=(60,100),  P=(40,60),   K=(40,60),   T=(20,27),  H=(80,90),  pH=(5.5,7.0), R=(150,300)),
        "maize":       dict(N=(60,100),  P=(55,75),   K=(55,75),   T=(18,27),  H=(55,75),  pH=(5.5,7.5), R=(60,120)),
        "chickpea":    dict(N=(35,55),   P=(65,85),   K=(75,95),   T=(15,25),  H=(14,25),  pH=(5.5,7.5), R=(65,100)),
        "kidneybeans": dict(N=(15,25),   P=(60,80),   K=(18,25),   T=(18,28),  H=(18,25),  pH=(5.5,7.5), R=(100,150)),
        "pigeonpeas":  dict(N=(15,25),   P=(65,85),   K=(18,25),   T=(18,28),  H=(40,60),  pH=(5.5,7.0), R=(100,150)),
        "mothbeans":   dict(N=(15,25),   P=(40,60),   K=(18,25),   T=(25,35),  H=(40,55),  pH=(3.5,7.0), R=(40,60)),
        "mungbean":    dict(N=(15,25),   P=(40,60),   K=(18,25),   T=(25,35),  H=(80,90),  pH=(6.0,7.5), R=(40,60)),
        "blackgram":   dict(N=(30,45),   P=(55,75),   K=(14,22),   T=(25,35),  H=(60,80),  pH=(6.0,7.5), R=(60,80)),
        "lentil":      dict(N=(15,25),   P=(65,85),   K=(18,25),   T=(15,25),  H=(60,75),  pH=(5.5,7.5), R=(35,60)),
        "pomegranate": dict(N=(15,25),   P=(10,20),   K=(38,50),   T=(21,35),  H=(85,95),  pH=(5.5,7.5), R=(100,130)),
        "banana":      dict(N=(85,115),  P=(65,85),   K=(38,55),   T=(25,35),  H=(75,90),  pH=(5.5,7.0), R=(75,125)),
        "mango":       dict(N=(15,25),   P=(15,25),   K=(28,40),   T=(27,38),  H=(45,60),  pH=(4.5,7.0), R=(90,120)),
        "grapes":      dict(N=(15,25),   P=(120,150), K=(185,215), T=(8,18),   H=(70,85),  pH=(5.5,7.5), R=(60,80)),
        "watermelon":  dict(N=(95,115),  P=(8,16),    K=(48,58),   T=(25,38),  H=(80,92),  pH=(6.0,7.0), R=(40,60)),
        "muskmelon":   dict(N=(95,115),  P=(8,16),    K=(48,58),   T=(28,38),  H=(90,97),  pH=(6.0,7.0), R=(20,30)),
        "apple":       dict(N=(0,20),    P=(120,150), K=(195,215), T=(0,12),   H=(90,97),  pH=(5.5,7.0), R=(100,125)),
        "orange":      dict(N=(0,20),    P=(10,20),   K=(8,18),    T=(10,18),  H=(90,97),  pH=(6.0,7.5), R=(100,135)),
        "papaya":      dict(N=(45,60),   P=(55,65),   K=(42,52),   T=(33,43),  H=(90,97),  pH=(6.5,7.5), R=(125,165)),
        "coconut":     dict(N=(18,28),   P=(15,25),   K=(28,40),   T=(25,35),  H=(90,97),  pH=(5.0,7.5), R=(130,175)),
        "cotton":      dict(N=(100,145), P=(16,25),   K=(18,28),   T=(23,35),  H=(75,85),  pH=(6.0,8.0), R=(60,80)),
        "jute":        dict(N=(60,90),   P=(40,60),   K=(38,50),   T=(23,30),  H=(65,75),  pH=(6.0,7.5), R=(150,200)),
        "coffee":      dict(N=(95,115),  P=(25,35),   K=(28,38),   T=(23,28),  H=(55,70),  pH=(6.0,7.0), R=(150,230)),
    }

    X_list, y_list = [], []
    for crop, p in crop_params.items():
        n = 100
        rows = np.column_stack([
            np.random.uniform(*p["N"],  n),
            np.random.uniform(*p["P"],  n),
            np.random.uniform(*p["K"],  n),
            np.random.uniform(*p["T"],  n),
            np.random.uniform(*p["H"],  n),
            np.random.uniform(*p["pH"], n),
            np.random.uniform(*p["R"],  n),
        ])
        X_list.extend(rows)
        y_list.extend([crop] * n)

    return np.array(X_list), np.array(y_list)


crop_model, crop_encoder = load_model()
X, y_labels = build_dataset()
y_enc  = crop_encoder.transform(y_labels)
preds  = crop_model.predict(X)
proba  = crop_model.predict_proba(X)        # shape (2200, 22)
classes = list(crop_encoder.classes_)

overall_acc = accuracy_score(y_enc, preds)

# Per-class accuracy
per_class_acc = {}
for i, cls in enumerate(classes):
    mask = y_enc == i
    per_class_acc[cls] = accuracy_score(y_enc[mask], preds[mask])

# Feature importances
feature_names = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
                 "Temperature", "Humidity", "Soil pH", "Rainfall"]
importances = crop_model.feature_importances_

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Crop Recommendation — Accuracy Report")
st.markdown("#### XGBoost Model · 22 Crops · Kisaan AI")
st.markdown("---")

# ─── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Overall Accuracy",    f"{overall_acc*100:.1f}%")
k2.metric("Crops Supported",     len(classes))
k3.metric("Model",               "XGBoost")
k4.metric("Features Used",       len(feature_names))

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Per-Class Accuracy Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🌿 Per-Class Accuracy")
st.caption("How accurately the model identifies each crop type.")

sorted_acc = dict(sorted(per_class_acc.items(), key=lambda x: x[1]))
crops_sorted = list(sorted_acc.keys())
vals_sorted  = list(sorted_acc.values())
bar_colors   = [GREEN if v >= 0.85 else AMBER if v >= 0.60 else RED for v in vals_sorted]

fig1, ax1 = plt.subplots(figsize=(12, 6))
bars = ax1.barh(crops_sorted, [v * 100 for v in vals_sorted],
                color=bar_colors, edgecolor="white", linewidth=0.6)

# Value labels
for bar, val in zip(bars, vals_sorted):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{val*100:.0f}%", va="center", fontsize=9, color="#333")

ax1.axvline(overall_acc * 100, color="#1565c0", linewidth=1.8,
            linestyle="--", label=f"Overall avg ({overall_acc*100:.1f}%)")
ax1.set_xlim(0, 112)
ax1.set_xlabel("Accuracy (%)", fontsize=11)
ax1.set_title("Per-Crop Prediction Accuracy", fontsize=13, fontweight="bold")

legend_patches = [
    mpatches.Patch(color=GREEN, label="≥ 85% (High)"),
    mpatches.Patch(color=AMBER, label="60–84% (Medium)"),
    mpatches.Patch(color=RED,   label="< 60% (Low)"),
]
ax1.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color="#1565c0", lw=1.8, linestyle="--",
               label=f"Overall avg ({overall_acc*100:.1f}%)")
], fontsize=9)
ax1.grid(axis="x", alpha=0.3)
fig1.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔍 Feature Importance")
st.caption("Which soil/climate inputs matter most for the prediction.")

fi_sorted_idx = np.argsort(importances)
fi_sorted     = importances[fi_sorted_idx]
fn_sorted     = [feature_names[i] for i in fi_sorted_idx]
fi_colors     = [plt.cm.Greens(0.4 + 0.6 * v / max(importances)) for v in fi_sorted]

fig2, ax2 = plt.subplots(figsize=(9, 5))
bars2 = ax2.barh(fn_sorted, fi_sorted * 100, color=fi_colors, edgecolor="white")
for bar, val in zip(bars2, fi_sorted):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{val*100:.1f}%", va="center", fontsize=10)

ax2.set_xlabel("Importance (%)", fontsize=11)
ax2.set_title("Feature Importance (XGBoost)", fontsize=13, fontweight="bold")
ax2.set_xlim(0, max(fi_sorted) * 130)
ax2.grid(axis="x", alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Confusion Matrix (top-level heatmap)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🗺️ Confusion Matrix")
st.caption("Rows = actual crop · Columns = predicted crop. Diagonal = correct predictions.")

cm = confusion_matrix(y_enc, preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig3, ax3 = plt.subplots(figsize=(14, 11))
sns.heatmap(
    cm_norm,
    annot=True, fmt=".0%", linewidths=0.4,
    cmap="Greens", ax=ax3,
    xticklabels=classes, yticklabels=classes,
    annot_kws={"size": 7},
    cbar_kws={"label": "Row-normalised accuracy"}
)
ax3.set_xlabel("Predicted Crop", fontsize=11)
ax3.set_ylabel("Actual Crop", fontsize=11)
ax3.set_title("Confusion Matrix (row-normalised)", fontsize=13, fontweight="bold")
ax3.tick_params(axis="x", rotation=45, labelsize=8)
ax3.tick_params(axis="y", rotation=0,  labelsize=8)
fig3.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Confidence Distribution
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📈 Prediction Confidence Distribution")
st.caption("How confident the model is when it makes a prediction (max softmax probability).")

max_conf = proba.max(axis=1) * 100   # confidence for the top predicted class
correct  = (preds == y_enc)

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.hist(max_conf[correct],  bins=30, color=GREEN, alpha=0.75, label="Correct predictions")
ax4.hist(max_conf[~correct], bins=30, color=RED,   alpha=0.75, label="Wrong predictions")
ax4.axvline(max_conf.mean(), color="navy", linestyle="--", linewidth=1.5,
            label=f"Mean confidence ({max_conf.mean():.1f}%)")
ax4.set_xlabel("Confidence (%)", fontsize=11)
ax4.set_ylabel("Number of Samples", fontsize=11)
ax4.set_title("Model Confidence: Correct vs Incorrect Predictions", fontsize=13, fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)
fig4.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Accuracy Gauge (donut)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
c_gauge, c_prec = st.columns(2)

with c_gauge:
    st.subheader("🎯 Overall Accuracy Gauge")
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    sizes  = [overall_acc * 100, (1 - overall_acc) * 100]
    colors = [GREEN, "#e0e0e0"]
    wedges, _ = ax5.pie(sizes, colors=colors, startangle=90,
                        wedgeprops=dict(width=0.45, edgecolor="white"))
    ax5.text(0, 0, f"{overall_acc*100:.1f}%", ha="center", va="center",
             fontsize=26, fontweight="bold", color=GREEN)
    ax5.text(0, -0.28, "Overall Accuracy", ha="center", va="center",
             fontsize=11, color="#555")
    ax5.set_title("Crop Recommendation Model", fontsize=11, pad=10)
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Precision / Recall bar per class
# ══════════════════════════════════════════════════════════════════════════════
with c_prec:
    st.subheader("📋 Precision vs Recall (per crop)")
    report = classification_report(y_enc, preds, target_names=classes, output_dict=True)
    prec_vals = [report[c]["precision"] for c in classes]
    rec_vals  = [report[c]["recall"]    for c in classes]

    x = np.arange(len(classes))
    w = 0.4
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    ax6.bar(x - w/2, prec_vals, w, label="Precision", color="#1565c0", alpha=0.8)
    ax6.bar(x + w/2, rec_vals,  w, label="Recall",    color=GREEN,     alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(classes, rotation=90, fontsize=7)
    ax6.set_ylim(0, 1.1)
    ax6.set_ylabel("Score")
    ax6.set_title("Precision & Recall by Crop", fontsize=11, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.3)
    fig6.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

# ─── Raw Report Table ──────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📄 Full Classification Report (click to expand)"):
    import pandas as pd
    rows = []
    for c in classes:
        rows.append({
            "Crop":      c.title(),
            "Precision": f"{report[c]['precision']*100:.1f}%",
            "Recall":    f"{report[c]['recall']*100:.1f}%",
            "F1-Score":  f"{report[c]['f1-score']*100:.1f}%",
            "Support":   int(report[c]["support"]),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Kisaan AI 🌾 · Crop Accuracy Report · XGBoost Classifier</center>",
    unsafe_allow_html=True
)
