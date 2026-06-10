# =====================================================
# 🏦 AML Risk Detection App (Streamlit + TGNN-style)
# =====================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import torch,pickle
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


#pickle_in = open('model.pkl','rb')
#classifier = pickle.load(pickle_in)
# -----------------------------------------------------
# 1️⃣ Page Config
# -----------------------------------------------------
st.set_page_config(page_title="AML Risk Detection", layout="wide")

# TODO: replace with the deployed Vercel insights dashboard URL
INSIGHTS_DASHBOARD_URL = "https://your-vercel-dashboard.vercel.app"

st.title("🏦 Anti-Money Laundering Risk Detection")
st.markdown(
    "Upload a **transaction CSV** to detect **suspicious accounts** "
    "using a **Temporal Graph Neural Network (TGNN-style)** model."
)

# Demo Preview (shown automatically once assets/demo.gif or assets/demo.mp4 is added)
demo_gif = "assets/demo.gif"
demo_video = "assets/demo.mp4"

if os.path.exists(demo_gif):
    st.image(demo_gif, caption="📺 App Demo", use_container_width=True)
elif os.path.exists(demo_video):
    st.video(demo_video)

# Instructions
with st.expander("📌 How to use this app — Click to expand"):
    st.markdown("""
    ### ✅ CSV File Format
    Your CSV must contain **at least these 3 columns** (any naming is fine):

    | Column Type | Accepted Names |
    |---|---|
    | **Sender** | sender, from, src, origin |
    | **Receiver** | receiver, to, dst, target |
    | **Amount** | amount, value, money, amt |

    ### 📄 Example CSV:
    ```
    sender,receiver,amount
    ACC001,ACC002,5000
    ACC002,ACC003,12000
    ACC003,ACC001,9500
    ```

    ### ⚠️ Rules:
    - File must be **.csv** format
    - Max file size: **200MB**
    - Each row = one transaction
    - Use the **Risk Threshold** slider to adjust sensitivity

    ### 🧠 Optional SAML-D-style Columns
    If your CSV also includes any of these (like the [SAML-D dataset](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)),
    they're automatically used as extra model features:
    `Payment_type`, `Payment_currency`, `Received_currency`,
    `Sender_bank_location`, `Receiver_bank_location`.

    If an `Is_laundering` column is present, it's used to compute
    **real ROC-AUC / Precision@K** instead of synthetic demo labels.
    """)

# -----------------------------------------------------
# 2️⃣ Helper: Auto-detect column names
# -----------------------------------------------------
def find_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key in col.lower():
                return col
    return None

# Match a column by exact name, ignoring case/spaces/underscores.
# Used for optional SAML-D-style columns (e.g. "Payment_type" == "payment type").
def find_optional_column(df, target):
    target_norm = target.lower().replace(" ", "").replace("_", "")
    for col in df.columns:
        if col.lower().replace(" ", "").replace("_", "") == target_norm:
            return col
    return None

# -----------------------------------------------------
# 3️⃣ TGNN-style Model (Simplified & Stable)
# -----------------------------------------------------
class AML_TGNN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16):
        super().__init__()
        self.msg = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [N, 2]
        m = torch.relu(self.msg(x))          # message passing proxy
        m = m.unsqueeze(0)                   # [1, N, hidden]
        out, _ = self.gru(m)                 # temporal modeling
        scores = torch.sigmoid(self.out(out.squeeze(0))).squeeze()
        return scores

# -----------------------------------------------------
# 4️⃣ File Upload
# -----------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.link_button("📊 Open Insights Dashboard", INSIGHTS_DASHBOARD_URL)

    # Auto-detect columns
    sender_col = find_column(df, ["sender", "from", "src", "origin"])
    receiver_col = find_column(df, ["receiver", "to", "dst", "target"])
    amount_col = find_column(df, ["amount", "value", "money", "amt"])

    if not all([sender_col, receiver_col, amount_col]):
        st.error(
            "CSV must contain sender, receiver and amount columns "
            "(any naming allowed)."
        )
        st.stop()

    # -------------------------------------------------
    # 5️⃣ Build Graph
    # -------------------------------------------------
    accounts = pd.unique(df[[sender_col, receiver_col]].values.ravel())
    account_map = {acc: i for i, acc in enumerate(accounts)}
    N = len(accounts)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for _, row in df.iterrows():
        u = account_map[row[sender_col]]
        v = account_map[row[receiver_col]]
        G.add_edge(u, v, weight=row[amount_col])

    # -------------------------------------------------
    # 6️⃣ Node Features (base + optional SAML-D-style columns)
    # -------------------------------------------------
    extra_cols = {
        "Payment Type Diversity": find_optional_column(df, "payment_type"),
        "Sender Currency Diversity": find_optional_column(df, "payment_currency"),
        "Receiver Currency Diversity": find_optional_column(df, "received_currency"),
        "Sender Location Diversity": find_optional_column(df, "sender_bank_location"),
        "Receiver Location Diversity": find_optional_column(df, "receiver_bank_location"),
    }
    extra_cols = {name: col for name, col in extra_cols.items() if col}

    sender_loc_col = find_optional_column(df, "sender_bank_location")
    receiver_loc_col = find_optional_column(df, "receiver_bank_location")
    cross_border = bool(sender_loc_col and receiver_loc_col)

    laundering_col = find_optional_column(df, "is_laundering")

    feature_names = ["Avg Sent Amount", "Avg Received Amount"] + list(extra_cols.keys())
    if cross_border:
        feature_names.append("Cross-Border Ratio")

    if len(feature_names) > 2:
        st.info(f"📊 Extra columns detected — added as features: {', '.join(feature_names[2:])}")

    feature_series = {
        "Avg Sent Amount": df.groupby(sender_col)[amount_col].mean(),
        "Avg Received Amount": df.groupby(receiver_col)[amount_col].mean(),
    }

    for name, col in extra_cols.items():
        feature_series[name] = (
            df.groupby(sender_col)[col].nunique()
            .add(df.groupby(receiver_col)[col].nunique(), fill_value=0)
        )

    if cross_border:
        df["_cross_border"] = (df[sender_loc_col] != df[receiver_loc_col]).astype(int)
        feature_series["Cross-Border Ratio"] = (
            df.groupby(sender_col)["_cross_border"].mean()
            .add(df.groupby(receiver_col)["_cross_border"].mean(), fill_value=0) / 2
        )

    features = torch.zeros(N, len(feature_names))
    for acc, idx in account_map.items():
        for j, name in enumerate(feature_names):
            val = feature_series[name].get(acc, 0)
            features[idx, j] = 0 if pd.isna(val) else val

    features = (features - features.mean(0)) / (features.std(0) + 1e-6)

    # -------------------------------------------------
    # 7️⃣ Model Prediction
    # -------------------------------------------------
    model = AML_TGNN(in_dim=features.shape[1])
    model.eval()

    with torch.no_grad():
        scores = model(features)

    threshold = st.slider("🚨 Risk Threshold", 0.0, 1.0, 0.5)
    preds = (scores > threshold).int()

    # -------------------------------------------------
    # 8️⃣ Metrics
    # -------------------------------------------------
    if laundering_col:
        laundering_accounts = (
            set(df.loc[df[laundering_col] == 1, sender_col])
            | set(df.loc[df[laundering_col] == 1, receiver_col])
        )
        eval_labels = np.array([
            1 if acc in laundering_accounts else 0 for acc in account_map.keys()
        ])
        labels_source = "real `Is_laundering` ground truth"
    else:
        eval_labels = (features[:, 0] + features[:, 1] > 0).int().numpy()
        labels_source = "synthetic demo labels"

    def precision_at_k(y_true, y_score, k):
        idx = np.argsort(y_score)[::-1][:k]
        return y_true[idx].mean()

    st.subheader("📊 Model Performance")
    st.caption(f"Evaluation labels: {labels_source}")

    if len(set(eval_labels)) < 2:
        st.warning("Cannot compute AUC/Precision@K — evaluation labels contain only one class.")
    else:
        auc = roc_auc_score(eval_labels, scores.numpy())
        st.metric("ROC-AUC", f"{auc:.3f}")

        for k in [5, 10, 20]:
            st.metric(f"Precision@{k}", f"{precision_at_k(eval_labels, scores.numpy(), k):.3f}")

    # -------------------------------------------------
    # 9️⃣ Transaction Graph Visualization
    # -------------------------------------------------
    st.subheader("🕸️ Transaction Graph with AML Risk Highlighting")

    pos = nx.spring_layout(G, seed=42)

    node_colors = [
        "red" if preds[n] == 1 else "skyblue"
        for n in G.nodes()
    ]

    edge_widths = [
        G[u][v]["weight"] / df[amount_col].max() * 3
        for u, v in G.edges()
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(
        G, pos,
        node_color=node_colors,
        node_size=600,
        edge_color="gray",
        width=edge_widths,
        with_labels=True,
        ax=ax
    )

    ax.set_title("Transaction Graph with AML Risk Highlighting")
    st.pyplot(fig)

    # -------------------------------------------------
    # 🔟 Suspicious Accounts Table
    # -------------------------------------------------
    st.subheader("🚩 Flagged Accounts")

    suspicious = [
        {"Account": acc, "Risk Score": float(scores[i])}
        for acc, i in account_map.items()
        if preds[i] == 1
    ]

    if suspicious:
        st.dataframe(pd.DataFrame(suspicious))
    else:
        st.success("No suspicious accounts detected.")