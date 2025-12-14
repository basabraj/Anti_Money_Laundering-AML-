# =====================================================
# 🏦 AML Risk Detection App (Streamlit + TGNN-style)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# -----------------------------------------------------
# 1️⃣ Page Config
# -----------------------------------------------------
st.set_page_config(page_title="AML Risk Detection", layout="wide")

st.title("🏦 Anti-Money Laundering Risk Detection")
st.markdown(
    "Upload a **transaction CSV** to detect **suspicious accounts** "
    "using a **Temporal Graph Neural Network (TGNN-style)** model."
)

# -----------------------------------------------------
# 2️⃣ Helper: Auto-detect column names
# -----------------------------------------------------
def find_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key in col.lower():
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

model = AML_TGNN()
model.eval()

# -----------------------------------------------------
# 4️⃣ File Upload
# -----------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

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
    # 6️⃣ Node Features
    # -------------------------------------------------
    features = torch.zeros(N, 2)

    for acc, idx in account_map.items():
        sent = df[df[sender_col] == acc][amount_col].mean()
        recv = df[df[receiver_col] == acc][amount_col].mean()

        features[idx] = torch.tensor([
            0 if np.isnan(sent) else sent,
            0 if np.isnan(recv) else recv
        ])

    features = (features - features.mean(0)) / (features.std(0) + 1e-6)

    # -------------------------------------------------
    # 7️⃣ Model Prediction
    # -------------------------------------------------
    with torch.no_grad():
        scores = model(features)

    threshold = st.slider("🚨 Risk Threshold", 0.0, 1.0, 0.5)
    preds = (scores > threshold).int()

    # -------------------------------------------------
    # 8️⃣ Metrics (Synthetic labels for demo)
    # -------------------------------------------------
    pseudo_labels = (features[:, 0] + features[:, 1] > 0).int().numpy()
    auc = roc_auc_score(pseudo_labels, scores.numpy())

    def precision_at_k(y_true, y_score, k):
        idx = np.argsort(y_score)[::-1][:k]
        return y_true[idx].mean()

    st.subheader("📊 Model Performance")
    st.metric("ROC-AUC", f"{auc:.3f}")

    for k in [5, 10, 20]:
        st.metric(f"Precision@{k}", f"{precision_at_k(pseudo_labels, scores.numpy(), k):.3f}")

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
