# -----------------------------------------------------
# 🏦 AML Risk Detection Flask App (TGNN-style)
# -----------------------------------------------------

from flask import Flask, render_template, request
import os
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
from werkzeug.utils import secure_filename
import numpy as np

# -----------------------------------------------------
# Flask Configuration
# -----------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cpu")

# -----------------------------------------------------
# Column Utilities
# -----------------------------------------------------
def normalize_columns(df):
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )
    return df

def find_column(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None

# -----------------------------------------------------
# TGNN-style Model (FIXED)
# -----------------------------------------------------
class AML_TGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.msg = nn.Linear(2, 16)
        self.gru = nn.GRU(16, 16, batch_first=True)
        self.classifier = nn.Linear(16, 2)

    def forward(self, node_features, graph):
        num_nodes = node_features.size(0)
        messages = torch.zeros(num_nodes, 16, device=node_features.device)

        # Message Passing
        for node in sorted(graph.nodes()):
            neighbors = list(graph.predecessors(node))
            if neighbors:
                agg = node_features[neighbors].mean(dim=0)
            else:
                agg = node_features[node]
            messages[node] = self.msg(agg)

        # Each node = one sequence step
        messages = messages.unsqueeze(1)  # (num_nodes, 1, 16)

        out, _ = self.gru(messages)        # GRU manages hidden internally
        logits = self.classifier(out.squeeze(1))

        return logits

# Load model
model = AML_TGNN().to(device)
model.eval()

# -----------------------------------------------------
# Routes
# -----------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="Please upload a CSV file.")

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        # Load CSV
        df = pd.read_csv(path)
        df = normalize_columns(df)
        cols = df.columns.tolist()

        sender_col = find_column(
            ["sender", "from_account", "src_account", "debit_account"],
            cols
        )
        receiver_col = find_column(
            ["receiver", "to_account", "dst_account", "credit_account"],
            cols
        )
        amount_col = find_column(
            ["amount", "txn_amount", "transaction_amount", "value"],
            cols
        )

        if not sender_col or not receiver_col or not amount_col:
            return render_template(
                "index.html",
                error="CSV must contain sender, receiver and amount columns."
            )

        df = df.rename(columns={
            sender_col: "sender",
            receiver_col: "receiver",
            amount_col: "amount"
        })

        # Build Graph
        accounts = list(set(df["sender"]).union(set(df["receiver"])))
        acc_map = {a: i for i, a in enumerate(accounts)}
        num_nodes = len(accounts)

        features = torch.zeros(num_nodes, 2, device=device)

        for acc in accounts:
            idx = acc_map[acc]
            sent = df[df["sender"] == acc]["amount"].mean()
            recv = df[df["receiver"] == acc]["amount"].mean()
            features[idx] = torch.tensor([
                0 if np.isnan(sent) else sent,
                0 if np.isnan(recv) else recv
            ])

        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for _, row in df.iterrows():
            u = acc_map[row["sender"]]
            v = acc_map[row["receiver"]]
            G.add_edge(u, v, weight=row["amount"])

        # Inference
        with torch.no_grad():
            logits = model(features, G)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        suspicious_accounts = [
            accounts[i] for i in range(num_nodes) if preds[i] == 1
        ]

        return render_template(
            "index.html",
            result="Suspicious Accounts Detected",
            suspicious_accounts=suspicious_accounts,
            confidence=float(torch.max(probs).item()),
            model_name="TGNN-style AML Model"
        )

    except Exception as e:
        return render_template("index.html", error=f"Prediction failed: {e}")

# -----------------------------------------------------
# Run App
# -----------------------------------------------------
if __name__ == "__main__":
    print("🚀 AML Flask App running at http://127.0.0.1:5000")
    app.run(debug=True)
