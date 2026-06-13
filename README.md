#  Anti-Money Laundering using Temporal Graph Neural Networks (TGNN)

##  Project Overview

This project focuses on **Anti-Money Laundering (AML)** by modeling banking transactions as a **dynamic graph** and identifying suspicious accounts using a **Temporal Graph Neural Network (TGNN)**.
Unlike traditional rule-based systems, the model learns **evolving transaction relationships over time**, enabling more accurate detection of complex money-laundering patterns.

The system is implemented using **PyTorch** and deployed as an interactive **Streamlit web application**.

---

## Key Features

*  Transaction modeling as a **directed temporal graph**
*  TGNN-style **message passing + GRU temporal learning**
*  Detection of **suspicious accounts**
*  Model evaluation using **AUC** and **Precision@K**

---

##  Model Architecture (TGNN-Style)

1. **Graph Construction**

   * Nodes → Bank accounts
   * Edges → Transactions (sender → receiver)

2. **Message Passing**

   * Each account aggregates information from neighboring accounts

3. **Temporal Modeling**

   * A **GRU** captures how transaction behavior evolves over time

4. **Prediction**

   * Binary classification: **Normal / Suspicious**

---

## Evaluation Metrics

* **AUC (Area Under ROC Curve)** – overall discrimination capability
* **Precision@K** – effectiveness of detecting top-K risky accounts

These metrics are displayed directly in the dashboard.

---

## Dataset

* Expected CSV format:

  ```csv
  sender,receiver,amount,timestamp
  ```

* Additional public datasets for training/testing:

  * [IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
  * [Synthetic Transaction Monitoring Dataset AML](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)

---

##  Streamlit Application

The web app allows users to:

* Upload a banking transaction dataset
* Visualize transaction graphs
* View suspicious accounts
* Inspect AUC and Precision@K scores

###  Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Insights Dashboard (Next.js)

A companion analytics dashboard for exploring AML transaction data — built with **Next.js**, **Tailwind CSS**, and **Recharts**.

* **Live demo:** [insights-dashboard-indol.vercel.app](https://insights-dashboard-indol.vercel.app)
* CSV files are parsed and analyzed **entirely in the browser** (via a Web Worker) — nothing is uploaded to a server, so even multi-gigabyte files work
* Sections: Overview, Transaction Graph, Fraud Accounts, Transactions
* Supports optional SAML-D-style columns (`Payment_type`, `Payment_currency`, `Sender_bank_location`, `Receiver_bank_location`, `Is_laundering`) as extra risk-scoring features

### Run Locally

```bash
cd insights-dashboard
npm install
npm run dev
```

---

## Project Structure

```bash
aml_tgnn_streamlit/
│
├── app.py                 # Streamlit AML application
├── requirements.txt       # Project dependencies
├── templates/             # UI templates
├── uploads/               # Uploaded datasets (ignored in Git)
├── model.pkl              # Saved TGNN model
├── insights-dashboard/    # Next.js companion dashboard (deployed on Vercel)
├── README.md              # Project documentation
└── .gitignore
```

---

## Technologies Used

* **Python**
* **PyTorch**
* **NetworkX**
* **Pandas / NumPy**
* **Streamlit**
* **Scikit-learn (metrics)**

---

##  Use Cases

* Banking AML monitoring
* Fraud detection
* Financial risk analysis
* Graph-based anomaly detection

---

##  Future Enhancements

* Integration with **Graph Transformers**
* Real-time transaction streaming
* Explainable AI (XAI) for AML decisions
* Scalability with distributed graph processing

---

## Author

**Basabraj Biswas**

 GitHub: [https://github.com/basabraj](https://github.com/basabraj)
