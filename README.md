# Transaction Integrity Engine

> A collection of machine learning models for automating, validating, and forecasting financial transactions in enterprise finance workflows.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

---

## Overview

Modern enterprise finance is drowning in transactional data — bills, purchase orders, bank statements, ledger entries. The **Transaction Integrity Engine** is a reference set of ML models that streamline these workflows, surface anomalies, and forecast cash positions, so finance teams can shift from manual reconciliation to oversight.

The repository is organized as bite-sized, readable Python scripts — each one a self-contained example of a classic finance ML problem.

## What it covers

| Category | Use case | Technique |
|---|---|---|
| **Reconciliation & Matching** | Bank reconciliation | Random Forest classifier |
| | Match purchase orders to bills | Classification |
| **Anomaly Detection** | Outlier detection in bills | Local Outlier Factor |
| | Outlier detection in purchase orders | Local Outlier Factor |
| **Forecasting** | Cash flow forecasting | ARIMA |
| | Cash flow forecasting | Generalized Additive Models (GAM) |
| | Product demand forecasting | Graph Neural Networks (GAN) |
| **Automation** | Accounts payable end-to-end | Multi-model pipeline |
| **Intelligent Assistance** | Financial audit recommender | Recommender system |
| | Bill coding feedback loop | Reinforcement learning |

## Why it matters

- **Computer vision** can read scanned bills and extract structured fields.
- **Classifiers** match transactions and predict account codes.
- **Anomaly detection** flags suspicious entries across GL, AP, expenses, and payroll.
- **Time-series and GAM models** forecast cash positions and predict payment timing.
- **Graph neural networks** model financial entities and relationships — useful for both demand forecasting and detecting illicit transaction patterns (e.g., money laundering).

## End-to-end example: Accounts Payable

Multiple models can be chained into a single workflow:

1. Scan a bill — extract data via OCR / computer vision.
2. Predict the account coding via a classifier.
3. Run anomaly detection on the bill.
4. Surface to a human for validation.
5. Pay the bill.
6. Reconcile the payment against the matching bank debit.

See [Automating_account_payables](Automating_account_payables) for a starter implementation.

## Repository layout

```
.
├── Bank_reconciliation                        # RandomForest classifier
├── Matching_PO_to_Bills                       # PO ↔ bill matching
├── Outlier_detection_Bill                     # LOF-based anomaly detection
├── Outlier_detection_PO                       # LOF-based anomaly detection
├── Forecasting_cash_flows_ARIMA               # ARIMA time-series
├── Forecasting_cash_flows_GAM                 # GAM-based forecast
├── GAN_product_demand                         # GNN for demand forecasting
├── Automating_account_payables                # End-to-end AP pipeline
├── Recommender_system_for_financial_audit     # Audit recommender
├── RL_user_feedback_bill                      # RL feedback loop

```

## Getting started

### Requirements

- Python 3.8 or higher
- `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `pmdarima`, `pygam`, `torch`, `torch-geometric`

### Quick install

```bash
pip install pandas numpy scikit-learn statsmodels pmdarima pygam torch torch-geometric
```

### Run a model

Each script expects a CSV input (paths are referenced inside the script — adjust as needed):

```bash
python Bank_reconciliation
python Outlier_detection_Bill
python Forecasting_cash_flows_ARIMA
```

> **Note:** The scripts are reference implementations meant to be adapted to your data. Column names, file paths, and hyperparameters should be tuned for your own datasets.

## Roadmap

- [ ] Add `.py` extensions and standardize CLI entry points
- [ ] Sample datasets for each model
- [ ] Unit tests + CI
- [ ] Docker image with all dependencies
- [ ] Notebook walkthroughs

