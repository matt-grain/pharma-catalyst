---
title: "Clinical Toxicity Prediction: ML Model Benchmarking Study"
document_id: RPT-2024-TOX-0029
version: 1.1
date: 2024-10-05
author: Dr. Sophie Delacroix, Computational Toxicology Group
department: Drug Safety Sciences — Predictive Toxicology
classification: Internal Use Only
---

# Clinical Toxicity Prediction: ML Model Benchmarking Study

## Executive Summary

This report evaluates ML approaches for predicting clinical trial toxicity using the ClinTox dataset (N=1,484 compounds, FDA-approved drugs and those that failed clinical trials). The task is binary classification: compounds that caused toxicity-related clinical trial failures vs. approved drugs. Key challenge: extreme class imbalance (only 5.3% positive/toxic). Best result: MLP neural network with Morgan fingerprints achieves ROC-AUC 0.82 with SMOTE oversampling.

## Dataset Characteristics

ClinTox contains two tasks — we focus on CT_TOX (clinical trial toxicity):
- 1,484 compounds total
- 79 toxic (5.3%), 1,405 non-toxic (94.7%)
- Extreme class imbalance requires careful handling

Class imbalance mitigation strategies evaluated:

| Strategy | ROC-AUC | Precision | Recall | Notes |
|----------|---------|-----------|--------|-------|
| No balancing | 0.71 | 0.12 | 0.65 | High recall but useless precision |
| Class weights | 0.76 | 0.18 | 0.72 | Better but still poor precision |
| SMOTE oversampling | 0.82 | 0.25 | 0.78 | Best balance |
| Random undersampling | 0.79 | 0.21 | 0.81 | High recall, loses information |
| ADASYN | 0.80 | 0.23 | 0.76 | Similar to SMOTE |

## Key Toxicity Descriptors

Feature importance for toxicity prediction differs from other ADMET tasks:

| Descriptor | Importance | Toxicity Signal |
|------------|------------|-----------------|
| NumHAcceptors | 0.112 | Many acceptors → metabolic liability |
| MolWt | 0.098 | High MW → more off-target binding |
| NumAromaticRings | 0.091 | Aromatic amines → genotoxicity alerts |
| TPSA | 0.078 | Very low TPSA → hERG liability |
| MolLogP | 0.075 | High LogP → phospholipidosis risk |
| NumRotatableBonds | 0.062 | High flexibility → promiscuous binding |
| fr_NH2 | 0.058 | Primary amines → reactive metabolites |
| BertzCT | 0.045 | High complexity → unpredictable metabolism |

## Model Comparison

Stratified 5-fold CV with SMOTE applied inside each fold (no data leakage).

| Model | Features | ROC-AUC | Precision | Recall | F1 | Notes |
|-------|----------|---------|-----------|--------|-----|-------|
| LogisticRegression | Morgan 1024 | 0.72 | 0.15 | 0.68 | 0.25 | baseline |
| RandomForest | Morgan 2048 | 0.77 | 0.19 | 0.71 | 0.30 | moderate |
| XGBoost | Morgan 2048 + desc | 0.80 | 0.22 | 0.74 | 0.34 | good |
| MLP (128-64-1) | Morgan 1024 | 0.82 | 0.25 | 0.78 | 0.38 | best AUC |
| MLP (128-64-1) | Morgan 2048 + desc | 0.81 | 0.24 | 0.75 | 0.36 | desc didn't help |
| GCN (3-layer) | Molecular graph | 0.78 | 0.20 | 0.72 | 0.31 | underfits |

## Key Findings

1. **Neural networks outperform tree-based models** on clinical toxicity — the MLP captures non-linear toxicity patterns that gradient boosting misses. This is opposite to solubility/BBB where tree models dominate.

2. **SMOTE inside CV folds is critical** — applying SMOTE before splitting causes data leakage and inflates AUC by ~0.05. Always oversample inside the CV loop.

3. **Descriptors add less value than expected** — Morgan fingerprints alone (1024 bits) perform nearly as well as fingerprints + descriptors. Toxicity is driven by specific substructural motifs that fingerprints capture well.

4. **Precision remains poor** across all models (max 0.25). This is expected with 5.3% prevalence — even a good model produces many false positives. The model is useful for flagging, not for definitive toxicity prediction.

5. **Structural alerts complement ML** — known toxic substructures (aromatic amines, Michael acceptors, epoxides) should be flagged separately alongside model predictions.

## Recommendations

1. Deploy MLP (128-64-1) with Morgan 1024 + SMOTE as the primary toxicity flag model.
2. Use the model as a **screening filter** (high sensitivity) not a **diagnostic** (would need high specificity).
3. Combine ML predictions with structural alert screening (Derek Nexus or in-house rules).
4. Report predictions as risk tiers: Low (p < 0.2), Medium (0.2-0.5), High (p > 0.5) rather than binary.
5. Calibrate probability outputs — current model is poorly calibrated due to SMOTE. Platt scaling on a held-out non-SMOTE validation set is required.
