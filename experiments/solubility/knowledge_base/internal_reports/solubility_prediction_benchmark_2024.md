---
title: "Aqueous Solubility Prediction: ML Benchmarking Study"
document_id: RPT-2024-DMPK-0083
version: 1.4
date: 2024-11-10
author: Dr. Antoine Mercier, DMPK Modeling Group
department: R&D Informatics — DMPK
classification: Internal Use Only
---

# Aqueous Solubility Prediction: ML Benchmarking Study

## Executive Summary

Aqueous solubility (logS) is a critical ADMET property governing oral bioavailability. This report benchmarks ML approaches on our internal solubility dataset (N=3,412 compounds, shake-flask method) and the public ESOL dataset (N=1,128, Delaney 2004). Key finding: gradient boosting with Morgan fingerprints + selected RDKit descriptors achieves RMSE 0.65, outperforming the General Solubility Equation (GSE) baseline of 1.01.

## Dataset Characteristics

Internal dataset: 3,412 compounds measured by shake-flask method at pH 7.4, 25°C. Distribution: mean logS = -3.2, std = 1.8, range [-8.5, 1.2]. The ESOL dataset (1,128 compounds) was used for external validation and comparison with published methods.

Key challenges:
- Bimodal distribution — highly soluble fragments vs. poorly soluble drug candidates
- Measurement noise — shake-flask reproducibility ±0.3 logS units
- Limited representation of salts and cocrystals

## Descriptor Importance for Solubility

Feature importance analysis (SHAP values) identified the following key predictors:

| Descriptor | SHAP Importance | Relationship | Physical Basis |
|------------|-----------------|--------------|----------------|
| MolLogP | 0.185 | Higher LogP → lower solubility | Lipophilicity opposes dissolution |
| MolWt | 0.142 | Higher MW → lower solubility | Crystal lattice energy |
| NumAromaticRings | 0.098 | More rings → lower solubility | Planar stacking, crystal packing |
| TPSA | 0.087 | Higher TPSA → higher solubility | Polar groups favor hydration |
| NumRotatableBonds | 0.072 | More rotatable bonds → higher solubility | Disrupts crystal packing |
| FractionCSP3 | 0.065 | Higher sp3 fraction → higher solubility | 3D shape disrupts stacking |
| NumHDonors | 0.058 | Complex — U-shaped | H-bonds aid hydration but also crystal |
| BertzCT | 0.041 | Higher complexity → lower solubility | Correlated with MW and ring count |

## Model Comparison

All models: stratified 5-fold CV, 3 repeats. ESOL dataset used for external validation.

| Model | Features | RMSE (Internal) | RMSE (ESOL) | R² | Train Time |
|-------|----------|-----------------|-------------|-----|-----------|
| GSE (Yalkowsky) | LogP + MP | 1.01 | 1.15 | 0.62 | — |
| LinearRegression | Morgan 2048 | 0.92 | 1.08 | 0.71 | 2s |
| RandomForest | Morgan 2048 | 0.78 | 0.89 | 0.81 | 35s |
| RandomForest | Morgan + 20 desc | 0.71 | 0.82 | 0.85 | 42s |
| XGBoost | Morgan + 20 desc | 0.67 | 0.76 | 0.87 | 30s |
| HistGradientBoosting | Morgan + 20 desc | 0.65 | 0.74 | 0.88 | 18s |
| GCN (3-layer) | Molecular graph | 0.72 | 0.85 | 0.84 | 8min |
| Ensemble (HGB + XGB + RF) | Morgan + MACCS + 20 desc | 0.61 | 0.70 | 0.90 | 95s |

## Key Findings

1. **HistGradientBoosting is the best single model** for solubility prediction — faster and slightly better than XGBoost on our internal data. Scikit-learn native, no extra dependency.

2. **LogP is the single most important feature**, consistent with the thermodynamic relationship between lipophilicity and aqueous solubility (logS ≈ 0.5 - logP, from GSE).

3. **Aromatic ring count is underrated** — adding `NumAromaticRings` and `AromaticProportion` as explicit features improved all models by 0.03-0.05 RMSE, beyond what fingerprints capture.

4. **Ensemble provides diminishing returns**: HGB+XGB+RF ensemble gains 0.04 RMSE over HGB alone at 5x training cost. Acceptable for production but not for rapid iteration.

5. **GNNs underperform** gradient boosting on this dataset size. With N>10K compounds, GNNs may become competitive.

## Recommendations

1. Deploy HistGradientBoosting with Morgan (2048) + top-20 descriptors as the production model.
2. Include `AromaticProportion` (aromatic atoms / total heavy atoms) as an explicit feature.
3. Apply log-transform awareness — predictions should respect the logS scale and physical bounds.
4. Scaffold-split validation is essential — random splits overestimate performance by ~0.1 RMSE.
5. Flag predictions for compounds with MW > 700 or logP > 6 as out-of-domain.
