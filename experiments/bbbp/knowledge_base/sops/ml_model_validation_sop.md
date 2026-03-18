---
title: "Standard Operating Procedure: ML Model Validation for Drug Discovery"
document_id: SOP-ML-VAL-003
version: 2.0
date: 2024-04-15
author: Dr. Julien Brossard, Head of AI/ML Governance
department: Data Science & AI — Governance Office
classification: Internal Use Only
effective_date: 2024-05-01
review_date: 2025-05-01
---

# SOP-ML-VAL-003: ML Model Validation for Drug Discovery

## 1. Purpose and Scope

This SOP defines the minimum validation requirements for machine learning models used in drug discovery decisions. It applies to all classification and regression models that:
- Inform compound selection or triage decisions
- Predict molecular properties (ADMET, activity, selectivity)
- Are referenced in regulatory submissions

Out of scope: Exploratory research models, literature-mining tools, internal dashboards without decision impact.

## 2. Definitions

- **Model**: A trained ML algorithm with fixed parameters that maps molecular representations to predictions
- **Applicability Domain (AD)**: The chemical space within which the model's predictions are considered reliable
- **Performance Drift**: Degradation of model metrics over time as the chemical space of new compounds diverges from training data
- **Holdout Set**: Data reserved exclusively for final evaluation, never used during training or hyperparameter tuning

## 3. Data Splitting Requirements

### 3.1 Cross-Validation

- **Minimum**: Stratified 5-fold cross-validation
- **Recommended**: Stratified 5-fold repeated 3 times (for variance estimation)
- **For small datasets (N < 500)**: Leave-one-out or stratified 10-fold

### 3.2 Holdout Test Set

- Reserve 15-20% of data as a holdout test set
- **Stratified by class label** (for classification) or target distribution quartile (for regression)
- **Scaffold-based splitting** recommended for molecular data: group by Murcko scaffold to prevent information leakage from structural analogs
- Holdout set must NEVER be used for hyperparameter tuning

### 3.3 Temporal Validation (When Available)

- If data spans multiple time periods, validate on the most recent 20% (temporal split)
- This tests generalization to future compounds, which is the real deployment scenario

## 4. Performance Thresholds

### 4.1 Classification Models

| Metric | Minimum | Target | Notes |
|--------|---------|--------|-------|
| ROC-AUC | 0.85 | > 0.90 | Primary metric |
| Sensitivity (Recall) | 0.80 | > 0.85 | For safety-critical properties |
| Specificity | 0.80 | > 0.85 | For efficacy-critical properties |
| Balanced Accuracy | 0.80 | > 0.85 | Accounts for class imbalance |
| Brier Score | < 0.20 | < 0.15 | Probability calibration |
| Matthews Correlation Coefficient | > 0.60 | > 0.70 | Robust to class imbalance |

### 4.2 Regression Models

| Metric | Minimum | Target | Notes |
|--------|---------|--------|-------|
| R² | > 0.60 | > 0.75 | Coefficient of determination |
| RMSE | < 0.5 log units | < 0.3 | For log-scale properties |
| MAE | < 0.4 log units | < 0.25 | Less sensitive to outliers |
| Concordance Index | > 0.70 | > 0.80 | Rank correlation |

### 4.3 External Validation

- Test on at least one independent public dataset
- Performance drop from internal to external ≤ 5% (AUC)
- If drop > 5%, investigate domain shift and document limitations

## 5. Applicability Domain Assessment

### 5.1 Similarity-Based AD

- For each prediction, compute maximum Tanimoto similarity to training set
- **In domain**: max Tanimoto ≥ 0.3 (using Morgan FP, radius=2)
- **Uncertain**: 0.2 ≤ max Tanimoto < 0.3
- **Out of domain**: max Tanimoto < 0.2

### 5.2 Feature Space AD

- PCA on training features → define 95% confidence ellipsoid
- New compounds outside the ellipsoid flagged as out-of-domain

### 5.3 Reporting

All predictions must include:
- Predicted value/class
- Confidence/probability score
- AD flag (in-domain / uncertain / out-of-domain)

## 6. Model Documentation

### 6.1 Model Card (Mandatory)

Every deployed model must have a model card documenting:
- Task description and target property
- Training data source, size, class distribution
- Molecular representation (fingerprints, descriptors, graphs)
- Algorithm and key hyperparameters
- Cross-validation and holdout performance metrics
- Applicability domain definition
- Known limitations and failure modes
- Training date and responsible scientist

### 6.2 Version Control

- Each model version assigned a semantic version (e.g., v2.1.0)
- Training data snapshot archived with model
- Performance metrics recorded in model registry
- Git tag for code version used in training

## 7. Re-Validation Triggers

Re-validate the model when any of the following occurs:

1. **New experimental data**: > 100 new compounds with measured values
2. **Performance drift**: Rolling AUC/R² drops below minimum threshold
3. **Scope expansion**: Model applied to new chemical series or therapeutic area
4. **Code changes**: Modification to feature computation or preprocessing
5. **Dependency update**: Major version change of ML library (sklearn, XGBoost)
6. **Scheduled**: Annual re-validation regardless of triggers

## 8. Approval and Deployment

### 8.1 Review Process

1. Data scientist completes model card and validation report
2. Peer review by independent data scientist (not involved in training)
3. Domain expert (medicinal chemist or pharmacologist) reviews applicability
4. AI/ML Governance sign-off
5. Model registered in central model registry

### 8.2 Deployment Checklist

- [ ] Model card complete
- [ ] All performance thresholds met
- [ ] External validation completed
- [ ] Applicability domain defined
- [ ] Bias audit performed
- [ ] Monitoring dashboard configured
- [ ] Rollback procedure documented
- [ ] Version tagged in model registry
