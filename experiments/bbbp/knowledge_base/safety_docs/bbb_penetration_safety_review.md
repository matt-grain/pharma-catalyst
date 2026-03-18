---
title: "Safety Review: BBB Penetration Prediction Model — Risk Assessment"
document_id: SAF-2024-CNS-0091
version: 1.2
date: 2024-08-20
author: Dr. Isabelle Moreau, Drug Safety & Pharmacovigilance
reviewer: Dr. Thomas Keller, Head of Computational Safety Sciences
department: Drug Safety Sciences
classification: Confidential — Internal Use Only
---

# Safety Review: BBB Penetration Prediction Model

## Purpose

This document assesses the safety implications of deploying a machine learning model for BBB penetration prediction in the drug discovery pipeline. It evaluates risks from prediction errors and establishes minimum validation requirements before production deployment.

## Risk Classification

BBB penetration prediction is classified as **Medium-High Risk** under our AI/ML model governance framework because:
- Prediction errors can lead to advancement of ineffective CNS candidates (false positives) or failure to identify CNS-active compounds with safety liabilities (false negatives)
- The model output directly influences compound progression decisions
- Regulatory submissions may reference model predictions for CNS safety assessments

## Error Impact Analysis

### False Positive Risk (Predicted BBB+ when actually BBB-)

**Scenario**: Model predicts a compound crosses the BBB, but it does not.

**Impact**:
- Compound advances as CNS candidate but fails in vivo BBB assay (brain/plasma ratio)
- Wasted resources on pharmacokinetic studies (est. $50K-200K per compound)
- Delay in program timeline (3-6 months per false lead)
- No direct patient safety risk (compound would fail before clinical stage)

**Severity**: Medium
**Likelihood**: Moderate (model precision 0.92 → ~8% false positive rate)

### False Negative Risk (Predicted BBB- when actually BBB+)

**Scenario**: Model predicts a compound does NOT cross the BBB, but it does.

**Impact**:
- **For non-CNS programs**: Compound may reach brain causing unexpected CNS side effects (sedation, seizures, mood changes)
- **For CNS programs**: Viable candidate is incorrectly triaged out → missed therapeutic opportunity
- **Patient safety concern**: Unintended CNS exposure in non-CNS drugs is a regulatory safety signal

**Severity**: High (for non-CNS programs with safety implications)
**Likelihood**: Moderate (model recall 0.90 → ~10% false negative rate)

### Risk Matrix

| Error Type | Severity | Likelihood | Risk Level | Mitigation |
|------------|----------|------------|------------|------------|
| False Positive (CNS program) | Medium | Moderate | **Medium** | In vivo confirmation before candidate selection |
| False Negative (CNS program) | Medium | Moderate | **Medium** | Ensemble with orthogonal methods |
| False Positive (non-CNS) | Low | Low | **Low** | Standard DMPK screening catches this |
| False Negative (non-CNS) | High | Moderate | **HIGH** | Mandatory BBB prediction for all programs |

## Model Validation Requirements

### Before Deployment (Mandatory)

1. **Performance thresholds** (per SOP-ML-VAL-003):
   - ROC-AUC ≥ 0.85 on held-out test set
   - Sensitivity (recall for BBB+ class) ≥ 0.80
   - Specificity ≥ 0.80
   - Balanced accuracy ≥ 0.80

2. **External validation**: Test on at least one independent public dataset (e.g., BBBP from MoleculeNet) with ROC-AUC ≥ 0.80

3. **Applicability domain assessment**: Define chemical space coverage using Tanimoto similarity to training set. Flag predictions for compounds with max Tanimoto < 0.3 as "out of domain"

4. **Calibration**: Platt scaling or isotonic regression on predicted probabilities. Brier score ≤ 0.15

5. **Bias audit**: Verify no systematic performance differences across:
   - Compound MW ranges (< 300, 300-450, > 450)
   - Lipophilicity ranges (LogP < 1, 1-3, > 3)
   - Charge states (neutral, basic, acidic, zwitterionic)

### Ongoing Monitoring

1. **Quarterly re-validation** against new experimental data (minimum 50 new compounds per quarter)
2. **Performance drift detection**: Alert if rolling AUC drops below 0.83 (2% below threshold)
3. **Version control**: All model versions archived with training data snapshot and performance metrics

## Regulatory Considerations

### FDA Guidance Alignment

- **ICH M7**: For mutagenic impurities, computational predictions (including BBB) must be documented with model version, training data scope, and applicability domain
- **FDA Drug Safety Guidance (2023)**: AI/ML models used in safety assessment require documentation of training data, validation methodology, and known limitations
- **EMA Reflection Paper on AI (2023)**: Models influencing go/no-go decisions should have documented performance metrics and human oversight

### Documentation Requirements for Regulatory Submissions

If model predictions are referenced in IND/NDA submissions:
1. Model description (algorithm, features, training data size and source)
2. Validation report (cross-validation + external test set metrics)
3. Applicability domain definition
4. Known limitations and failure modes
5. Human expert review confirmation for key decisions

## Recommendations

1. **Deploy with guardrails**: All predictions must include confidence scores. Predictions with probability 0.4-0.6 flagged as "uncertain" requiring experimental confirmation.
2. **Mandatory for non-CNS safety**: All non-CNS program compounds must be screened for unintended BBB penetration.
3. **Not a replacement for in vivo**: Model predictions complement but do not replace MDCK-MDR1 or in vivo brain exposure studies for candidate selection.
4. **Annual model retraining**: Incorporate new experimental data annually and re-validate.
