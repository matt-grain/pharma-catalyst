---
title: "BBB Penetration Prediction: ML Model Benchmarking Study"
document_id: RPT-2024-CNS-0047
version: 2.1
date: 2024-09-15
author: Dr. Marie-Claire Dupont, Computational Chemistry Group
department: R&D Informatics — CNS Therapeutic Area
classification: Internal Use Only
---

# BBB Penetration Prediction: ML Model Benchmarking Study

## Executive Summary

This report presents a systematic benchmarking of machine learning approaches for predicting blood-brain barrier (BBB) penetration using our internal compound library (N=2,847) and the public BBBP dataset (N=2,039). We evaluated four model architectures across five molecular representation strategies. Key finding: ensemble methods combining Morgan fingerprints with physicochemical descriptors achieve the best balance of performance (ROC-AUC 0.94) and interpretability. Graph neural networks showed marginal improvement (+0.01 AUC) but at significantly higher computational cost and reduced interpretability.

## Dataset Characteristics

Our internal dataset comprises 2,847 compounds tested in the MDCK-MDR1 permeability assay, supplemented with PAMPA-BBB results for 1,203 compounds. Class distribution: 58% BBB+ (penetrating), 42% BBB- (non-penetrating). The public BBBP dataset was used for external validation.

Notable dataset challenges:
- Class imbalance (58/42 split) — moderate, addressed via stratified splitting
- Scaffold bias — 23% of BBB+ compounds share Markush scaffolds from our CNS program
- Activity cliffs — 147 compound pairs with >0.85 Tanimoto similarity but opposite BBB labels

## Molecular Representations Evaluated

### Fingerprint Comparison

| Fingerprint | Bits | ROC-AUC (Internal) | ROC-AUC (BBBP) | Training Time |
|-------------|------|---------------------|-----------------|---------------|
| Morgan (r=2) | 1024 | 0.891 | 0.872 | 2.3s |
| Morgan (r=2) | 2048 | 0.894 | 0.876 | 3.1s |
| MACCS Keys | 167 | 0.862 | 0.851 | 1.1s |
| AtomPair | 2048 | 0.878 | 0.864 | 4.7s |
| TopologicalTorsion | 2048 | 0.871 | 0.858 | 5.2s |
| Morgan + MACCS (concat) | 1191 | 0.903 | 0.889 | 3.8s |

Morgan fingerprints (radius=2, 2048 bits) provided the best single-representation performance. Concatenating Morgan with MACCS keys yielded a consistent +1.2% improvement, suggesting complementary structural information captured by the two approaches.

### Physicochemical Descriptors

We evaluated 208 RDKit molecular descriptors. Feature importance analysis (permutation-based) identified the following as most predictive:

| Descriptor | Importance | Direction | Threshold |
|------------|------------|-----------|-----------|
| TPSA | 0.142 | Lower = more permeable | < 90 A² |
| MolLogP | 0.128 | Optimal range | 1.0 - 3.5 |
| MolWt | 0.097 | Lower preferred | < 450 Da |
| NumHDonors | 0.089 | Fewer preferred | ≤ 3 |
| NumRotatableBonds | 0.071 | Fewer preferred | ≤ 8 |
| FractionCSP3 | 0.058 | Higher preferred | > 0.25 |
| NumAromaticRings | 0.045 | Moderate | 1-3 |
| BertzCT | 0.038 | Lower complexity | < 600 |

Adding the top 20 descriptors to Morgan fingerprints improved all models by 2-4% AUC.

## Model Comparison

All models evaluated with stratified 5-fold cross-validation, repeated 3 times. Reported metrics are mean ± std.

| Model | Features | ROC-AUC | Precision | Recall | F1 | Train Time |
|-------|----------|---------|-----------|--------|----|-----------:|
| LogisticRegression | Morgan 2048 | 0.876±0.012 | 0.84 | 0.81 | 0.82 | 4s |
| LogisticRegression | Morgan+Desc | 0.895±0.010 | 0.86 | 0.84 | 0.85 | 6s |
| RandomForest (500) | Morgan+Desc | 0.921±0.008 | 0.89 | 0.87 | 0.88 | 45s |
| XGBoost | Morgan+Desc | 0.932±0.007 | 0.91 | 0.88 | 0.89 | 38s |
| XGBoost + RF Ensemble | Morgan+MACCS+Desc | 0.941±0.006 | 0.92 | 0.90 | 0.91 | 95s |
| GCN (3-layer) | Molecular Graph | 0.929±0.015 | 0.90 | 0.87 | 0.88 | 12min |
| AttentiveFP | Molecular Graph | 0.938±0.012 | 0.91 | 0.89 | 0.90 | 18min |
| MPNN | Molecular Graph | 0.935±0.014 | 0.90 | 0.88 | 0.89 | 22min |

## Key Findings

1. **Ensemble methods are optimal for production**: The XGBoost + RandomForest ensemble with Morgan+MACCS+descriptors achieves ROC-AUC 0.941 with training under 2 minutes. This meets our SOP deployment threshold (>0.85 AUC).

2. **GNNs offer marginal gains at high cost**: AttentiveFP reaches 0.938 AUC but requires 18 minutes training, GPU infrastructure, and provides limited feature interpretability. The gain over the ensemble (+0.003 AUC) does not justify the operational overhead.

3. **Descriptor selection matters**: Adding all 208 descriptors degraded XGBoost performance by 0.8% vs. the top-20 selected set, likely due to noise from irrelevant features. Feature selection (mutual information + permutation importance) is essential.

4. **TPSA is the single most predictive descriptor** for BBB penetration, consistent with the biophysical understanding that polar surface area governs passive diffusion across lipid bilayers.

## Recommendations

1. **Adopt XGBoost + RF ensemble** with Morgan (2048) + MACCS + top-20 descriptors as the production model for BBB screening.
2. **Implement SMILES sanitization** — 3.2% of our compound library contains invalid SMILES that cause RDKit errors. Add preprocessing with `Chem.MolFromSmiles()` validation.
3. **Stratified splitting is mandatory** — random splits overestimate performance by ~2% due to scaffold leakage.
4. **Re-benchmark annually** as new GNN architectures (e.g., GPS, Graphormer) may close the efficiency gap.
5. **Calibrate probability outputs** — current models are overconfident on borderline compounds (Pe = 2-4 × 10⁻⁶ cm/s). Platt scaling recommended.
