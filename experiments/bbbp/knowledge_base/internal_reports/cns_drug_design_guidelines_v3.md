---
title: "CNS Drug Design Guidelines: Physicochemical Property Criteria"
document_id: GL-2024-CNS-0012
version: 3.0
date: 2024-06-01
author: Dr. Philippe Laurent, Head of Medicinal Chemistry — CNS
department: Medicinal Chemistry — CNS Therapeutic Area
classification: Internal Use Only
---

# CNS Drug Design Guidelines v3.0

## Purpose

This document defines the physicochemical property criteria for designing compounds intended to penetrate the blood-brain barrier (BBB). These guidelines should be applied during hit identification, lead optimization, and candidate selection for all CNS programs.

## BBB-Specific Property Ranges

The following ranges are derived from analysis of our internal CNS compound library (N=4,200 compounds with MDCK-MDR1 and PAMPA-BBB data) combined with published CNS MPO criteria.

### Primary Filters (Mandatory)

| Property | Optimal Range | Acceptable Range | CNS+ Hit Rate |
|----------|--------------|-----------------|---------------|
| Molecular Weight (MW) | 200-400 Da | 150-450 Da | 78% in optimal |
| Topological Polar Surface Area (TPSA) | 40-80 A² | 20-90 A² | 82% in optimal |
| Calculated LogP (CLogP) | 1.5-3.0 | 0.5-4.0 | 71% in optimal |
| H-Bond Donors (HBD) | 0-2 | 0-3 | 85% with ≤2 |
| H-Bond Acceptors (HBA) | 2-6 | 1-8 | 74% in optimal |
| Rotatable Bonds | 2-6 | 0-8 | 76% in optimal |
| pKa (most basic center) | 7.5-10.5 | 6.0-11.0 | — |

### CNS Multiparameter Optimization (MPO) Score

We use a modified CNS MPO score (adapted from Wager et al., ACS Chem. Neurosci. 2010):

```
CNS_MPO = f(CLogP) + f(CLogD) + f(MW) + f(TPSA) + f(HBD) + f(pKa)
```

Each component scores 0-1 based on desirability functions. Total score range: 0-6.

| MPO Score | Classification | Recommendation |
|-----------|---------------|----------------|
| ≥ 4.5 | Highly favorable | Advance to in vivo |
| 3.5 - 4.5 | Moderate | Optimize if possible |
| < 3.5 | Unfavorable | Deprioritize for CNS |

Internal validation: 89% of our clinical CNS candidates had MPO ≥ 4.0.

## Efflux Transporter Considerations

### P-glycoprotein (P-gp) Efflux

P-gp efflux is the primary reason compounds with acceptable passive permeability fail to achieve brain exposure. Key guidelines:

1. **Efflux ratio (ER) threshold**: ER < 2.5 in MDCK-MDR1 assay indicates acceptable P-gp liability
2. **Structural alerts for P-gp substrates**:
   - Tertiary amines with CLogP > 3.5
   - Compounds with > 3 HBD
   - Molecules with TPSA > 90 A²
   - Macrocycles (MW > 600)
3. **Descriptor flags**: Compounds with high BertzCT (complexity > 700) are 2.3x more likely to be P-gp substrates

### BCRP Efflux

Breast cancer resistance protein (BCRP) is emerging as a secondary efflux concern:
- Sulfonate and glucuronide moieties are BCRP alerts
- Monitor BCRP ER alongside P-gp in dual-transfected assays

## Key Structural Alerts for Poor BBB Penetration

The following substructures are associated with >70% probability of BBB- classification:

1. **Carboxylic acids** (pKa < 5) — ionized at physiological pH, poor passive permeability
2. **Multiple hydroxyl groups** (≥ 3 OH) — excessive H-bonding, high TPSA
3. **Quaternary ammonium** — permanently charged, cannot cross lipid bilayer
4. **Sulfonic acids / sulfonates** — strong acid, high PSA
5. **Phosphate groups** — charged, MW penalty
6. **Polysaccharide moieties** — high MW, hydrophilic
7. **Zwitterions with ΔpKa > 4** — charged at all physiological pH values

## Recommended Descriptor Panel for ML Screening

For computational BBB prediction models, we recommend the following descriptor panel (ordered by predictive importance from our benchmarking study RPT-2024-CNS-0047):

### Tier 1 (Essential — always include)
- TPSA, MolLogP, MolWt, NumHDonors, NumHAcceptors

### Tier 2 (Recommended — significant improvement)
- NumRotatableBonds, FractionCSP3, NumAromaticRings, BertzCT, LabuteASA

### Tier 3 (Optional — marginal improvement)
- Chi0v, HallKierAlpha, PEOE_VSA descriptors, MQN descriptors, Kappa indices

### Fingerprints
- Morgan (radius=2, 2048 bits) as primary
- MACCS (167 keys) as complementary — captures pharmacophoric patterns not in Morgan

## Application to Hit Triage

When triaging virtual screening hits for CNS programs:

1. Apply primary filters (Section 2) — reject compounds outside acceptable ranges
2. Calculate CNS MPO score — rank by score
3. Flag P-gp structural alerts — mark for follow-up in efflux assay
4. Run ML BBB prediction model — require predicted probability > 0.7
5. Visual inspection by medicinal chemist — assess synthetic accessibility and novelty
