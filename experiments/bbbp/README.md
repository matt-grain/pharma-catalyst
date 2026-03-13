# BBBP Experiment

**Blood-Brain Barrier Penetration Prediction**

A binary classification task predicting whether molecules can cross the blood-brain barrier (BBB), crucial for CNS drug development.

## Dataset

- **Source**: MoleculeNet (BBBP)
- **Size**: ~2,050 molecules
- **Task**: Binary classification (penetrates BBB or not)
- **Features**: Morgan fingerprints (1024 bits, radius 2)

## Baseline

| Metric | Value |
|--------|-------|
| Model | LogisticRegression |
| ROC-AUC | 0.8951 |
| Direction | Higher is better |

## Usage

```bash
# Run this experiment
uv run python -m pharma_agents.main --experiment bbbp

# Or set via environment
PHARMA_EXPERIMENT=bbbp uv run python -m pharma_agents.main
```

## Improvement Ideas

- **Features**: Add physicochemical descriptors (MolLogP, TPSA, MW)
- **Model**: Try GradientBoosting, RandomForest, or SVM
- **Fingerprints**: Experiment with MACCS keys or different radius/bits
- **Class imbalance**: Use class weights or SMOTE
