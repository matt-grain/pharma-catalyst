# ClinTox Experiment

**Clinical Trial Toxicity Prediction**

A binary classification task predicting whether drugs will fail clinical trials due to toxicity. This is a critical safety endpoint in drug development.

## Dataset

- **Source**: MoleculeNet (ClinTox)
- **Size**: ~1,484 molecules
- **Task**: Binary classification (CT_TOX: 1 = failed due to toxicity, 0 = passed)
- **Features**: Morgan fingerprints (baseline: 1024 bits, radius 2)
- **Class imbalance**: 112 toxic / 1372 non-toxic (~7.5% positive)

## Baseline

| Metric | Value |
|--------|-------|
| Model | TensorFlow MLP (128-64-1) |
| ROC-AUC | 0.6989 |
| Direction | Higher is better |

## Usage

```bash
# Run this experiment
uv run python -m pharma_agents.main --experiment clintox

# Or set via environment
PHARMA_EXPERIMENT=clintox uv run python -m pharma_agents.main
```

## Baseline Model

The starting point that agents will improve:

### Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Configuration

| Component | Value |
|-----------|-------|
| Features | Morgan fingerprints (1024 bits) |
| Fingerprint radius | 2 |
| Model | TensorFlow MLP |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary crossentropy |
| Train/Val split | 80/20 (stratified) |
| Early stopping | patience=5 |
| **Baseline ROC-AUC** | **0.6989** |

---

## Improvement Ideas

- **Features**: Add physicochemical descriptors (TPSA, LogP, MW)
- **Features**: MACCS keys, RDKit fingerprints, concatenated fingerprints
- **Architecture**: Deeper network, different activation functions (GELU, SiLU)
- **Regularization**: L2 weight decay, different dropout rates
- **Class imbalance**: Class weights, focal loss, SMOTE oversampling
- **Ensemble**: Combine with RandomForest or XGBoost

---

*Optimized by pharma-agents crew. Powered by CrewAI + TensorFlow.*
