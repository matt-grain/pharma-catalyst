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

## Baseline Model (Before Modification)

The starting point that agents were given to improve:

### Code

```python
def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    Convert a molecule (as SMILES string) to a numerical feature vector.

    Morgan Fingerprints:
    - Look at each atom and its neighborhood (up to 'radius' bonds away)
    - Hash these neighborhoods into a fixed-size bit vector
    - If two molecules share substructures, they'll have similar fingerprints

    Args:
        smiles: Molecule as SMILES string
        radius: How far to look around each atom (2 = up to 2 bonds away)
        n_bits: Size of output vector (1024-dimensional fingerprint)

    Returns:
        numpy array of 0s and 1s representing molecular structure
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprintAsNumPy(mol)
    return fp


# Model: LogisticRegression for binary classification
model = LogisticRegression(
    max_iter=1000,      # Ensure convergence
    random_state=42,    # Fixed seed for reproducibility
    n_jobs=-1           # Use all CPU cores
)
# Why LogisticRegression?
# - Simple, interpretable baseline for binary classification
# - Works well with sparse, high-dimensional data
# - Fast training on small datasets
```

### Configuration

| Component | Value |
|-----------|-------|
| Features | Morgan fingerprints (1024 bits) |
| Fingerprint radius | 2 (captures atom neighborhoods up to 2 bonds away) |
| Model | LogisticRegression |
| max_iter | 1000 |
| Train/Val split | 80/20 (stratified) |
| Dataset | BBBP (~2,050 molecules) |
| Target | Binary (1 = penetrates BBB, 0 = does not) |
| **Baseline ROC-AUC** | **0.8951** |

### Limitations

- **Morgan fingerprints** capture local substructures but miss global molecular properties
- **No physicochemical descriptors** - TPSA, LogP, etc. are not explicitly encoded
- **Fixed hyperparameters** - no tuning attempted
- **Linear model** - may miss non-linear relationships

---

## Flow Example Summary

```
BBBP Dataset (~2050 molecules)
    │
    ▼
SMILES strings ("CCO", "c1ccccc1", ...)
    │
    ▼
Morgan Fingerprints (1024-bit vector per molecule)
    │
    ▼
LogisticRegression ──► predicts BBB penetration probability
    │
    ▼
Compare to known labels ──► ROC-AUC = 0.8951 (baseline)

═══════════════════════════════════════════════════════════════

AGENT CREW KICKS IN:

1. Research Agent: "Increase fingerprint resolution and use
   RandomForest for non-linear patterns."

2. Model Agent: Implements radius=3, n_bits=2048, RandomForestClassifier

3. Evaluator Agent: Runs new model ──► ROC-AUC = 0.9265 (3.5% better)
   Recommendation: "KEEP"
```

**Key insight:** Larger fingerprints capture more structural detail, and RandomForest can find non-linear feature interactions that LogisticRegression misses.

---

## Results Summary

| Metric | Value |
|--------|-------|
| Baseline ROC-AUC | 0.8951 |
| Final ROC-AUC | 0.9265 |
| Improvement | 3.5% |
| Recommendation | KEEP |

---

## What the Agents Did

### 1. Hypothesis Agent (Research Scientist)

**Proposal:** Switch from LogisticRegression to RandomForestClassifier, and increase fingerprint resolution:
- Radius: 2 → 3 (capture larger molecular neighborhoods)
- Bits: 1024 → 2048 (reduce hash collisions)

**Reasoning:**
> BBB penetration depends on complex structural patterns that linear models may miss. RandomForest can capture non-linear feature interactions. Increasing fingerprint resolution captures more detailed substructures relevant to membrane permeability.

### 2. Model Agent (ML Engineer)

**Implementation:** Modified `train.py`:

```python
# Changed fingerprint parameters
def smiles_to_fingerprint(smiles: str, radius: int = 3, n_bits: int = 2048) -> np.ndarray:
    ...

# Changed model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
```

**Key changes:**
- Fingerprint: 1024 bits (radius 2) → 2048 bits (radius 3)
- Model: LogisticRegression → RandomForestClassifier

### 3. Evaluator Agent (QA Scientist)

**Evaluation report:**
```
ROC-AUC: 0.9265
BASELINE: 0.8951
IMPROVEMENT: 3.5%
RECOMMENDATION: KEEP
```

---

## Improvement Ideas

- **Features**: Add physicochemical descriptors (MolLogP, TPSA, MW)
- **Model**: Try GradientBoosting, XGBoost, or SVM
- **Fingerprints**: Experiment with MACCS keys or atom pair fingerprints
- **Class imbalance**: Use class weights or SMOTE
- **Feature selection**: Use feature importance to prune low-value bits
