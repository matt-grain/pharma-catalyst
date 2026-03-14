# BBBP Experiment

**Blood-Brain Barrier Penetration Prediction**

A binary classification task predicting whether molecules can cross the blood-brain barrier (BBB), crucial for CNS drug development.

## Dataset

- **Source**: MoleculeNet (BBBP)
- **Size**: ~2,050 molecules
- **Task**: Binary classification (penetrates BBB or not)
- **Features**: Morgan fingerprints (baseline: 1024 bits, radius 2)

## Results Summary

| Run | ROC-AUC | Improvement | Key Change |
|-----|---------|-------------|------------|
| Baseline | 0.8951 | - | LogisticRegression + Morgan FP (1024 bits) |
| Run 1 | 0.9300 | +3.9% | Added physicochemical descriptors (LogP, TPSA, MW, HBD, HBA) |
| **Run 2** | **0.9418** | **+5.2%** | Morgan FP (2048 bits, radius 3) + MACCS keys + XGBoost |

**Best Result: ROC-AUC 0.9418** (5.2% improvement over baseline)

---

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
    """Convert a molecule (as SMILES string) to a numerical feature vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprintAsNumPy(mol)
    return fp


# Model: LogisticRegression for binary classification
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
```

### Configuration

| Component | Value |
|-----------|-------|
| Features | Morgan fingerprints (1024 bits) |
| Fingerprint radius | 2 (captures atom neighborhoods up to 2 bonds away) |
| Model | LogisticRegression |
| Train/Val split | 80/20 (stratified) |
| Dataset | BBBP (~2,050 molecules) |
| Target | Binary (1 = penetrates BBB, 0 = does not) |
| **Baseline ROC-AUC** | **0.8951** |

---

## What the Agents Did

### Run 1: Physicochemical Descriptors (+3.9%)

**Hypothesis Agent:**
> Augment Morgan fingerprints with key RDKit physicochemical descriptors (LogP, TPSA, MW, HBD, HBA)

**Reasoning:**
> BBB penetration follows complex rules (like the 'Rule of 3' for CNS drugs) where specific thresholds of lipophilicity, polar surface area, and molecular weight determine permeability.

**Result:** ROC-AUC 0.9300

### Run 2: Extended Features + XGBoost (+5.2%)

**Hypothesis Agent:**
> Augment 2048-bit Morgan fingerprints (radius 3) with RDKit physicochemical descriptors and MACCS keys, then use XGBoost classifier.

**Reasoning:**
> Expanding fingerprint radius to 3 and bits to 2048 reduces hash collisions and captures larger structural neighborhoods. MACCS keys add interpretable pharmacophore features. XGBoost handles feature interactions better than linear models.

**Result:** ROC-AUC 0.9418

---

## Literature Insights

Papers gathered by the Archivist agent that informed hypotheses:

| Paper | Key Technique |
|-------|---------------|
| 2107.06773 | RGCN with drug-protein interactions + Mordred descriptors |
| 2208.09484 | Molecular feature modeling for BBB prediction |
| 2507.18557 | Deep learning for BBB permeability |

---

## Improvement Ideas (Future Runs)

- **Features**: Add Mordred descriptors, graph-based features
- **Model**: Try GNN architectures (RGCN, MPNN)
- **Ensemble**: Combine multiple fingerprint types
- **Drug-protein**: Include transporter interaction features
- **Attention**: Use molecular transformers

---

*Optimized by pharma-agents crew. Powered by CrewAI + Gemini.*
