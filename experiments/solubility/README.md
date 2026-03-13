# Solubility Experiment

**Aqueous Solubility Prediction (ESOL)**

A regression task predicting the aqueous solubility (logS) of molecules, a key ADMET property for drug development.

## Dataset

- **Source**: ESOL (Delaney, 2004)
- **Size**: ~1,128 molecules
- **Task**: Regression (predict logS)
- **Features**: Morgan fingerprints (1024 bits, radius 2)

## Baseline

| Metric | Value |
|--------|-------|
| Model | RandomForestRegressor |
| RMSE | 1.3175 |
| Direction | Lower is better |

## Usage

```bash
# Run this experiment
uv run python -m pharma_agents.main --experiment solubility

# Or set via environment
PHARMA_EXPERIMENT=solubility uv run python -m pharma_agents.main
```

## Improvement Ideas

- **Features**: Add physicochemical descriptors (MolLogP, MolWt, TPSA)
- **Model**: Try HistGradientBoosting, XGBoost, or SVR
- **Fingerprints**: Combine Morgan with MACCS keys
- **Feature selection**: Use RFE or feature importance pruning
