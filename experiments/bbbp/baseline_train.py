"""
BASELINE Training pipeline for BBBP (Blood-Brain Barrier Penetration) prediction.

THIS FILE IS THE REFERENCE BASELINE - NEVER MODIFIED.
Copy this to train.py to reset to baseline state.

The train() function must:
- Return ROC-AUC (float) on the validation set
- Use data from ../src/pharma_agents/data/bbbp.csv
- Complete in under 60 seconds
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def smiles_to_fingerprint(
    smiles: str, radius: int = 2, n_bits: int = 1024
) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprintAsNumPy(mol)
    return fp


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load BBBP dataset and convert to features/targets."""
    # Data is in src/pharma_agents/data/ (shared across experiments)
    # Path: experiments/bbbp/ -> project_root/src/pharma_agents/data/
    data_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "pharma_agents"
        / "data"
        / "bbbp.csv"
    )
    df = pd.read_csv(data_path)

    # Filter out invalid SMILES
    valid_mask = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df[valid_mask]

    # Convert SMILES to fingerprints
    X = np.array([smiles_to_fingerprint(smi) for smi in df["smiles"]])
    y: np.ndarray = df["p_np"].values  # Binary: 1 = penetrates BBB, 0 = does not

    return X, y


def train(verbose: bool = True) -> float:
    """
    Train model and return validation ROC-AUC.

    This is the function the agents optimize.
    """
    total_start = time.perf_counter()

    # Load data
    if verbose:
        print("[1/4] Loading data...")
    t0 = time.perf_counter()
    X, y = load_data()
    load_time = time.perf_counter() - t0
    if verbose:
        print(f"      Loaded {len(y)} molecules in {load_time:.2f}s")
        print(
            f"      Class distribution: {sum(y)} positive, {len(y) - sum(y)} negative"
        )

    # Split
    if verbose:
        print("[2/4] Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if verbose:
        print(f"      Train: {len(y_train)}, Val: {len(y_val)}")

    # Model - BASELINE: simple LogisticRegression
    if verbose:
        print("[3/4] Training LogisticRegression...")
    t0 = time.perf_counter()
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    if verbose:
        print(f"      Training completed in {train_time:.2f}s")

    # Evaluate
    if verbose:
        print("[4/4] Evaluating...")
    t0 = time.perf_counter()
    y_prob = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_prob)
    eval_time = time.perf_counter() - t0

    total_time = time.perf_counter() - total_start

    if verbose:
        print(f"\n{'=' * 40}")
        print("RESULTS")
        print(f"{'=' * 40}")
        print(f"Validation ROC_AUC: {roc_auc:.4f}")
        print(f"{'=' * 40}")
        print("TIMING")
        print(f"{'=' * 40}")
        print(f"Data loading:    {load_time:.2f}s")
        print(f"Training:        {train_time:.2f}s")
        print(f"Evaluation:      {eval_time:.2f}s")
        print(f"Total:           {total_time:.2f}s")

    return roc_auc


# BASELINE ROC-AUC: ~0.82

if __name__ == "__main__":
    roc_auc = train(verbose=True)
    print(f"\nValidation ROC_AUC: {roc_auc:.4f}")
