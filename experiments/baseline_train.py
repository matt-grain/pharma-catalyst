"""
BASELINE Training pipeline for ESOL solubility prediction.

THIS FILE IS THE REFERENCE BASELINE - NEVER MODIFIED.
Copy this to train.py to reset to baseline state.

The train() function must:
- Return RMSE (float) on the validation set
- Use data from ../data/esol.csv
- Complete in under 60 seconds
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    """Load ESOL dataset and convert to features/targets."""
    # Data is in src/pharma_agents/data/ (shared across experiments)
    data_path = (
        Path(__file__).parent.parent / "src" / "pharma_agents" / "data" / "esol.csv"
    )
    df = pd.read_csv(data_path)

    # Convert SMILES to fingerprints
    X = np.array([smiles_to_fingerprint(smi) for smi in df["smiles"]])
    y: np.ndarray = df["measured log solubility in mols per litre"].values  # type: ignore[assignment]

    return X, y


def train(verbose: bool = True) -> float:
    """
    Train model and return validation RMSE.

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

    # Split
    if verbose:
        print("[2/4] Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if verbose:
        print(f"      Train: {len(y_train)}, Val: {len(y_val)}")

    # Model - BASELINE: simple RandomForest
    if verbose:
        print("[3/4] Training RandomForest (n_estimators=100, max_depth=10)...")
    t0 = time.perf_counter()
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    if verbose:
        print(f"      Training completed in {train_time:.2f}s")

    # Evaluate
    if verbose:
        print("[4/4] Evaluating...")
    t0 = time.perf_counter()
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    eval_time = time.perf_counter() - t0

    total_time = time.perf_counter() - total_start

    if verbose:
        print(f"\n{'=' * 40}")
        print("RESULTS")
        print(f"{'=' * 40}")
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"{'=' * 40}")
        print("TIMING")
        print(f"{'=' * 40}")
        print(f"Data loading:    {load_time:.2f}s")
        print(f"Training:        {train_time:.2f}s")
        print(f"Evaluation:      {eval_time:.2f}s")
        print(f"Total:           {total_time:.2f}s")

    return rmse


# BASELINE RMSE: 1.3175

if __name__ == "__main__":
    rmse = train(verbose=True)
    print(f"\nValidation RMSE: {rmse:.4f}")
