"""
Training pipeline for ESOL solubility prediction.
Modified to use HistGradientBoostingRegressor and augmented features.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors


def get_mol_features(smi: str) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint and physicochemical descriptors."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(1024 + 5)
    # Morgan Fingerprint
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = gen.GetFingerprintAsNumPy(mol)
    # Physicochemical Descriptors
    # Use pyright ignore because RDKit descriptors are often not recognized by static analysis
    logp = float(Descriptors.MolLogP(mol))  # type: ignore # pyright: ignore [reportAttributeAccessIssue]
    mw = float(Descriptors.MolWt(mol))  # type: ignore # pyright: ignore [reportAttributeAccessIssue]
    rb = float(Descriptors.NumRotatableBonds(mol))  # type: ignore # pyright: ignore [reportAttributeAccessIssue]
    ha = float(Descriptors.HeavyAtomCount(mol))  # type: ignore # pyright: ignore [reportAttributeAccessIssue]
    aromatic_ratio = (
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / ha if ha > 0 else 0.0
    )
    desc_features = np.array([logp, mw, rb, ha, aromatic_ratio])
    return np.concatenate([fp, desc_features])


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load ESOL dataset and convert to features/targets."""
    # Data is in src/pharma_agents/data/ (shared across experiments)
    data_path = (
        Path(__file__).parent.parent / "src" / "pharma_agents" / "data" / "esol.csv"
    )
    df = pd.read_csv(data_path)

    # Convert SMILES to features
    X = np.array([get_mol_features(smi) for smi in df["smiles"]])
    y: np.ndarray = df["measured log solubility in mols per litre"].values  # type: ignore[assignment]

    return X, y


def train(verbose: bool = True) -> float:
    """
    Train model and return validation RMSE.
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

    # Model - HistGradientBoostingRegressor
    if verbose:
        print("[3/4] Training HistGradientBoostingRegressor...")
    t0 = time.perf_counter()
    model = HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.05, max_depth=10, random_state=42
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

    return float(rmse)


if __name__ == "__main__":
    rmse_val = train(verbose=True)
    print(f"\nValidation RMSE: {rmse_val:.4f}")
