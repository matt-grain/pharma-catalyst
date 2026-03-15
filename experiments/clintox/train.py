"""
BASELINE Training pipeline for ClinTox toxicity prediction.

THIS FILE IS THE REFERENCE BASELINE - NEVER MODIFIED.
Copy this to train.py to reset to baseline state.

The train() function must:
- Return ROC_AUC (float) on the validation set
- Use data from clintox.csv
- Complete in under 60 seconds

ClinTox contains drugs that failed clinical trials due to toxicity (CT_TOX=1)
vs drugs that passed (CT_TOX=0). This is a critical safety prediction task.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Suppress TensorFlow warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


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
    """Load ClinTox dataset and convert to features/targets."""
    data_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "pharma_agents"
        / "data"
        / "clintox.csv"
    )
    df = pd.read_csv(data_path)

    # Convert SMILES to fingerprints
    X = np.array([smiles_to_fingerprint(smi) for smi in df["smiles"]])
    y: np.ndarray = df["CT_TOX"].values  # 1 = toxic in clinical trial, 0 = passed

    return X, y


def build_model(input_dim: int):
    """Build a simple TensorFlow neural network for binary classification."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train(verbose: bool = True) -> float:
    """
    Train model and return validation ROC-AUC.

    This is the function the agents optimize.
    """
    total_start = time.perf_counter()

    # Load data
    if verbose:
        print("[1/4] Loading ClinTox data...")
    t0 = time.perf_counter()
    X, y = load_data()
    load_time = time.perf_counter() - t0
    if verbose:
        print(f"      Loaded {len(y)} molecules in {load_time:.2f}s")
        print(f"      Class balance: {y.sum():.0f} toxic / {len(y) - y.sum():.0f} non-toxic")

    # Split
    if verbose:
        print("[2/4] Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if verbose:
        print(f"      Train: {len(y_train)}, Val: {len(y_val)}")

    # Build and train model
    if verbose:
        print("[3/4] Training TensorFlow MLP...")
    t0 = time.perf_counter()

    model = build_model(input_dim=X_train.shape[1])

    # Train with early stopping
    import tensorflow as tf

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=30,  # Reduced for faster training
        batch_size=64,  # Larger batch for speed
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,  # Silent training
    )

    train_time = time.perf_counter() - t0
    if verbose:
        epochs_run = len(history.history["loss"])
        print(f"      Training completed in {train_time:.2f}s ({epochs_run} epochs)")

    # Evaluate
    if verbose:
        print("[4/4] Evaluating...")
    t0 = time.perf_counter()
    y_pred_proba = model.predict(X_val, verbose=0).flatten()
    roc_auc = roc_auc_score(y_val, y_pred_proba)
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


# BASELINE ROC_AUC: TBD (run to establish)

if __name__ == "__main__":
    roc_auc = train(verbose=True)
    print(f"\nROC_AUC: {roc_auc:.4f}")
