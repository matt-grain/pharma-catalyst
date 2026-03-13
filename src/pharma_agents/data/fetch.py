"""
Fetch molecular property datasets.

Supports:
- ESOL (solubility) - ~1,128 molecules
- BBBP (blood-brain barrier penetration) - ~2,050 molecules
"""

import urllib.request
from pathlib import Path


DATASETS = {
    "esol": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "filename": "esol.csv",
        "description": "ESOL (Estimated SOLubility) from Delaney (2004)",
    },
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "filename": "bbbp.csv",
        "description": "BBBP (Blood-Brain Barrier Penetration) from MoleculeNet",
    },
}


def download_dataset(name: str, output_path: Path | None = None) -> Path:
    """
    Download a molecular property dataset.

    Args:
        name: Dataset name ('esol' or 'bbbp')
        output_path: Where to save. Defaults to data/{name}.csv

    Returns:
        Path to downloaded file
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    dataset = DATASETS[name]

    if output_path is None:
        output_path = Path(__file__).parent / dataset["filename"]

    if output_path.exists():
        print(f"{name.upper()} dataset already exists at {output_path}")
        return output_path

    print(f"Downloading {dataset['description']}...")
    print(f"URL: {dataset['url']}")
    urllib.request.urlretrieve(dataset["url"], output_path)
    print(f"Saved to {output_path}")

    # Verify
    with open(output_path) as f:
        lines = f.readlines()
    print(f"Dataset has {len(lines) - 1} molecules")

    return output_path


def download_esol(output_path: Path | None = None) -> Path:
    """Download ESOL dataset (backward compatibility)."""
    return download_dataset("esol", output_path)


def download_bbbp(output_path: Path | None = None) -> Path:
    """Download BBBP dataset."""
    return download_dataset("bbbp", output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].lower()
    else:
        dataset_name = "bbbp"  # Default to BBBP for new experiment

    download_dataset(dataset_name)
