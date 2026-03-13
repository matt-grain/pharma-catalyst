"""
Fetch ESOL dataset.

ESOL (Estimated SOLubility) dataset from Delaney (2004).
~1,128 molecules with measured aqueous solubility.
"""

import urllib.request
from pathlib import Path


ESOL_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
)


def download_esol(output_path: Path | None = None) -> Path:
    """
    Download ESOL dataset from DeepChem S3 bucket.

    Args:
        output_path: Where to save. Defaults to data/esol.csv

    Returns:
        Path to downloaded file
    """
    if output_path is None:
        output_path = Path(__file__).parent / "esol.csv"

    if output_path.exists():
        print(f"ESOL dataset already exists at {output_path}")
        return output_path

    print(f"Downloading ESOL dataset from {ESOL_URL}...")
    urllib.request.urlretrieve(ESOL_URL, output_path)
    print(f"Saved to {output_path}")

    # Verify
    with open(output_path) as f:
        lines = f.readlines()
    print(f"Dataset has {len(lines) - 1} molecules")

    return output_path


if __name__ == "__main__":
    download_esol()
