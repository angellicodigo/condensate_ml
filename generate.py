"""Run all five regimes and write their GSD, XYZ, CSV and per-regime plots to OUTPUT_DIR."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from condensate_ml.helper.regimes import REGIMES, run_regime


def get_output_dir() -> Path:
    """Read OUTPUT_DIR from the environment or from a .env file next to this script."""
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir is None:
        raise RuntimeError("OUTPUT_DIR was not found in the environment or .env file")
    return Path(output_dir)


if __name__ == "__main__":
    output_dir = get_output_dir()
    for regime in REGIMES:
        run_regime(regime, output_dir)
