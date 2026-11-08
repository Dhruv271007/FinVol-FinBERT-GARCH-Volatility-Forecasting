"""Shared configuration; importing this module never reads or writes data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA")
OUTPUT = ROOT / "output"
CORRECTED_DATA = OUTPUT / "corrected_data"
PERSISTENCE_CAP = 0.999
VARIANCE_MARGIN = 1e-6
ISSUANCE_DELAY_MINUTES = 5
# Known end-coverage failure in the supplied cache. Keep the original files intact.
EXCLUDED = {"BRK-B": "Cached sentiment ends in October 2023; prices extend to June 2026."}
