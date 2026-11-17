import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Use environment variable if set (for development), otherwise use package location
if os.environ.get("JUMP_PRODUCTION_ROOT"):
    PROJ_ROOT = Path(os.environ["JUMP_PRODUCTION_ROOT"]).resolve()
else:
    PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Visualization defaults
DEFAULT_DPI = 150  # Default DPI for figure saving (CLI --dpi option default)

# CPU count respecting cgroup/Slurm limits (with macOS fallback)
try:
    JUMP_CPUS: int = len(os.sched_getaffinity(0))
except AttributeError:
    # macOS doesn't have sched_getaffinity
    JUMP_CPUS: int = os.cpu_count() or 1
