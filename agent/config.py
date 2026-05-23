"""
Centralised configuration for the Customer Service Data Analyst Agent.
All tunable constants live here — model names, paths, limits.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Nebius Token Factory ────────────────────────────────────────────────────
NEBIUS_API_KEY: str = os.getenv("NEBIUS_API_KEY", "")

# Main agent model: Nemotron-3-Super-120b-a12b
#   - 120B hybrid MoE, explicitly optimised for multi-agent AI and complex reasoning
#   - 127 Tok/s — fast enough for interactive CLI / Streamlit use
#   - Hosted in us-central1
MAIN_MODEL: str = "nvidia/nemotron-3-super-120b-a12b"
MAIN_MODEL_BASE_URL: str = "https://api.tokenfactory.us-central1.nebius.com/v1/"

# Router model: NVIDIA-Nemotron-3-Nano-30B-A3B
#   - Compact MoE, 60 Tok/s, ~5x cheaper than the Super model
#   - Query classification is a single short structured-output call — nano is sufficient
#   - Hosted in eu-north1 (default endpoint)
ROUTER_MODEL: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
ROUTER_MODEL_BASE_URL: str = "https://api.tokenfactory.nebius.com/v1/"

# ── Agent loop ──────────────────────────────────────────────────────────────
MAX_ITERATIONS: int = 12          # Fallback to prevent infinite loops

# ── Paths ───────────────────────────────────────────────────────────────────
DATASET_PATH: str = os.path.join("data", "bitext_dataset.csv")
CHECKPOINTS_DB: str = os.path.join("checkpoints", "memory.db")
PROFILES_DIR: str = "profiles"
