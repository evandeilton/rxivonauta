import asyncio
import logging
import argparse
import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ...existing imports...

# Rest of the main.py content remains the same, just moved to src/main.py
