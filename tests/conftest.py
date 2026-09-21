import sys
from pathlib import Path

# Make agent.py / app.py importable when running `pytest` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
