import sys
from pathlib import Path

# Let tests import the synthetic-slide generator as a top-level module.
sys.path.insert(0, str(Path(__file__).parent))
