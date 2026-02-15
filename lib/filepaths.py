import os
from pathlib import Path


ROOT_DATASETS = Path(os.path.join(
    Path(__file__).resolve().parents[1],
    "datasets"
))


ROOT_FONTS = Path(os.path.join(
    Path(__file__).resolve().parents[1],
    "assets/visuals/fonts"
))