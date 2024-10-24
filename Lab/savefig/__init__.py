"""
Module for creating deterministic PNG and SVG files.
On import, sets numpy random seed and matplotlib SVG hash salt to 42.
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pathlib import Path

np.random.seed(42)
mpl.rcParams['svg.hashsalt'] = "42"

def savefig(name, **kwargs):
    """
    Wrapper around pyplot's savefig.
    Saves figure as 600 DPI PNG and as SVG, with `metadata` argument set to `{'Date': None}`.
    Images are saved in the `images` directory.
    """
    kwargs['metadata'] = {'Date': None}
    Path("images").mkdir(exist_ok=True)
    plt.savefig(f"images/{name}.png", dpi=600, **kwargs)
    plt.savefig(f"images/{name}.svg", **kwargs)
