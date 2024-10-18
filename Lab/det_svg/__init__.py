"""
Module for creating deterministic SVG files.
On import, sets numpy random seed and matplotlib SVG hash salt to 42.
Exported "savefig" function has metadata argument set to {'Date': None}.
"""

from functools import partial
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

np.random.seed(42)
mpl.rcParams['svg.hashsalt'] = "42"
savefig = partial(plt.savefig, metadata = {'Date': None})
