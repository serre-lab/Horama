"""
Horama: Personal toolbox for experimenting with Feature Visualization
-------

This little personnal toolbox was created to help me play with modern
feature visualization.

Ref. Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained Optimization
     https://arxiv.org/abs/2306.06805
"""

__version__ = '0.3.0'

from .maco_fv import maco
from .fourier_fv import fourier
from .feature_accentuation import accentuation
from .plots import plot_maco
from .losses import dot_cossim
