"""
GPU-accelerated rolling window metrics.

This module re-exports the rolling_pr_rmsd from pairwise for convenience.
For batched rolling window operations, see pairwise.rolling_pr_rmsd.
"""

from .pairwise import rolling_pr_rmsd

__all__ = ['rolling_pr_rmsd']