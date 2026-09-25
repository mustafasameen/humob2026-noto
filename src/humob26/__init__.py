"""humob26: origin-destination gap-filling pipeline for the HuMob Challenge 2026.

Predicts a contiguous, unobserved 58-day span of grid-cell-to-grid-cell
mobility flow from what is observed before and after it, and scores and
submits those predictions in the challenge's own format. See README.md for
the method and the command-line interface.
"""
from __future__ import annotations

__version__ = "0.1.0"
