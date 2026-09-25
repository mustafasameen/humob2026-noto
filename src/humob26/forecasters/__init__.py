"""Pluggable forecaster adapters (see `base.py` for the interface).

Importing this package is always cheap (numpy/pandas only); importing a
specific adapter module registers its class but only requires that
adapter's own framework once you actually call it.
"""
from .base import Forecaster, available, get, load_all, register

__all__ = ["Forecaster", "available", "get", "load_all", "register"]
