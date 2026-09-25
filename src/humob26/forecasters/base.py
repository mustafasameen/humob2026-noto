"""The interface every forecaster adapter implements, and a name registry.

A forecaster only ever has to answer one question: given some context
series, what happens for the next `horizon` steps? Everything else in this
package (which two legs to run it on, how to blend them, which cells get a
joint call) lives in `foundation.py` and `combine.py`, not here.

This module has no heavy dependencies: importing it (or the registry) never
requires numpy-adjacent ML frameworks to be installed. Each adapter module
(`timesfm3`, `chronos2`, `patchtst_fm`) imports its own framework lazily,
inside its methods, so listing or selecting a forecaster by name never
requires that forecaster's framework to be present -- only actually calling
`predict`/`predict_joint`/`infill` does.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Forecaster(ABC):
    """A univariate (optionally also joint or infill-capable) forecaster.

    Every array below is float32/float64 numpy; `contexts` is a list (or
    stack) of 1-D series, one per output row, that need not all be the same
    length.
    """

    name: str

    @abstractmethod
    def predict(self, contexts, horizon: int):
        """(median, q10, q90), each shape (len(contexts), horizon)."""
        raise NotImplementedError

    def predict_joint(self, matrix, horizon: int):
        """(median, q10, q90) for a single multivariate call over all rows
        of `matrix` at once, for models that can share information across
        series in one forward pass. Not every model supports this."""
        raise NotImplementedError(f"{self.name} has no joint mode")

    def infill(self, series, mask):
        """Reconstruct the positions where `mask` is True, from the
        positions where it is False, for models that can condition on
        context on both sides of a gap in one pass. `series`/`mask` are
        (n, L); returns (median, q10, q90) of shape (n, L) each (the whole
        series, not just the masked span -- callers slice out what they
        need). Not every model supports this."""
        raise NotImplementedError(f"{self.name} has no infill mode")


_REGISTRY = {}


def register(name):
    """Class decorator: `@register("timesfm3")` makes a Forecaster
    subclass constructible via `get("timesfm3")`."""
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get(name):
    """The Forecaster subclass registered under `name`. Importing this
    function never imports any adapter module; only `available()` or an
    explicit `import humob26.forecasters.<name>` does, so listing names
    never requires every framework to be installed."""
    if name not in _REGISTRY:
        raise KeyError(f"no forecaster registered as {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available():
    """Names of every forecaster registered so far. Adapter modules
    register themselves on import, so this only reflects modules that have
    already been imported (see `load_all`)."""
    return sorted(_REGISTRY)


def load_all():
    """Import every bundled adapter module so its class registers itself,
    ignoring one whose framework is not installed. Safe to call even with
    only numpy/pandas installed: it will simply leave that adapter absent
    from `available()`."""
    for mod in ("timesfm3", "chronos2", "patchtst_fm"):
        try:
            __import__(f"humob26.forecasters.{mod}")
        except ImportError:
            continue
