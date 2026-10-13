"""CV strategies for the Parkinson's UPDRS experiments.

Two splitters are exported:

- ``splitter``: 20 % random row-level holdout (lab guide default for
  ``01_dummy`` / ``02_ridge``). Intentionally leaks ``patient_id`` across
  train/test — used as the comparable row-holdout baseline.
- ``grouped_splitter``: ``GroupKFold(n_splits=5)`` with ``patient_id``
  baked in via ``GroupedSplitter``. Passes the groups array to
  ``GroupKFold.split`` internally so ``skore.evaluate``'s ``splitter.split(X, y)``
  call works without a ``groups`` kwarg.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


# 20 % random row holdout — lab guide default for 01_dummy / 02_ridge.
splitter = 0.2


class GroupedSplitter:
    """Wrap ``GroupKFold`` so ``split(X, y)`` routes to ``split(X, y, groups)``.

    ``skore.evaluate`` and ``CrossValidationReport`` call
    ``splitter.split(X, y)`` without a ``groups`` argument. This wrapper
    bakes the ``groups`` array in at construction time so the call signature
    matches.

    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Group labels for each sample (e.g. ``patient_id``).
    n_splits : int, default=5
        Number of folds for ``GroupKFold``.
    """

    def __init__(self, groups, n_splits: int = 5) -> None:
        self._groups = np.asarray(groups)
        self._gkf = GroupKFold(n_splits=n_splits)

    def split(self, X, y=None, groups=None):  # noqa: ARG002
        """Yield (train_idx, test_idx) with no patient overlap per fold."""
        return self._gkf.split(X, y, groups=self._groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002
        """Return the number of folds."""
        return self._gkf.get_n_splits()


def make_grouped_splitter(n_splits: int = 5) -> "GroupedSplitter":
    """Return a ``GroupedSplitter`` loaded with the training ``patient_id`` groups.

    Lazy import of ``load_groups`` to avoid a circular dependency at module
    load time.

    Parameters
    ----------
    n_splits : int, default=5
        Number of ``GroupKFold`` folds.

    Returns
    -------
    GroupedSplitter
        Ready to pass as ``splitter=`` to ``skore.evaluate``.
    """
    from parkinsons.data import load_groups  # noqa: PLC0415

    return GroupedSplitter(groups=load_groups(), n_splits=n_splits)
