"""Dataset building modules for financial time-series ML.

Phase 3 implements dataset building with walk-forward splits.
"""

from .builder import DatasetBuilder, build_dataset_from_config, SplitResult, DatasetMetadata

__all__ = [
    "DatasetBuilder",
    "build_dataset_from_config",
    "SplitResult",
    "DatasetMetadata",
]