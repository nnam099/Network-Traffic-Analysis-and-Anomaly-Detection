"""P13 deterministic extraction protocol primitives.

The package is data/extraction only.  It contains no model, scaler, score, or
training code.  A dataset adapter remains inactive until a raw source is
cryptographically reserved.
"""

from .portable_extractor import ExtractedRow, extract_rows, semantic_content_hash

__all__ = ["ExtractedRow", "extract_rows", "semantic_content_hash"]
