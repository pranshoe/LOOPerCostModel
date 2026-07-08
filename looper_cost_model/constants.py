"""Fixed sizes and exceptions shared across the cost-model input representation.

These constants define the shape of the model's fixed-size input tensors and must be
kept in sync between the feature extractor (``data.features``) and the network
(``model``). Changing them changes the model input and requires re-extracting the
dataset.
"""

# Maximum number of affine transformations (interchange / reversal / skewing) the
# transformation-embedding LSTM represents per computation. Longer affine sequences
# (~0.1% of LOOPerSet schedules) are dropped during extraction.
MAX_NUM_TRANSFORMATIONS = 4

# Length of the tag vector encoding a single affine transformation (type + parameters).
MAX_TAGS = 16

# Maximum loop-nest depth per computation. Deeper nests are dropped.
MAX_DEPTH = 5

# Maximum length (number of operation vectors) of a computation's expression sequence.
MAX_EXPR_LEN = 66

# Maximum number of read/write accesses represented per computation.
MAX_ACCESSES = 15


class NbTransformationException(Exception):
    """A computation has more than ``MAX_NUM_TRANSFORMATIONS`` affine transformations."""


class NbAccessException(Exception):
    """A computation has more than ``MAX_ACCESSES`` memory accesses."""


class LoopsDepthException(Exception):
    """A computation's loop nest is deeper than ``MAX_DEPTH``."""


class ExpressionTooLongException(Exception):
    """A computation's expression sequence is longer than ``MAX_EXPR_LEN``."""
