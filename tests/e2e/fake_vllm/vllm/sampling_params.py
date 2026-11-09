"""Only what hl.data.generate imports for constrained decoding."""

import types


def StructuredOutputsParams(**kw):
    return types.SimpleNamespace(**kw)
