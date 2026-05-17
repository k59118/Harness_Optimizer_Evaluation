from __future__ import annotations


class BackboneError(RuntimeError):
    """Base error for backbone adapter failures."""


class BackboneConfigError(BackboneError):
    """Raised when a backbone config is invalid or incomplete."""


class BackboneRuntimeError(BackboneError):
    """Raised when a backbone runtime cannot complete normally."""
