"""Explicit quality exclusions, distinct from conversion failures."""


class RecordingRejected(ValueError):
    """Pickle-safe worker result carrying an auditable exclusion reason."""

    def __init__(self, reason, details):
        super().__init__(reason, details)
        self.reason = reason
        self.details = details

    def __str__(self):
        return self.reason
