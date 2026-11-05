"""Read physical-board photos through an explicitly supplied capture reader.

A photo alone does not establish a player's hand, decision context, or legal
actions. A reader must supply a complete Observation with its confidence and
verification metadata. The shared AnnotatedCaptureReader supports exact,
user-verified photo annotations; automatic recognition is not implemented here.
"""

from pathlib import Path

from ..capture import CaptureObservation, CaptureReader


class PhysicalBoardAdapter:
    """Offline photo input with no browser connection or action execution.

    Inject an AnnotatedCaptureReader for existing verified annotations, or a
    future reader implementing CaptureReader. Reader errors propagate; this
    adapter never fills in missing player state or changes verification flags.
    """

    def __init__(self, *, reader: CaptureReader):
        self.reader = reader

    def read_photo(self, photo_path: str | Path) -> CaptureObservation:
        """Read the selected local image through the supplied reader."""
        return self.reader.read(Path(photo_path).read_bytes())
