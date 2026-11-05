# Environment adapters

The engine-independent contracts in `colonist_rl.contracts` and topology in
`colonist_rl.board` are shared by simulation, Colonist, and physical-board input.
Environment-specific code lives in `src/colonist_rl/adapters/`:

| Module | Responsibility |
| --- | --- |
| `observation_io` | Versioned observation JSON and visibility-filtered event exports |
| `calibration` | Affine board-lattice to image coordinates |
| `capture` | Capture metadata, reader protocol, verified image annotations |
| `errors` | Shared perception exception |
| `colonist.browser` | Optional Playwright transport and browser lifecycle |
| `colonist.execution` | Authorized UI actions and verified before/after transitions |
| `physical` | Local photo ingestion through a supplied reader |

New code can import these modules directly. Existing `colonist_rl.perception`
and `colonist_rl.browser` imports remain supported and refer to the same objects.
The legacy browser module also retains its launch-factory patch boundary.
Observation/capture format names and versions are unchanged.

## Colonist

```python
from colonist_rl.adapters.colonist.browser import PlaywrightSession
from colonist_rl.adapters.colonist.execution import BrowserAdapter
```

Browser transport and execution still require the same explicit session setup,
verified reader, calibration, controls, and authorization described in
[browser integration](browser.md) and [capture handling](perception.md).
Importing the browser implementation does not launch Chromium or import
Playwright until a session is opened.

## Physical-board photos

```python
import json
from pathlib import Path

from colonist_rl.adapters.capture import AnnotatedCaptureReader
from colonist_rl.adapters.physical import PhysicalBoardAdapter

annotation = json.loads(Path("board-annotation.json").read_text())
adapter = PhysicalBoardAdapter(reader=AnnotatedCaptureReader([annotation]))
capture = adapter.read_photo("board-photo.jpg")
observation = capture.observation
```

The annotation format is the existing `colonist-rl.capture` envelope documented
in [capture handling](perception.md). It binds complete, explicitly verified
player state to the exact photo bytes by SHA-256. An unknown image, invalid image,
unverified annotation, or missing required observation fields is rejected.

`PhysicalBoardAdapter` requires a reader. A future recognizer can implement
`CaptureReader.read(image_bytes) -> CaptureObservation`; the adapter preserves
its verification, stability, and confidence metadata and propagates its errors.
There is no automatic terrain/number recognizer or physical action executor.
Photos alone do not establish private resources, development cards, legal
actions, or decision context, and missing values are never filled with zeros.
Callers must establish that complete state before using it for decisions.

The physical/shared modules do not import Colonist browser execution, Playwright,
Torch, Catanatron, or the simulator. Shared affine calibration is useful for
verified planar captures; arbitrary perspective photos may require a separate
geometry model and reader.
