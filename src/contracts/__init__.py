"""The calling convention: what a verb returns, and what "not chosen" means.

Every capability in this repository reaches an operator through `python -m <pkg>`
and through Python. Both callers must give the same answer, which takes two
agreements: what a result looks like (:mod:`.reporting`) and how a caller says
"I chose this" (:mod:`.options`). Both were already practised in several places
under several names; this package names them once.

Stdlib only. Nothing here imports from `src`, from `datagen` or from any
third-party package, which is what lets this package add no edge to the dependency
stack. Anything that needs an import is a utility and belongs in `src/utility/`.

:class:`Rendered` and :class:`Structured` are structural Protocols, so a class
conforms by having the method rather than by inheriting or registering, and
implementers never import them. :data:`UNSET` is the exception, because a
sentinel has to be the same object to be worth anything.

The reference implementations live outside this package.
`src/robot/grasping/deep/train/api.py` shows the whole shape end to end: a noun,
keyword-only factories, one verb, a frozen typed report, and side effects as
separate methods. `src/robot/execution/real_cell/preflight.py` holds the
smallest complete report: a StrEnum status, a frozen check, a frozen report, a
derived `.ok`, one `render()` and one pure producer function.

See `README.md` for the five rules.
"""

from __future__ import annotations

from src.contracts.options import UNSET, Maybe, chosen, resolve
from src.contracts.reporting import Rendered, Structured

__all__ = ["UNSET", "Maybe", "Rendered", "Structured", "chosen", "resolve"]
