"""Segment assignment strategies for collaborative documents.

Two strategies are provided:

* **Manual** -- the caller supplies an explicit mapping of paragraph index to
  contributor index.
* **Round-robin** -- paragraphs are distributed evenly across contributors in
  order.
"""

from __future__ import annotations

from handwrite.collaboration.contributor import Contributor

# Each assignment is ``(paragraph_index, contributor_index)``.
Assignment = tuple[int, int]


def assign_segments(
    paragraph_count: int,
    contributors: list[Contributor],
    manual_mapping: list[int] | None = None,
) -> list[Assignment]:
    """Assign each paragraph to a contributor.

    Parameters
    ----------
    paragraph_count:
        Total number of paragraphs in the document.
    contributors:
        The list of contributors (2--6).
    manual_mapping:
        Optional explicit mapping where ``manual_mapping[i]`` is the
        contributor index for paragraph *i*.  Must have length equal to
        ``paragraph_count``.  When *None*, round-robin assignment is used.

    Returns
    -------
    list[Assignment]
        Ordered list of ``(paragraph_index, contributor_index)`` tuples.
    """
    _validate_contributors(contributors)

    if manual_mapping is not None:
        return _assign_manual(paragraph_count, contributors, manual_mapping)

    return assign_segments_round_robin(paragraph_count, contributors)


def assign_segments_round_robin(
    paragraph_count: int,
    contributors: list[Contributor],
) -> list[Assignment]:
    """Distribute paragraphs across contributors in round-robin order.

    The first paragraph goes to contributor 0, the second to contributor 1,
    and so on, wrapping around.
    """
    _validate_contributors(contributors)
    return [
        (pidx, pidx % len(contributors))
        for pidx in range(paragraph_count)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIN_CONTRIBUTORS = 2
_MAX_CONTRIBUTORS = 6


def _validate_contributors(contributors: list[Contributor]) -> None:
    if not isinstance(contributors, list):
        raise TypeError("contributors must be a list of Contributor instances")
    count = len(contributors)
    if count < _MIN_CONTRIBUTORS or count > _MAX_CONTRIBUTORS:
        raise ValueError(
            f"Expected 2--{_MAX_CONTRIBUTORS} contributors, got {count}"
        )


def _assign_manual(
    paragraph_count: int,
    contributors: list[Contributor],
    manual_mapping: list[int],
) -> list[Assignment]:
    if len(manual_mapping) != paragraph_count:
        raise ValueError(
            f"manual_mapping length ({len(manual_mapping)}) must equal "
            f"paragraph_count ({paragraph_count})"
        )
    contributor_count = len(contributors)
    assignments: list[Assignment] = []
    for pidx, cidx in enumerate(manual_mapping):
        if not isinstance(cidx, int):
            raise TypeError(
                f"manual_mapping[{pidx}] must be int, got {type(cidx).__name__}"
            )
        if cidx < 0 or cidx >= contributor_count:
            raise ValueError(
                f"manual_mapping[{pidx}] = {cidx} is out of range "
                f"[0, {contributor_count - 1}]"
            )
        assignments.append((pidx, cidx))
    return assignments
