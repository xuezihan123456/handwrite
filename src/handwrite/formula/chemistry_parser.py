"""Chemical equation parser.

Parses chemical formulas and equations including:
- Subscript numbers: H2O -> H_2O
- Superscript charges: Na+ -> Na^+
- Reaction arrows: ->, ->, ->, =, reversible arrows
- Conditions above/below arrows: (triangle), catalyst, heat, etc.
- State annotations: (aq), (s), (g), (l)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChemToken:
    """A single chemical token: element, number, charge, etc."""
    text: str
    kind: str = "element"
    """Kind: 'element', 'number', 'charge', 'state', 'operator', 'space'."""


@dataclass(frozen=True)
class ChemCompound:
    """A chemical compound like H2O, NaCl, Ca(OH)2."""
    tokens: list[ChemToken] = field(default_factory=list)
    raw: str = ""


class ArrowType(Enum):
    """Types of chemical reaction arrows."""
    FORWARD = auto()          # ->
    REVERSE = auto()          # <->
    EQUILIBRIUM = auto()      # <=>
    YIELD = auto()            # ->
    NOT_YIELD = auto()        # -/->
    EQUALS = auto()           # =


@dataclass(frozen=True)
class ChemArrow:
    """A reaction arrow with optional conditions."""
    arrow_type: ArrowType = ArrowType.FORWARD
    condition_above: str = ""
    condition_below: str = ""
    raw: str = ""


@dataclass(frozen=True)
class ChemEquation:
    """A complete chemical equation: reactants -> products."""
    reactants: list[ChemCompound] = field(default_factory=list)
    arrow: ChemArrow = field(default_factory=ChemArrow)
    products: list[ChemCompound] = field(default_factory=list)
    raw: str = ""


# ---------------------------------------------------------------------------
# State annotations
# ---------------------------------------------------------------------------

_STATE_ANNOTATIONS = {"aq", "s", "g", "l", "aq.", "s.", "g.", "l."}

# Unicode superscript / subscript maps.
_SUPERSCRIPT_MAP = str.maketrans("0123456789+-()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾")
_SUBSCRIPT_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

@dataclass
class _ChemTokenizer:
    """State machine tokenizer for chemical equations."""

    source: str
    pos: int = 0

    def _peek(self) -> Optional[str]:
        if self.pos < len(self.source):
            return self.source[self.pos]
        return None

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        return ch

    def _eof(self) -> bool:
        return self.pos >= len(self.source)

    def _peek_str(self, length: int) -> str:
        return self.source[self.pos:self.pos + length]

    def tokenise(self) -> list[ChemToken]:
        tokens: list[ChemToken] = []
        buf: list[str] = []
        buf_kind: str = "element"

        def flush() -> None:
            nonlocal buf, buf_kind
            if buf:
                tokens.append(ChemToken(text="".join(buf), kind=buf_kind))
                buf = []
                buf_kind = "element"

        while not self._eof():
            ch = self._peek()

            # Whitespace.
            if ch is not None and ch.isspace():
                flush()
                self._advance()
                tokens.append(ChemToken(text=" ", kind="space"))
                continue

            # Reaction arrows (multi-character).
            if ch == "-":
                arrow = self._try_parse_arrow()
                if arrow is not None:
                    flush()
                    tokens.append(ChemToken(text=arrow, kind="arrow"))
                    continue

            if ch == "<":
                arrow = self._try_parse_arrow()
                if arrow is not None:
                    flush()
                    tokens.append(ChemToken(text=arrow, kind="arrow"))
                    continue

            if ch == "=":
                flush()
                self._advance()
                tokens.append(ChemToken(text="=", kind="arrow"))
                continue

            if ch == "\u2192":  # ->
                flush()
                self._advance()
                tokens.append(ChemToken(text="\u2192", kind="arrow"))
                continue

            if ch == "\u21cc":  # <=>
                flush()
                self._advance()
                tokens.append(ChemToken(text="\u21cc", kind="arrow"))
                continue

            # Digits -> subscript numbers in compounds.
            if ch is not None and ch.isdigit():
                if buf_kind != "number":
                    flush()
                    buf_kind = "number"
                buf.append(ch)
                self._advance()
                continue

            # Charges: + or - (not part of an arrow).
            if ch in ("+", "-") and not self._looks_like_arrow():
                # If preceded by a number or element, it's a charge.
                if buf or (tokens and tokens[-1].kind in ("element", "number")):
                    flush()
                    buf_kind = "charge"
                    buf.append(ch)
                    self._advance()
                    # Check for 2+, 3-, etc.
                    continue
                # Otherwise might be a standalone +/- token.
                flush()
                self._advance()
                tokens.append(ChemToken(text=ch, kind="operator"))
                continue

            # State annotations: (aq), (s), (g), (l) -- check before general parens.
            if ch == "(":
                state = self._try_parse_state()
                if state is not None:
                    flush()
                    tokens.append(ChemToken(text=state, kind="state"))
                    continue

            # Parentheses.
            if ch in ("(", ")"):
                flush()
                self._advance()
                tokens.append(ChemToken(text=ch, kind="element"))
                continue

            # Delta symbol for conditions.
            if ch == "\u0394" or ch == "\u03b4":
                flush()
                self._advance()
                tokens.append(ChemToken(text="\u0394", kind="condition"))
                continue

            # Default: element letters.
            if buf_kind != "element":
                flush()
                buf_kind = "element"
            buf.append(ch)
            self._advance()

        flush()
        return tokens

    def _try_parse_arrow(self) -> Optional[str]:
        """Try to parse a multi-character arrow starting at current position."""
        s = self._peek_str(4)
        if s.startswith("<=>"):
            self.pos += 3
            return "<=>"
        if s.startswith("->"):
            self.pos += 2
            return "->"
        if s.startswith("<->"):
            self.pos += 3
            return "<->"
        if s.startswith("-/->"):
            self.pos += 4
            return "-/->"
        if s.startswith("-->"):
            self.pos += 3
            return "-->"
        return None

    def _looks_like_arrow(self) -> bool:
        """Check if current position looks like part of an arrow."""
        s = self._peek_str(4)
        return s.startswith("->") or s.startswith("<=>") or s.startswith("<->")

    def _try_parse_state(self) -> Optional[str]:
        """Try to parse a state annotation like (aq), (s), etc."""
        saved = self.pos
        if self._peek() != "(":
            return None
        self._advance()  # skip (
        inner: list[str] = []
        while not self._eof() and self._peek() != ")":
            inner.append(self._advance())
        if self._eof():
            self.pos = saved
            return None
        self._advance()  # skip )
        text = "".join(inner).strip().lower().rstrip(".")
        if text in {"aq", "s", "g", "l"}:
            return f"({text})"
        self.pos = saved
        return None


def _parse_arrow_token(text: str) -> ChemArrow:
    """Convert an arrow token into a ChemArrow."""
    mapping = {
        "->": ArrowType.FORWARD,
        "-->": ArrowType.YIELD,
        "\u2192": ArrowType.YIELD,
        "\u21d2": ArrowType.YIELD,
        "<->": ArrowType.REVERSE,
        "<=>": ArrowType.EQUILIBRIUM,
        "\u21cc": ArrowType.EQUILIBRIUM,
        "=": ArrowType.EQUALS,
        "-/->": ArrowType.NOT_YIELD,
    }
    arrow_type = mapping.get(text, ArrowType.FORWARD)
    return ChemArrow(arrow_type=arrow_type, raw=text)


def _group_tokens_to_compounds(tokens: list[ChemToken]) -> list[ChemCompound]:
    """Group a flat token list into compounds separated by + or space."""
    compounds: list[ChemCompound] = []
    current: list[ChemToken] = []
    raw_parts: list[str] = []

    for tok in tokens:
        if tok.kind == "space":
            if current:
                compounds.append(ChemCompound(tokens=list(current), raw="".join(raw_parts)))
                current = []
                raw_parts = []
            continue
        if tok.text == "+" and tok.kind == "operator":
            if current:
                compounds.append(ChemCompound(tokens=list(current), raw="".join(raw_parts)))
                current = []
                raw_parts = []
            continue
        current.append(tok)
        raw_parts.append(tok.text)

    if current:
        compounds.append(ChemCompound(tokens=list(current), raw="".join(raw_parts)))

    return compounds


def _split_conditions(text: str) -> tuple[str, str]:
    """Split arrow conditions from a condition string.

    Returns (above, below) condition strings.
    """
    text = text.strip()
    if not text:
        return "", ""
    # Simple heuristic: if there's a newline, split on it.
    parts = text.split("\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_chemistry(source: str) -> ChemEquation:
    r"""Parse a chemical equation string.

    Args:
        source: Chemical equation string, e.g. ``"2H2 + O2 -> 2H2O"``.

    Returns:
        A :class:`ChemEquation` with reactants, arrow, and products.

    Examples::

        >>> eq = parse_chemistry("H2 + O2 -> H2O")
        >>> eq.reactants[0].raw
        'H2'
        >>> eq.arrow.arrow_type
        <ArrowType.FORWARD: 1>
    """
    tokens = _ChemTokenizer(source).tokenise()

    # Split into left (reactants) and right (products) by arrow.
    arrow_token: Optional[ChemToken] = None
    left_tokens: list[ChemToken] = []
    right_tokens: list[ChemToken] = []

    found_arrow = False
    for tok in tokens:
        if tok.kind == "arrow" and not found_arrow:
            arrow_token = tok
            found_arrow = True
            continue
        if found_arrow:
            right_tokens.append(tok)
        else:
            left_tokens.append(tok)

    # Extract condition tokens between arrow and compounds.
    # Conditions are typically placed above/below the arrow in display,
    # in text they may appear after the arrow or as special tokens.
    condition_above = ""
    condition_below = ""

    # Look for condition tokens in the source near the arrow.
    arrow_pos = source.find("->") if "->" in source else source.find("\u2192")
    if arrow_pos == -1:
        arrow_pos = source.find("=")
    if arrow_pos > 0:
        # Check for text above/below arrow if multi-line.
        lines = source.split("\n")
        if len(lines) >= 2:
            condition_above = lines[0].strip()
            condition_below = lines[-1].strip() if len(lines) > 2 else ""

    # Separate condition tokens from product tokens.
    product_tokens: list[ChemToken] = []
    for tok in right_tokens:
        if tok.kind == "condition":
            if not condition_above:
                condition_above = tok.text
            elif not condition_below:
                condition_below = tok.text
        elif tok.kind == "space" and not product_tokens:
            # Skip leading spaces after arrow.
            continue
        else:
            product_tokens.append(tok)

    arrow = ChemArrow(
        arrow_type=_parse_arrow_token(arrow_token.text).arrow_type if arrow_token else ArrowType.FORWARD,
        condition_above=condition_above,
        condition_below=condition_below,
        raw=arrow_token.text if arrow_token else "",
    )

    reactants = _group_tokens_to_compounds(left_tokens)
    products = _group_tokens_to_compounds(product_tokens)

    return ChemEquation(
        reactants=reactants,
        arrow=arrow,
        products=products,
        raw=source,
    )
