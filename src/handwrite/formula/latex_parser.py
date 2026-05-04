"""LaTeX formula parser using a state-machine tokenizer.

Supports: fractions, superscripts/subscripts, square roots, integrals,
sums, matrices, and Greek letters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParseNode:
    """Base class for all LaTeX AST nodes."""
    pass


@dataclass(frozen=True)
class TextNode(ParseNode):
    """A run of literal characters."""
    text: str


@dataclass(frozen=True)
class FractionNode(ParseNode):
    r"""``\frac{numerator}{denominator}``."""
    numerator: list[ParseNode]
    denominator: list[ParseNode]


@dataclass(frozen=True)
class SuperscriptNode(ParseNode):
    r"""``x^{exp}`` or ``x^a``."""
    content: list[ParseNode]


@dataclass(frozen=True)
class SubscriptNode(ParseNode):
    r"""``x_{sub}`` or ``x_a``."""
    content: list[ParseNode]


@dataclass(frozen=True)
class SqrtNode(ParseNode):
    r"""``\sqrt{content}`` or ``\sqrt[n]{content}``."""
    content: list[ParseNode]
    degree: Optional[list[ParseNode]] = None


@dataclass(frozen=True)
class IntegralNode(ParseNode):
    r"""``\int_{lower}^{upper}``."""
    lower: list[ParseNode] = field(default_factory=list)
    upper: list[ParseNode] = field(default_factory=list)


@dataclass(frozen=True)
class SumNode(ParseNode):
    r"""``\sum_{lower}^{upper}``."""
    lower: list[ParseNode] = field(default_factory=list)
    upper: list[ParseNode] = field(default_factory=list)


@dataclass(frozen=True)
class MatrixNode(ParseNode):
    r"""``\begin{matrix}...\end{matrix}`` and variants (pmatrix, bmatrix)."""
    rows: list[list[list[ParseNode]]] = field(default_factory=list)
    kind: str = "matrix"


@dataclass(frozen=True)
class GreekNode(ParseNode):
    r"""``\alpha``, ``\beta``, etc."""
    name: str


@dataclass(frozen=True)
class GroupNode(ParseNode):
    """A braced group ``{...}`` used internally during parsing."""
    children: list[ParseNode]


# ---------------------------------------------------------------------------
# Greek letter table
# ---------------------------------------------------------------------------

GREEK_MAP: dict[str, str] = {
    "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "delta": "\u03b4",
    "epsilon": "\u03b5", "zeta": "\u03b6", "eta": "\u03b7", "theta": "\u03b8",
    "iota": "\u03b9", "kappa": "\u03ba", "lambda": "\u03bb", "mu": "\u03bc",
    "nu": "\u03bd", "xi": "\u03be", "pi": "\u03c0", "rho": "\u03c1",
    "sigma": "\u03c3", "tau": "\u03c4", "upsilon": "\u03c5", "phi": "\u03c6",
    "chi": "\u03c7", "psi": "\u03c8", "omega": "\u03c9",
    "Alpha": "\u0391", "Beta": "\u0392", "Gamma": "\u0393", "Delta": "\u0394",
    "Epsilon": "\u0395", "Zeta": "\u0396", "Eta": "\u0397", "Theta": "\u0398",
    "Iota": "\u0399", "Kappa": "\u039a", "Lambda": "\u039b", "Mu": "\u039c",
    "Nu": "\u039d", "Xi": "\u039e", "Pi": "\u03a0", "Rho": "\u03a1",
    "Sigma": "\u03a3", "Tau": "\u03a4", "Upsilon": "\u03a5", "Phi": "\u03a6",
    "Chi": "\u03a7", "Psi": "\u03a8", "Omega": "\u03a9",
    "infty": "\u221e", "partial": "\u2202", "nabla": "\u2207",
    "times": "\u00d7", "cdot": "\u00b7", "pm": "\u00b1", "mp": "\u2213",
    "leq": "\u2264", "geq": "\u2265", "neq": "\u2260",
    "approx": "\u2248", "equiv": "\u2261",
    "rightarrow": "\u2192", "leftarrow": "\u2190",
    "Rightarrow": "\u21d2", "Leftarrow": "\u21d0",
    "forall": "\u2200", "exists": "\u2203",
    "in": "\u2208", "notin": "\u2209",
    "subset": "\u2282", "supset": "\u2283",
    "cup": "\u222a", "cap": "\u2229",
    "emptyset": "\u2205", "ldots": "\u2026", "cdots": "\u22ef",
    "quad": "\u2003", "qquad": "\u2003\u2003",
    "sin": "sin", "cos": "cos", "tan": "tan",
    "log": "log", "ln": "ln", "lim": "lim",
    "max": "max", "min": "min", "exp": "exp",
}


# ---------------------------------------------------------------------------
# Tokenizer state machine
# ---------------------------------------------------------------------------

class _State(Enum):
    """Tokenizer states."""
    TEXT = auto()
    BACKSLASH = auto()
    COMMAND = auto()
    BRACE_OPEN = auto()
    SUBSCRIPT = auto()
    SUPERSCRIPT = auto()


@dataclass
class _Tokenizer:
    """State-machine based LaTeX tokenizer.

    Converts raw LaTeX string into a flat token stream consumed by the parser.
    """

    source: str
    pos: int = 0

    # -- character access --

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

    # -- public tokenise --

    def tokenise(self) -> list[_Token]:
        tokens: list[_Token] = []
        state = _State.TEXT
        buf: list[str] = []

        while not self._eof():
            ch = self._peek()

            if state is _State.TEXT:
                if ch == "\\":
                    if buf:
                        tokens.append(_Token("text", "".join(buf)))
                        buf.clear()
                    state = _State.BACKSLASH
                    self._advance()
                elif ch == "{":
                    if buf:
                        tokens.append(_Token("text", "".join(buf)))
                        buf.clear()
                    tokens.append(_Token("lbrace", "{"))
                    self._advance()
                elif ch == "}":
                    if buf:
                        tokens.append(_Token("text", "".join(buf)))
                        buf.clear()
                    tokens.append(_Token("rbrace", "}"))
                    self._advance()
                elif ch == "^":
                    if buf:
                        tokens.append(_Token("text", "".join(buf)))
                        buf.clear()
                    tokens.append(_Token("sup", "^"))
                    self._advance()
                elif ch == "_":
                    if buf:
                        tokens.append(_Token("text", "".join(buf)))
                        buf.clear()
                    tokens.append(_Token("sub", "_"))
                    self._advance()
                elif ch == "&":
                    if buf:
                        tokens.append(_Token("text", "".join(buf)))
                        buf.clear()
                    tokens.append(_Token("ampersand", "&"))
                    self._advance()
                elif ch == "\n":
                    if buf:
                        tokens.append(_Token("text", "".join(buf)))
                        buf.clear()
                    tokens.append(_Token("newline", "\n"))
                    self._advance()
                else:
                    buf.append(ch)
                    self._advance()

            elif state is _State.BACKSLASH:
                if ch is not None and ch.isalpha():
                    buf.append(ch)
                    self._advance()
                    state = _State.COMMAND
                elif ch is not None:
                    # Escaped single character like \\ or \{ or \,
                    buf.append(ch)
                    tokens.append(_Token("command", "".join(buf)))
                    buf.clear()
                    self._advance()
                    state = _State.TEXT
                else:
                    break

            elif state is _State.COMMAND:
                if ch is not None and ch.isalpha():
                    buf.append(ch)
                    self._advance()
                else:
                    tokens.append(_Token("command", "".join(buf)))
                    buf.clear()
                    state = _State.TEXT

        # Flush remaining buffer.
        if buf:
            if state is _State.COMMAND:
                tokens.append(_Token("command", "".join(buf)))
            else:
                tokens.append(_Token("text", "".join(buf)))

        return tokens


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


# ---------------------------------------------------------------------------
# Recursive descent parser
# ---------------------------------------------------------------------------

@dataclass
class _Parser:
    """Parses a flat token stream into a tree of ParseNodes."""

    tokens: list[_Token]
    pos: int = 0

    def _peek(self) -> Optional[_Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self) -> _Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _eof(self) -> bool:
        return self.pos >= len(self.tokens)

    def _expect(self, kind: str) -> _Token:
        tok = self._advance()
        if tok.kind != kind:
            raise ValueError(f"Expected token {kind!r}, got {tok.kind!r} ({tok.value!r})")
        return tok

    # -- entry point --

    def parse(self) -> list[ParseNode]:
        nodes = self._parse_sequence(stop_at_rbrace=False)
        return self._flatten(nodes)

    # -- core parsing --

    def _parse_sequence(self, *, stop_at_rbrace: bool) -> list[ParseNode]:
        """Parse a sequence of nodes until EOF or `}`."""
        nodes: list[ParseNode] = []
        while not self._eof():
            tok = self._peek()
            if tok is None:
                break
            if stop_at_rbrace and tok.kind == "rbrace":
                break
            if tok.kind == "rbrace":
                break
            node = self._parse_one()
            if node is not None:
                nodes.append(node)
        return nodes

    def _parse_one(self) -> Optional[ParseNode]:
        tok = self._peek()
        if tok is None:
            return None

        if tok.kind == "text":
            self._advance()
            return TextNode(tok.value) if tok.value else None

        if tok.kind == "lbrace":
            return self._parse_group()

        if tok.kind == "sup":
            self._advance()
            content = self._parse_single_arg()
            return SuperscriptNode(content)

        if tok.kind == "sub":
            self._advance()
            content = self._parse_single_arg()
            return SubscriptNode(content)

        if tok.kind == "command":
            return self._parse_command()

        if tok.kind in ("ampersand", "newline"):
            self._advance()
            return None

        # Skip unknown tokens.
        self._advance()
        return None

    def _parse_group(self) -> GroupNode:
        self._expect("lbrace")
        children = self._parse_sequence(stop_at_rbrace=True)
        if self._peek() and self._peek().kind == "rbrace":
            self._advance()
        return GroupNode(children)

    def _parse_single_arg(self) -> list[ParseNode]:
        """Parse a single argument: either a braced group or a single token."""
        tok = self._peek()
        if tok is None:
            return []
        if tok.kind == "lbrace":
            group = self._parse_group()
            return group.children
        # Single character token.
        node = self._parse_one()
        return [node] if node is not None else []

    def _parse_command(self) -> Optional[ParseNode]:
        tok = self._advance()
        cmd = tok.value

        if cmd == "frac":
            num = self._parse_single_arg()
            den = self._parse_single_arg()
            return FractionNode(num, den)

        if cmd == "sqrt":
            # Check for optional degree: \sqrt[n]{...}
            degree: Optional[list[ParseNode]] = None
            if self._peek() and self._peek().kind == "command" and self._peek().value == "[":
                # This won't happen since [ is not alpha, handle via text
                pass
            # Check for [degree] syntax - in our tokenizer, [ comes as text
            if self._peek() and self._peek().kind == "text" and self._peek().value.startswith("["):
                bracket_tok = self._advance()
                inner = bracket_tok.value[1:]  # strip leading [
                if "]" in inner:
                    degree_text = inner.split("]", 1)[0]
                    remainder = inner.split("]", 1)[1]
                    if degree_text:
                        degree = [TextNode(degree_text)]
                    if remainder:
                        # Push remainder back - uncommon case, skip for simplicity
                        pass
            content = self._parse_single_arg()
            return SqrtNode(content, degree=degree)

        if cmd == "int":
            node: ParseNode = IntegralNode()
            node = self._attach_limits(node)
            return node

        if cmd == "sum":
            node = SumNode()
            node = self._attach_limits(node)
            return node

        if cmd in GREEK_MAP:
            return GreekNode(cmd)

        # Matrix environments.
        if cmd in ("begin", "end"):
            return self._parse_matrix_env(cmd)

        # Unknown command: render as text.
        return TextNode(cmd)

    def _attach_limits(self, node: ParseNode) -> ParseNode:
        """Attach _{lower} and ^{upper} limits to an IntegralNode or SumNode."""
        lower: list[ParseNode] = []
        upper: list[ParseNode] = []

        for _ in range(2):  # at most one sub and one sup
            tok = self._peek()
            if tok is None:
                break
            if tok.kind == "sub":
                self._advance()
                lower = self._parse_single_arg()
            elif tok.kind == "sup":
                self._advance()
                upper = self._parse_single_arg()
            else:
                break

        if isinstance(node, IntegralNode):
            return IntegralNode(lower=lower, upper=upper)
        if isinstance(node, SumNode):
            return SumNode(lower=lower, upper=upper)
        return node

    def _parse_matrix_env(self, begin_or_end: str) -> Optional[ParseNode]:
        r"""Parse ``\begin{env}...\end{env}`` matrix block."""
        self._expect("lbrace")
        env_name_tok = self._peek()
        if env_name_tok is None:
            return None
        env_name = env_name_tok.value
        self._advance()
        self._expect("rbrace")

        if begin_or_end == "end":
            return None

        # Parse matrix body: rows separated by \\, cells by &
        rows: list[list[list[ParseNode]]] = [[[]]]
        current_cells: list[list[ParseNode]] = rows[0]

        while not self._eof():
            tok = self._peek()
            if tok is None:
                break
            # Check for \end
            if tok.kind == "command" and tok.value == "end":
                self._advance()
                self._expect("lbrace")
                self._advance()  # env name
                self._expect("rbrace")
                break

            if tok.kind == "ampersand":
                self._advance()
                current_cells.append([])
                continue

            if tok.kind == "newline":
                self._advance()
                continue

            if tok.kind == "command" and tok.value == "\\":
                self._advance()
                # Check for optional [spacing] after \\
                if (self._peek() and self._peek().kind == "text"
                        and self._peek().value.startswith("[")):
                    self._advance()
                rows.append([])
                current_cells = rows[-1]
                current_cells.append([])
                continue

            node = self._parse_one()
            if node is not None:
                if not current_cells:
                    current_cells.append([])
                current_cells[-1].append(node)

        # Flatten single-node cells.
        flat_rows = []
        for row in rows:
            flat_row = [self._flatten(cell) for cell in row if cell]
            if flat_row:
                flat_rows.append(flat_row)

        return MatrixNode(rows=flat_rows, kind=env_name)

    # -- helpers --

    def _flatten(self, nodes: list[ParseNode]) -> list[ParseNode]:
        """Unwrap GroupNode wrappers, merging adjacent TextNode."""
        flat: list[ParseNode] = []
        for node in nodes:
            if isinstance(node, GroupNode):
                flat.extend(self._flatten(node.children))
            else:
                flat.append(node)
        return self._merge_text(flat)

    @staticmethod
    def _merge_text(nodes: list[ParseNode]) -> list[ParseNode]:
        """Merge consecutive TextNode instances."""
        merged: list[ParseNode] = []
        for node in nodes:
            if merged and isinstance(merged[-1], TextNode) and isinstance(node, TextNode):
                merged[-1] = TextNode(merged[-1].text + node.text)
            else:
                merged.append(node)
        return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_latex(source: str) -> list[ParseNode]:
    r"""Parse a LaTeX math string into an AST.

    Args:
        source: LaTeX math string, e.g. ``r"\frac{a}{b} + \sqrt{x}"``.

    Returns:
        A list of :class:`ParseNode` representing the formula tree.

    Examples::

        >>> nodes = parse_latex(r"\frac{1}{2}")
        >>> nodes[0]
        FractionNode(numerator=[TextNode(text='1')], denominator=[TextNode(text='2')])
    """
    tokens = _Tokenizer(source).tokenise()
    return _Parser(tokens).parse()
