"""Multi-discipline formula handwriting generation.

Supports LaTeX math formulas (fractions, integrals, matrices, etc.)
and chemical equation rendering in handwritten style.
"""

from handwrite.formula.chemistry_parser import (
    ChemArrow,
    ChemCompound,
    ChemEquation,
    ChemToken,
    parse_chemistry,
)
from handwrite.formula.formula_engine import FormulaConfig, render_chemistry, render_latex_formula
from handwrite.formula.formula_layout import BBox, FormulaLayout, LayoutConfig
from handwrite.formula.formula_renderer import FormulaRenderer, RenderConfig
from handwrite.formula.latex_parser import (
    FractionNode,
    GreekNode,
    IntegralNode,
    MatrixNode,
    ParseNode,
    SqrtNode,
    SubscriptNode,
    SumNode,
    SuperscriptNode,
    TextNode,
    parse_latex,
)

__all__ = [
    # Engine
    "FormulaConfig",
    "render_latex_formula",
    "render_chemistry",
    # LaTeX parser
    "parse_latex",
    "ParseNode",
    "TextNode",
    "FractionNode",
    "SuperscriptNode",
    "SubscriptNode",
    "SqrtNode",
    "IntegralNode",
    "SumNode",
    "MatrixNode",
    "GreekNode",
    # Layout
    "FormulaLayout",
    "LayoutConfig",
    "BBox",
    # Renderer
    "FormulaRenderer",
    "RenderConfig",
    # Chemistry
    "parse_chemistry",
    "ChemEquation",
    "ChemCompound",
    "ChemToken",
    "ChemArrow",
]
