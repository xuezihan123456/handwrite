"""Tests for the formula handwriting module."""

from __future__ import annotations

import pytest
from PIL import Image

from handwrite.formula.chemistry_parser import (
    ArrowType,
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


# ======================================================================
# LaTeX parser tests
# ======================================================================


class TestParseLatex:
    """Tests for the LaTeX parser."""

    def test_plain_text(self) -> None:
        nodes = parse_latex("abc")
        assert len(nodes) == 1
        assert isinstance(nodes[0], TextNode)
        assert nodes[0].text == "abc"

    def test_fraction(self) -> None:
        nodes = parse_latex(r"\frac{a}{b}")
        assert len(nodes) == 1
        frac = nodes[0]
        assert isinstance(frac, FractionNode)
        assert len(frac.numerator) == 1
        assert isinstance(frac.numerator[0], TextNode)
        assert frac.numerator[0].text == "a"
        assert len(frac.denominator) == 1
        assert isinstance(frac.denominator[0], TextNode)
        assert frac.denominator[0].text == "b"

    def test_superscript(self) -> None:
        nodes = parse_latex(r"x^{2}")
        assert len(nodes) >= 2
        sup = None
        for n in nodes:
            if isinstance(n, SuperscriptNode):
                sup = n
                break
        assert sup is not None
        assert len(sup.content) == 1

    def test_subscript(self) -> None:
        nodes = parse_latex(r"x_{i}")
        sub = None
        for n in nodes:
            if isinstance(n, SubscriptNode):
                sub = n
                break
        assert sub is not None
        assert len(sub.content) == 1

    def test_single_char_superscript(self) -> None:
        nodes = parse_latex(r"x^2")
        sup = None
        for n in nodes:
            if isinstance(n, SuperscriptNode):
                sup = n
                break
        assert sup is not None
        assert len(sup.content) == 1

    def test_sqrt(self) -> None:
        nodes = parse_latex(r"\sqrt{x}")
        assert len(nodes) == 1
        assert isinstance(nodes[0], SqrtNode)
        assert len(nodes[0].content) == 1

    def test_integral(self) -> None:
        nodes = parse_latex(r"\int_{0}^{1} x\,dx")
        integral = None
        for n in nodes:
            if isinstance(n, IntegralNode):
                integral = n
                break
        assert integral is not None
        assert len(integral.lower) >= 1
        assert len(integral.upper) >= 1

    def test_sum(self) -> None:
        nodes = parse_latex(r"\sum_{i=0}^{n} x_i")
        summation = None
        for n in nodes:
            if isinstance(n, SumNode):
                summation = n
                break
        assert summation is not None

    def test_greek_letter(self) -> None:
        nodes = parse_latex(r"\alpha + \beta")
        greek_nodes = [n for n in nodes if isinstance(n, GreekNode)]
        assert len(greek_nodes) == 2
        assert greek_nodes[0].name == "alpha"
        assert greek_nodes[1].name == "beta"

    def test_matrix(self) -> None:
        nodes = parse_latex(r"\begin{matrix} a & b \\ c & d \end{matrix}")
        matrix = None
        for n in nodes:
            if isinstance(n, MatrixNode):
                matrix = n
                break
        assert matrix is not None
        assert len(matrix.rows) == 2
        assert len(matrix.rows[0]) == 2
        assert len(matrix.rows[1]) == 2

    def test_empty_input(self) -> None:
        nodes = parse_latex("")
        assert nodes == []

    def test_nested_fraction(self) -> None:
        nodes = parse_latex(r"\frac{\frac{a}{b}}{c}")
        assert len(nodes) == 1
        outer = nodes[0]
        assert isinstance(outer, FractionNode)
        assert len(outer.numerator) == 1
        assert isinstance(outer.numerator[0], FractionNode)

    def test_complex_expression(self) -> None:
        nodes = parse_latex(r"E = mc^{2}")
        assert len(nodes) > 0
        # Should not raise any exception.

    def test_bmatrix(self) -> None:
        nodes = parse_latex(r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}")
        matrix = None
        for n in nodes:
            if isinstance(n, MatrixNode):
                matrix = n
                break
        assert matrix is not None
        assert matrix.kind == "bmatrix"

    def test_infinity(self) -> None:
        nodes = parse_latex(r"\infty")
        greek = None
        for n in nodes:
            if isinstance(n, GreekNode):
                greek = n
                break
        assert greek is not None
        assert greek.name == "infty"


# ======================================================================
# Layout tests
# ======================================================================


class TestFormulaLayout:
    """Tests for the formula layout engine."""

    def test_text_layout(self) -> None:
        layout = FormulaLayout(LayoutConfig(base_font_size=40))
        items = layout.layout([TextNode("abc")])
        assert len(items) == 1
        assert items[0].kind == "text"
        assert items[0].bbox.width > 0
        assert items[0].bbox.height == 40

    def test_fraction_layout_produces_line(self) -> None:
        layout = FormulaLayout(LayoutConfig(base_font_size=40))
        nodes = parse_latex(r"\frac{a}{b}")
        items = layout.layout(nodes)
        kinds = [item.kind for item in items]
        assert "fraction_line" in kinds

    def test_fraction_layout_height(self) -> None:
        layout = FormulaLayout(LayoutConfig(base_font_size=40))
        frac_items = layout.layout(parse_latex(r"\frac{a}{b}"))
        text_items = layout.layout(parse_latex("a"))
        # Fraction should be taller than plain text.
        frac_max_y = max(item.bbox.bottom for item in frac_items)
        text_max_y = max(item.bbox.bottom for item in text_items)
        assert frac_max_y > text_max_y

    def test_sqrt_layout_produces_radical(self) -> None:
        layout = FormulaLayout(LayoutConfig(base_font_size=40))
        items = layout.layout(parse_latex(r"\sqrt{x}"))
        kinds = [item.kind for item in items]
        assert "radical_sign" in kinds

    def test_integral_layout_produces_sign(self) -> None:
        layout = FormulaLayout(LayoutConfig(base_font_size=40))
        items = layout.layout(parse_latex(r"\int_{0}^{1}"))
        kinds = [item.kind for item in items]
        assert "integral_sign" in kinds

    def test_sum_layout_produces_sign(self) -> None:
        layout = FormulaLayout(LayoutConfig(base_font_size=40))
        items = layout.layout(parse_latex(r"\sum_{i=0}^{n}"))
        kinds = [item.kind for item in items]
        assert "sum_sign" in kinds

    def test_matrix_layout_produces_brackets(self) -> None:
        layout = FormulaLayout(LayoutConfig(base_font_size=40))
        items = layout.layout(parse_latex(r"\begin{matrix} a & b \\ c & d \end{matrix}"))
        kinds = [item.kind for item in items]
        assert "matrix_bracket" in kinds

    def test_empty_nodes_returns_empty(self) -> None:
        layout = FormulaLayout()
        items = layout.layout([])
        assert items == []


# ======================================================================
# Renderer tests
# ======================================================================


class TestFormulaRenderer:
    """Tests for the formula renderer."""

    def test_render_returns_grayscale_image(self) -> None:
        nodes = parse_latex(r"\frac{a}{b}")
        layout = FormulaLayout(LayoutConfig(base_font_size=40))
        items = layout.layout(nodes)
        min_x = min(item.bbox.x for item in items)
        min_y = min(item.bbox.y for item in items)
        max_x = max(item.bbox.right for item in items)
        max_y = max(item.bbox.bottom for item in items)
        canvas_bbox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)

        renderer = FormulaRenderer(RenderConfig(seed=42))
        image = renderer.render(items, canvas_bbox)

        assert isinstance(image, Image.Image)
        assert image.mode == "L"
        assert image.size[0] > 0
        assert image.size[1] > 0

    def test_render_with_padding(self) -> None:
        nodes = parse_latex("x")
        layout = FormulaLayout(LayoutConfig(base_font_size=20))
        items = layout.layout(nodes)
        min_x = min(item.bbox.x for item in items)
        min_y = min(item.bbox.y for item in items)
        max_x = max(item.bbox.right for item in items)
        max_y = max(item.bbox.bottom for item in items)
        canvas_bbox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)

        renderer = FormulaRenderer(RenderConfig(padding=(30, 30, 30, 30), seed=42))
        image = renderer.render(items, canvas_bbox)

        assert image.size[0] > 30
        assert image.size[1] > 30

    def test_render_empty(self) -> None:
        renderer = FormulaRenderer()
        image = renderer.render([], BBox())
        assert isinstance(image, Image.Image)


# ======================================================================
# Chemistry parser tests
# ======================================================================


class TestParseChemistry:
    """Tests for the chemistry parser."""

    def test_simple_equation(self) -> None:
        eq = parse_chemistry("H2 + O2 -> H2O")
        assert len(eq.reactants) == 2
        assert len(eq.products) == 1
        assert eq.arrow.arrow_type == ArrowType.FORWARD

    def test_subscript_numbers(self) -> None:
        eq = parse_chemistry("H2O")
        assert len(eq.reactants) == 1
        compound = eq.reactants[0]
        number_tokens = [t for t in compound.tokens if t.kind == "number"]
        assert len(number_tokens) == 1
        assert number_tokens[0].text == "2"

    def test_charge(self) -> None:
        eq = parse_chemistry("Na+")
        compound = eq.reactants[0]
        charge_tokens = [t for t in compound.tokens if t.kind == "charge"]
        assert len(charge_tokens) == 1
        assert charge_tokens[0].text == "+"

    def test_yield_arrow(self) -> None:
        eq = parse_chemistry("A -> B")
        assert eq.arrow.arrow_type == ArrowType.FORWARD

    def test_equilibrium_arrow(self) -> None:
        eq = parse_chemistry("A <=> B")
        assert eq.arrow.arrow_type == ArrowType.EQUILIBRIUM

    def test_unicode_arrow(self) -> None:
        eq = parse_chemistry("A \u2192 B")
        assert eq.arrow.arrow_type == ArrowType.YIELD

    def test_state_annotation(self) -> None:
        eq = parse_chemistry("NaCl(aq) + AgNO3(aq)")
        # State annotations should be parsed.
        assert len(eq.reactants) >= 1

    def test_multiple_products(self) -> None:
        eq = parse_chemistry("A + B -> C + D")
        assert len(eq.products) == 2

    def test_empty_input(self) -> None:
        eq = parse_chemistry("")
        assert eq.reactants == []
        assert eq.products == []

    def test_complex_equation(self) -> None:
        eq = parse_chemistry("2H2 + O2 -> 2H2O")
        assert len(eq.reactants) >= 2
        assert len(eq.products) >= 1

    def test_delta_condition(self) -> None:
        eq = parse_chemistry("CaCO3 -> CaO + CO2 \u0394")
        assert len(eq.products) >= 2


# ======================================================================
# Engine tests
# ======================================================================


class TestRenderLatexFormula:
    """Integration tests for render_latex_formula."""

    def test_fraction_image(self) -> None:
        img = render_latex_formula(r"\frac{1}{2}", FormulaConfig(seed=42))
        assert isinstance(img, Image.Image)
        assert img.mode == "L"
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_complex_formula(self) -> None:
        img = render_latex_formula(
            r"E = mc^{2} + \frac{\partial f}{\partial x}",
            FormulaConfig(font_size=36, seed=42),
        )
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0

    def test_integral_formula(self) -> None:
        img = render_latex_formula(
            r"\int_{0}^{\infty} e^{-x} \, dx",
            FormulaConfig(seed=42),
        )
        assert isinstance(img, Image.Image)

    def test_sum_formula(self) -> None:
        img = render_latex_formula(
            r"\sum_{i=0}^{n} x_i^{2}",
            FormulaConfig(seed=42),
        )
        assert isinstance(img, Image.Image)

    def test_matrix_formula(self) -> None:
        img = render_latex_formula(
            r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}",
            FormulaConfig(seed=42),
        )
        assert isinstance(img, Image.Image)

    def test_sqrt_formula(self) -> None:
        img = render_latex_formula(r"\sqrt{a^{2} + b^{2}}", FormulaConfig(seed=42))
        assert isinstance(img, Image.Image)

    def test_greek_letters(self) -> None:
        img = render_latex_formula(r"\alpha + \beta = \gamma", FormulaConfig(seed=42))
        assert isinstance(img, Image.Image)

    def test_output_size_config(self) -> None:
        img = render_latex_formula(
            r"\frac{1}{2}",
            FormulaConfig(output_width=200, output_height=100, seed=42),
        )
        assert isinstance(img, Image.Image)
        # Width should be at most 200 (aspect ratio preserved).
        assert img.size[0] <= 200

    def test_empty_latex(self) -> None:
        img = render_latex_formula("", FormulaConfig(seed=42))
        assert isinstance(img, Image.Image)

    def test_reproducible_with_seed(self) -> None:
        img1 = render_latex_formula(r"\frac{a}{b}", FormulaConfig(seed=123))
        img2 = render_latex_formula(r"\frac{a}{b}", FormulaConfig(seed=123))
        assert list(img1.getdata()) == list(img2.getdata())


class TestRenderChemistry:
    """Integration tests for render_chemistry."""

    def test_simple_equation_image(self) -> None:
        img = render_chemistry("H2 + O2 -> H2O", FormulaConfig(seed=42))
        assert isinstance(img, Image.Image)
        assert img.mode == "L"
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_ionic_equation(self) -> None:
        img = render_chemistry("Na+ + Cl- -> NaCl", FormulaConfig(seed=42))
        assert isinstance(img, Image.Image)

    def test_equilibrium(self) -> None:
        img = render_chemistry("N2 + 3H2 <=> 2NH3", FormulaConfig(seed=42))
        assert isinstance(img, Image.Image)

    def test_combustion(self) -> None:
        img = render_chemistry("CH4 + 2O2 -> CO2 + 2H2O", FormulaConfig(seed=42))
        assert isinstance(img, Image.Image)

    def test_empty_equation(self) -> None:
        img = render_chemistry("", FormulaConfig(seed=42))
        assert isinstance(img, Image.Image)
