"""OlympiadBench's official open-ended answer judge as a miles custom RM.

The equivalence logic is adapted from OpenBMB/OlympiadBench's
``eval/auto_scoring_judge.py`` (MIT license), retrieved 2026-08-13.
"""

from __future__ import annotations

import math
import re

import sympy as sp
from sympy import Eq, Pow, simplify, sympify
from sympy.parsing.latex import parse_latex


class AutoScoringJudge:
    """Official OlympiadBench symbolic and numerical equivalence judge."""

    def __init__(self) -> None:
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8

    @staticmethod
    def split_by_comma(expression: str) -> list[str]:
        bracket_depth = 0
        parts = []
        start = 0
        for index, character in enumerate(expression):
            if character in ("(", "["):
                bracket_depth += 1
            elif character in (")", "]"):
                bracket_depth -= 1
            elif character == "," and bracket_depth == 0:
                parts.append(expression[start:index].strip())
                start = index + 1
        if start < len(expression):
            parts.append(expression[start:].strip())
        return parts

    @staticmethod
    def trans_plus_minus_sign(expressions: list[str]) -> list[str]:
        expanded = []
        for expression in expressions:
            if "\\pm" in expression:
                expanded.extend((expression.replace("\\pm", "+"), expression.replace("\\pm", "-")))
            else:
                expanded.append(expression)
        return expanded

    def judge(self, ground_truth: str, prediction: str, precision: float = 1e-8) -> bool:
        precisions = precision if isinstance(precision, list) else [precision]
        try:
            ground_truth, prediction = self.preprocess(ground_truth, prediction)
        except Exception:
            return False
        if ground_truth == prediction:
            return True

        ground_truth = re.sub(r"[\u4e00-\u9fff]+", "", ground_truth)
        prediction = re.sub(r"[\u4e00-\u9fff]+", "", prediction)
        references = self.trans_plus_minus_sign(self.split_by_comma(ground_truth))
        predictions = self.trans_plus_minus_sign(self.split_by_comma(prediction))
        if len(precisions) <= 1:
            precisions *= len(references)
        if len(references) != len(predictions):
            return False

        index = -1
        while references:
            index = (index + 1) % len(references)
            reference = references[index]
            self.precision = precisions[index]
            for candidate in predictions:
                if self.is_equal(reference, candidate):
                    references.remove(reference)
                    predictions.remove(candidate)
                    precisions.remove(self.precision)
                    break
            else:
                return False
        return True

    @staticmethod
    def is_interval(expression: str) -> bool:
        return expression.startswith(("(", "[")) and expression.endswith((")", "]"))

    def sympy_sub_pi(self, expression):
        return expression.subs(self.pi, math.pi)

    def is_equal(self, reference: str, prediction: str) -> bool:
        if reference == prediction and reference:
            return True
        if self.is_interval(reference) and self.is_interval(prediction):
            try:
                if self.interval_equal(reference, prediction):
                    return True
            except Exception:
                return False
        try:
            if self.numerical_equal(reference, prediction):
                return True
        except Exception:
            pass
        try:
            if self.expression_equal(reference, prediction) and not (
                "=" in reference and "=" in prediction
            ):
                return True
        except Exception:
            pass
        try:
            return self.equation_equal(reference, prediction)
        except Exception:
            return False

    def numerical_equal(self, reference: str, prediction: str) -> bool:
        reference_value = float(reference)
        prediction_value = float(prediction)
        return any(
            abs(candidate - prediction_value) <= self.precision * 1.01
            for candidate in (reference_value / 100, reference_value, reference_value * 100)
        )

    def expression_equal(self, reference: str, prediction: str) -> bool:
        def extract_expression(expression: str) -> str:
            return expression.split("=", maxsplit=1)[-1].strip()

        reference_sympy = sympify(parse_latex(extract_expression(reference)))
        prediction_sympy = sympify(parse_latex(extract_expression(prediction)))
        if reference_sympy == prediction_sympy:
            return True
        reference_sympy = self.sympy_sub_pi(reference_sympy)
        prediction_sympy = self.sympy_sub_pi(prediction_sympy)
        if reference_sympy.has(sp.Symbol) != prediction_sympy.has(sp.Symbol):
            return False
        if not reference_sympy.has(sp.Symbol):
            if not (self.can_compute_power(reference_sympy) and self.can_compute_power(prediction_sympy)):
                return False
            return abs(reference_sympy.evalf() - prediction_sympy.evalf()) <= self.precision * 1.01
        return abs(simplify(reference_sympy - prediction_sympy).evalf()) < 1e-3

    @staticmethod
    def equation_equal(reference: str, prediction: str) -> bool:
        def simplify_equation(equation: str):
            left, right = equation.split("=")
            parsed = Eq(parse_latex(left), parse_latex(right))
            return simplify(parsed.lhs - parsed.rhs)

        reference_sympy = simplify_equation(reference)
        prediction_sympy = simplify_equation(prediction)
        ratios = (
            simplify(reference_sympy / prediction_sympy),
            simplify(prediction_sympy / reference_sympy),
        )
        return any(ratio.is_Integer and ratio != 0 for ratio in ratios)

    def interval_equal(self, reference: str, prediction: str) -> bool:
        def compare_interval(left: str, right: str) -> bool:
            if left[0] != right[0] or left[-1] != right[-1]:
                return False
            left_items = left.strip("[]()").split(",")
            right_items = right.strip("[]()").split(",")
            return all(self.expression_equal(a, b) for a, b in zip(left_items, right_items, strict=True))

        if reference == prediction:
            return True
        reference_intervals = reference.split("\\cup")
        prediction_intervals = prediction.split("\\cup")
        return len(reference_intervals) == len(prediction_intervals) and all(
            compare_interval(left, right)
            for left, right in zip(reference_intervals, prediction_intervals, strict=True)
        )

    def preprocess(self, reference: str, prediction: str) -> tuple[str, str]:
        def extract_boxed_content(latex: str) -> str:
            results = ""
            for match in re.finditer(r"\\boxed{", latex):
                start = match.end()
                end = start
                stack = 1
                while stack > 0 and end < len(latex):
                    if latex[end] == "{":
                        stack += 1
                    elif latex[end] == "}":
                        stack -= 1
                    end += 1
                if stack:
                    raise ValueError("mismatched braces in LaTeX answer")
                results += latex[start : end - 1] + ","
            if results:
                return results
            last_line = latex.strip().split("\n")[-1]
            dollar_answers = re.findall(r"\$(.*?)\$", last_line)
            return "".join(f"{answer}," for answer in dollar_answers) if dollar_answers else latex

        def replace_special_symbols(expression: str) -> str:
            if "\\in " in expression:
                expression = expression.split("\\in ", maxsplit=1)[1]
            for signal, replacement in self.special_signal_map.items():
                expression = expression.replace(signal, replacement)
            expression = expression.strip(  # noqa: B005 - official judge uses this character set
                "\n$,.:;^_=+`!@#$%^&*~，。"
            )
            return re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", r"\1", expression)

        return (
            replace_special_symbols(extract_boxed_content(reference)),
            replace_special_symbols(extract_boxed_content(prediction)),
        )

    @staticmethod
    def can_compute_power(expression) -> bool:
        if not isinstance(expression, Pow):
            return True
        base, exponent = expression.as_base_exp()
        return bool(base.is_number and exponent.is_number and abs(exponent.evalf()) <= 1000)

_JUDGE: AutoScoringJudge | None = None


def get_judge() -> AutoScoringJudge:
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = AutoScoringJudge()
    return _JUDGE


async def score_olympiadbench(args, samples, **kwargs) -> list[float]:
    """Score a batch of ``Sample`` objects with OlympiadBench's judge."""
    del args, kwargs
    judge = get_judge()
    return [
        float(
            judge.judge(
                str(sample.label),
                sample.response,
                float((sample.metadata or {}).get("precision", 1e-8)),
            )
        )
        for sample in samples
    ]


def build_preflight_probes(label, metadata) -> tuple[str, str]:
    del metadata
    return f"Answer: \\boxed{{{label}}}", "Answer: \\boxed{-987654321}"
