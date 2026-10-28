"""The attention heatmap helpers the Streamlit UI uses (ui/viz.py)."""
from __future__ import annotations

import re
import sys

import pytest

from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "ui"))
from viz import attention_alphas, attention_html, peak_ratio, position_rows  # noqa: E402


def alphas_in(html: str) -> list[float]:
    return [float(a) for a in re.findall(r"rgba\(\d+,\d+,\d+,([0-9.]+)\)", html)]


def test_top_token_is_darkest_and_shading_is_relative():
    assert attention_alphas([0.1, 0.4, 0.2]) == pytest.approx([0.25, 1.0, 0.5])
    out = alphas_in(attention_html(["a", "b", "c"], [0.1, 0.4, 0.2], max_alpha=0.8))
    assert out == pytest.approx([0.2, 0.8, 0.4], abs=1e-3)


def test_one_span_per_token_with_weight_tooltip():
    html = attention_html(["great", "film", "!"], [0.5, 0.3, 0.2])
    assert html.count("<span") == 3
    assert 'title="#0 · weight 0.5000"' in html


def test_user_text_is_escaped():
    html = attention_html(["<script>alert(1)</script>", "&"], [0.5, 0.5])
    assert "<script>" not in html and "&lt;script&gt;" in html and "&amp;" in html


def test_unknown_words_are_struck_through():
    html = attention_html(["ok", "zzq"], [0.5, 0.5], unk=[False, True])
    spans = html.split("<span")[1:]
    assert "line-through" not in spans[0] and "line-through" in spans[1]
    assert "out of vocabulary" in spans[1]


def test_no_attention_renders_plain_tokens():
    assert alphas_in(attention_html(["a", "b"])) == [0.0, 0.0]


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        attention_html(["a", "b"], [1.0])


def test_position_rows_and_peak_ratio():
    rows = position_rows([0.25, 0.5, 0.25])
    assert [r["position"] for r in rows] == [0, 1, 2]
    assert [r["relative"] for r in rows] == [0.5, 1.0, 0.5]
    assert rows[1]["token"] == "#1"
    assert position_rows([0.5, 0.5], ["a", "b"])[1]["token"] == "b"
    assert peak_ratio([0.25] * 4) == 1.0 and peak_ratio([1.0, 0, 0, 0]) == 4.0
