"""Pure helpers that turn the API's attention numbers into something to look at.

No Streamlit import here, so the heatmap logic is unit-tested in tests/test_ui_viz.py.
"""
from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence

ATTN_RGB = (42, 120, 214)      # blue: more attention = more opaque


def attention_alphas(weights: Sequence[float]) -> List[float]:
    """Scale weights so the most-attended token is fully colored (relative, not absolute)."""
    peak = max(weights, default=0.0)
    return [w / peak if peak > 0 else 0.0 for w in weights]


def attention_html(tokens: Sequence[str], weights: Optional[Sequence[float]] = None,
                   unk: Optional[Sequence[bool]] = None, max_alpha: float = 0.85) -> str:
    """The review as inline spans, each shaded by its attention weight.

    Tokens are HTML-escaped (they come from user text). Out-of-vocabulary tokens
    are struck through: the model saw them only as <unk>. Hovering shows the weight.
    """
    if weights is not None and len(weights) != len(tokens):
        raise ValueError("one attention weight per token is required")
    alphas = attention_alphas(weights) if weights is not None else [0.0] * len(tokens)
    unk = unk or [False] * len(tokens)
    spans = []
    for i, (tok, a, is_unk) in enumerate(zip(tokens, alphas, unk)):
        title = f"#{i}" + (f" · weight {weights[i]:.4f}" if weights is not None else "")
        title += " · out of vocabulary (<unk>)" if is_unk else ""
        style = f"background:rgba({ATTN_RGB[0]},{ATTN_RGB[1]},{ATTN_RGB[2]},{a * max_alpha:.3f});"
        if a * max_alpha > 0.5:
            style += "color:#fff;"
        if is_unk:
            style += "text-decoration:line-through;opacity:.7;"
        spans.append(f'<span title="{html.escape(title)}" style="{style}padding:1px 3px;'
                     f'border-radius:3px;margin:1px;display:inline-block">{html.escape(tok)}</span>')
    return f'<div style="line-height:1.9;font-size:0.95rem">{"".join(spans)}</div>'


def position_rows(weights: Sequence[float], tokens: Optional[Sequence[str]] = None) -> List[Dict]:
    """Rows for a position-by-weight chart (used when the words were never stored)."""
    peak = max(weights, default=0.0)
    return [{"position": i, "weight": float(w), "relative": float(w / peak) if peak else 0.0,
             "token": tokens[i] if tokens else f"#{i}"} for i, w in enumerate(weights)]


def peak_ratio(weights: Sequence[float]) -> float:
    """Top weight x length: 1 = uniform attention, n = all on one token."""
    return max(weights) * len(weights) if weights else 0.0
