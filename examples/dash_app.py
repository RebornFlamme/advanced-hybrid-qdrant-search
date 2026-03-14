"""Dash application for advanced Qdrant search."""

from __future__ import annotations

import dash
from dash import Input, Output, dcc, html

from qdrant_advanced_search import QueryExecutor, SearchResult

# ---------------------------------------------------------------------------
# Module-level executor (model loads once at import time)
# ---------------------------------------------------------------------------

executor = QueryExecutor()

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_CARD_STYLE = {
    "border": "1px solid #e0e0e0",
    "borderRadius": "8px",
    "padding": "16px 20px",
    "marginBottom": "12px",
    "backgroundColor": "#fff",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
}

_TAG_STYLE = {
    "display": "inline-block",
    "backgroundColor": "#eef2ff",
    "color": "#4361ee",
    "borderRadius": "4px",
    "padding": "2px 8px",
    "fontSize": "0.78rem",
    "fontWeight": "600",
    "marginRight": "6px",
    "marginBottom": "4px",
}

_META_STYLE = {
    "fontSize": "0.8rem",
    "color": "#888",
    "marginBottom": "8px",
}

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "maxWidth": "860px",
        "margin": "40px auto",
        "fontFamily": "'Segoe UI', sans-serif",
        "padding": "0 20px",
        "backgroundColor": "#f8f9fc",
        "minHeight": "100vh",
    },
    children=[
        html.H1(
            "Recherche avancée Qdrant",
            style={"marginBottom": "6px", "fontSize": "1.6rem", "color": "#1a1a2e"},
        ),
        html.P(
            "Requête simple ou complexe (c: pre: sem:/keywords: req: sem:/keywords:)",
            style={"color": "#777", "fontSize": "0.85rem", "marginBottom": "16px"},
        ),
        dcc.Input(
            id="query-input",
            placeholder='ex: c: pre: sem: "plage" lim: 50 req: sem: "bronzage"',
            debounce=True,
            type="text",
            style={
                "width": "100%",
                "padding": "11px 14px",
                "fontSize": "0.95rem",
                "border": "1px solid #ccc",
                "borderRadius": "6px",
                "boxSizing": "border-box",
                "marginBottom": "20px",
                "backgroundColor": "#fff",
            },
        ),
        dcc.Loading(
            id="loading",
            type="circle",
            children=html.Div(id="results-container"),
        ),
    ],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tag_chips(tags_str: str) -> list[html.Span]:
    """Convert a tags string into a list of styled chip spans.

    Args:
        tags_str: Tag string like "#SPORT, #NATURE".

    Returns:
        List of html.Span elements, one per tag.
    """
    if not tags_str:
        return []
    return [
        html.Span(tag.strip(), style=_TAG_STYLE)
        for tag in tags_str.split(",")
        if tag.strip()
    ]


def _result_card(rank: int, result: SearchResult) -> html.Div:
    """Build a result card for a single SearchResult.

    Args:
        rank: 1-based rank position.
        result: The SearchResult to display.

    Returns:
        An html.Div card component.
    """
    snippet = result.paragraph_text[:400] + (
        "\u2026" if len(result.paragraph_text) > 400 else ""
    )

    return html.Div(
        style=_CARD_STYLE,
        children=[
            # Header row: rank + doc id + paragraph id
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "marginBottom": "6px",
                },
                children=[
                    html.Span(
                        f"#{rank}  \u2014  Document {result.document_id}",
                        style={
                            "fontWeight": "700",
                            "fontSize": "1rem",
                            "color": "#1a1a2e",
                        },
                    ),
                    html.Span(
                        f"Paragraphe {result.paragraph_id}",
                        style=_META_STYLE,
                    ),
                ],
            ),
            # Tags
            html.Div(
                _tag_chips(result.tags),
                style={"marginBottom": "10px"},
            ),
            # Paragraph snippet
            html.P(
                snippet,
                style={
                    "fontSize": "0.88rem",
                    "color": "#333",
                    "lineHeight": "1.55",
                    "margin": "0",
                    "whiteSpace": "pre-wrap",
                },
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


@app.callback(
    Output("results-container", "children"),
    Input("query-input", "value"),
)
def update_results(query: str | None) -> html.Div:
    """Update the results container based on the query input.

    Args:
        query: The raw query string entered by the user.

    Returns:
        A Dash html.Div containing the search results or an error message.
    """
    if not query or not query.strip():
        return html.Div()

    try:
        results = executor.execute(query.strip())
    except Exception as exc:  # noqa: BLE001
        return html.Div(
            f"Erreur : {exc}",
            style={
                "color": "#c0392b",
                "padding": "12px",
                "fontWeight": "600",
                "backgroundColor": "#fdecea",
                "borderRadius": "6px",
            },
        )

    if not results:
        return html.Div(
            "Aucun résultat.",
            style={"color": "#666", "fontStyle": "italic", "padding": "10px"},
        )

    count_label = html.P(
        f"{len(results)} r\u00e9sultat{'s' if len(results) > 1 else ''}",
        style={
            "fontWeight": "700",
            "marginBottom": "14px",
            "color": "#444",
            "fontSize": "0.95rem",
        },
    )

    cards = [_result_card(i + 1, r) for i, r in enumerate(results)]

    return html.Div([count_label, *cards])
