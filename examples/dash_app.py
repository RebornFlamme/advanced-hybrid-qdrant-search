"""Dash application for advanced Qdrant search."""

from __future__ import annotations

from pathlib import Path

import dash
from dash import Input, Output, dcc, html
from qdrant_client.models import FieldCondition, Filter, MatchValue

from qdrant_advanced_search import QueryExecutor

# ---------------------------------------------------------------------------
# Module-level executor (model loads once at import time)
# ---------------------------------------------------------------------------

executor = QueryExecutor(parquet_path=Path(__file__).parent.parent / "documents.parquet")

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


def _fetch_payloads(doc_ids: list[int]) -> dict[int, dict]:
    """Fetch one paragraph payload per document ID from Qdrant.

    Args:
        doc_ids: List of document IDs to fetch.

    Returns:
        Mapping of document_id to its first matching payload dict.
    """
    if not doc_ids:
        return {}
    records, _ = executor.client.scroll(
        collection_name=executor.collection_name,
        scroll_filter=Filter(
            should=[
                FieldCondition(key="document_id", match=MatchValue(value=doc_id))
                for doc_id in doc_ids
            ]
        ),
        limit=len(doc_ids) * 10,
        with_payload=True,
    )
    payloads: dict[int, dict] = {}
    for r in records:
        if r.payload:
            did = int(r.payload.get("document_id", -1))
            if did in doc_ids and did not in payloads:
                payloads[did] = r.payload
    return payloads


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


def _result_card(rank: int, doc_id: int, payload: dict) -> html.Div:
    """Build a result card for a single document.

    Args:
        rank: 1-based rank position.
        doc_id: The document ID.
        payload: Qdrant payload dict for one paragraph of this document.

    Returns:
        An html.Div card component.
    """
    paragraph_text = payload.get("text", "")
    snippet = paragraph_text[:400] + ("…" if len(paragraph_text) > 400 else "")
    tags_str = payload.get("tags", "")
    paragraph_id = payload.get("paragraph_id", "")

    return html.Div(
        style=_CARD_STYLE,
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "marginBottom": "6px",
                },
                children=[
                    html.Span(
                        f"#{rank}  —  Document {doc_id}",
                        style={"fontWeight": "700", "fontSize": "1rem", "color": "#1a1a2e"},
                    ),
                    html.Span(f"Paragraphe {paragraph_id}", style=_META_STYLE),
                ],
            ),
            html.Div(_tag_chips(tags_str), style={"marginBottom": "10px"}),
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
        doc_ids = executor.execute(query.strip())
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

    if not doc_ids:
        return html.Div(
            "Aucun résultat.",
            style={"color": "#666", "fontStyle": "italic", "padding": "10px"},
        )

    payloads = _fetch_payloads(doc_ids)

    count_label = html.P(
        f"{len(doc_ids)} résultat{'s' if len(doc_ids) > 1 else ''}",
        style={"fontWeight": "700", "marginBottom": "14px", "color": "#444", "fontSize": "0.95rem"},
    )

    cards = [
        _result_card(i + 1, doc_id, payloads.get(doc_id, {}))
        for i, doc_id in enumerate(doc_ids)
    ]

    return html.Div([count_label, *cards])


if __name__ == "__main__":
    app.run(debug=True)
