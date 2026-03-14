"""Dash application for advanced Qdrant search."""

from __future__ import annotations

import dash
from dash import Input, Output, dcc, html

from query_executor import QueryExecutor

# ---------------------------------------------------------------------------
# Module-level executor (model loads once at import time)
# ---------------------------------------------------------------------------

executor = QueryExecutor()

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div(
    style={"maxWidth": "900px", "margin": "40px auto", "fontFamily": "sans-serif", "padding": "0 20px"},
    children=[
        html.H1(
            "Recherche avancée Qdrant",
            style={"marginBottom": "24px", "fontSize": "1.8rem", "color": "#1a1a2e"},
        ),
        dcc.Input(
            id="query-input",
            placeholder="Entrez votre requête… ex: c: pre: sem: \"plage\" lim: 50 req: sem: \"bronzage\"",
            debounce=True,
            type="text",
            style={
                "width": "100%",
                "padding": "10px 14px",
                "fontSize": "1rem",
                "border": "1px solid #ccc",
                "borderRadius": "6px",
                "boxSizing": "border-box",
                "marginBottom": "20px",
            },
        ),
        dcc.Loading(
            id="loading",
            type="default",
            children=html.Div(id="results-container"),
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
            style={"color": "red", "padding": "10px", "fontWeight": "bold"},
        )

    if not doc_ids:
        return html.Div(
            "Aucun résultat",
            style={"color": "#666", "fontStyle": "italic", "padding": "10px"},
        )

    count_label = html.P(
        f"{len(doc_ids)} résultat{'s' if len(doc_ids) > 1 else ''}",
        style={"fontWeight": "bold", "marginBottom": "12px", "color": "#333"},
    )

    items = [
        html.Li(
            f"Document ID : {doc_id}",
            style={
                "padding": "8px 12px",
                "borderBottom": "1px solid #eee",
                "fontFamily": "monospace",
                "fontSize": "0.95rem",
            },
        )
        for doc_id in doc_ids
    ]

    return html.Div(
        [
            count_label,
            html.Ul(
                items,
                style={
                    "listStyle": "none",
                    "margin": "0",
                    "padding": "0",
                    "border": "1px solid #ddd",
                    "borderRadius": "6px",
                    "overflow": "hidden",
                },
            ),
        ]
    )
