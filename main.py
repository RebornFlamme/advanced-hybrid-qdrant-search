"""Entry point for the Qdrant advanced search Dash application."""

from __future__ import annotations

from examples.dash_app import app


def main() -> None:
    """Run the Dash development server."""
    app.run(debug=True)


if __name__ == "__main__":
    main()
