"""Views package."""

from .about_view import render_about_view
from .analyze_view import render_analyze_view
from .ddi_checker_view import render_ddi_checker_view
from .drug_search_view import render_drug_search_view

__all__ = [
    "render_about_view",
    "render_analyze_view",
    "render_ddi_checker_view",
    "render_drug_search_view",
]
