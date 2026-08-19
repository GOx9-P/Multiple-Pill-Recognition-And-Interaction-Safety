"""UI configuration flags and design tokens for the Streamlit application."""

from __future__ import annotations

# When True, the detect/interaction buttons populate labeled demo data instead of
# calling real inference. Set to False once model artifacts are ready.
UI_DEMO_MODE = True

# Design tokens
COLOR_BACKGROUND = "#F4F8FC"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#0F172A"
COLOR_TEXT_MUTED = "#64748B"
COLOR_SUCCESS = "#16A34A"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#DC2626"
COLOR_INFO = "#0EA5E9"

# Section accents
COLOR_SECTION_1 = "#3B82F6"
COLOR_SECTION_2 = "#8B5CF6"

# Header
COLOR_HEADER_BG = "#031D56"
COLOR_HEADER_TEXT = "#FFFFFF"
COLOR_HEADER_SUBTITLE = "#8EA8C3"

# Legacy aliases kept for compatibility with existing modules
COLOR_PRIMARY = COLOR_SECTION_1
COLOR_PRIMARY_DARK = "#1E3A8A"
COLOR_SECONDARY = "#0F766E"
