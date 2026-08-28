"""Healthcare & Clinical Decision Support CSS System for Streamlit.

Implements the Unified Multi-Platform Clinical Workspace:
- Desktop Clinical Split-View
- iPhone 17 Pro Max Interactive Simulator (440x956 px reference screen viewport)
- Mobile-Native Responsive Web Experience (Seamless on real smartphones)
- White-dominant surface architecture for Light Mode (#FFFFFF / #F8FAFC / #0F172A)
- High contrast, WCAG 2.1 AAA accessible typography & touch targets (Inter, Outfit, JetBrains Mono)
- Unambiguous clinical severity signaling (Critical: Red, Moderate: Amber, Duplicate: Purple, Safe: Emerald)
"""

from __future__ import annotations

import streamlit as st


def inject_healthcare_css() -> None:
    """Inject centralized clinical design tokens and responsive CSS into Streamlit."""
    st.markdown(
        """
        <style>
        /* Import Professional Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Root Tokens — Strictly White-First Clinical Medical Palette */
        :root {
            --bg-canvas: #F8FAFC;
            --bg-surface: #FFFFFF;
            --bg-surface-elevated: #F1F5F9;
            --bg-input: #FFFFFF;
            
            --border-subtle: #E2E8F0;
            --border-medium: #CBD5E1;
            --border-active: #0891B2;
            
            --text-primary: #0F172A;
            --text-secondary: #1E293B;
            --text-muted: #64748B;
            
            --accent-brand: #0891B2;
            --accent-brand-hover: #0E7490;
            --accent-brand-light: rgba(8, 145, 178, 0.08);
            
            --sev-critical: #DC2626;
            --sev-critical-bg: #FEF2F2;
            --sev-critical-border: #F87171;
            --sev-critical-text: #7F1D1D;
            
            --sev-moderate: #D97706;
            --sev-moderate-bg: #FFFBEB;
            --sev-moderate-border: #FCD34D;
            --sev-moderate-text: #78350F;
            
            --sev-duplicate: #9333EA;
            --sev-duplicate-bg: #FAF5FF;
            --sev-duplicate-border: #D8B4FE;
            --sev-duplicate-text: #581C87;
            
            --sev-safe: #059669;
            --sev-safe-bg: #ECFDF5;
            --sev-safe-border: #6EE7B7;
            --sev-safe-text: #064E3B;
            
            --sev-unresolved: #4F46E5;
            --sev-unresolved-bg: #EEF2FF;
            --sev-unresolved-border: #A5B4FC;
            --sev-unresolved-text: #312E81;
            
            --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.06), 0 1px 2px 0 rgba(0, 0, 0, 0.04);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
            --shadow-phone: 0 20px 50px -10px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(0, 0, 0, 0.06);
            
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-full: 9999px;
            --radius-phone: 48px;
        }

        /* Streamlit Global Base & Outer Containers */
        html, body, #root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"], [data-testid="stSidebar"], .block-container {
            background-color: var(--bg-canvas) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: var(--text-primary) !important;
        }

        /* Header Bar Clearance */
        header[data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.85) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border-bottom: 1px solid var(--border-subtle) !important;
            z-index: 999 !important;
        }

        /* Main Container Layout */
        .block-container {
            padding-top: 4.5rem !important; /* Clearance below fixed Streamlit header */
            padding-bottom: 3rem !important;
            max-width: 1360px !important;
            background-color: var(--bg-canvas) !important;
        }

        /* Typography Scale */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: 0 !important;
        }

        p, span, label, li {
            color: var(--text-secondary);
        }

        /* Streamlit Markdown Text & List Styling */
        .stMarkdown p {
            color: var(--text-secondary) !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
        }

        .stMarkdown li {
            color: var(--text-secondary) !important;
            font-size: 0.9rem !important;
            line-height: 1.6 !important;
        }

        .stMarkdown strong {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        /* High-Contrast Code Elements */
        code, .stMarkdown code {
            font-family: 'JetBrains Mono', monospace !important;
            background-color: var(--bg-surface-elevated) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-size: 0.85em !important;
            font-weight: 600 !important;
        }

        /* Labels and Text Contrast */
        .stSelectbox label, .stTextInput label, .stTextArea label, .stRadio label {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            margin-bottom: 6px !important;
        }

        /* -------------------------------------------------------------
           VIEW SWITCHER TOOLBAR STAGE (Fixed Outside Phone)
           ------------------------------------------------------------- */
        .preview-toolbar-stage {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            width: 100%;
        }

        .preview-toolbar-stage [data-testid="stRadio"] {
            margin: 0 auto !important;
        }

        .preview-toolbar-stage [data-testid="stRadio"] > div {
            flex-direction: row;
            background: var(--bg-surface);
            border: 1px solid var(--border-medium);
            border-radius: var(--radius-full);
            padding: 4px 6px;
            box-shadow: var(--shadow-md);
            gap: 6px;
        }

        .preview-toolbar-stage [data-testid="stRadio"] label {
            padding: 6px 14px !important;
            border-radius: var(--radius-full) !important;
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            margin-bottom: 0 !important;
        }

        /* -------------------------------------------------------------
           IPHONE 17 PRO MAX HARDWARE CHASSIS & SCREEN VIEWPORT
           ------------------------------------------------------------- */
        .st-key-iphone_17_simulator {
            max-width: 440px !important; /* Reference logical width */
            margin: 0 auto 3rem auto !important;
            background: #FFFFFF !important; /* Strictly White Light Mode */
            border: 10px solid #E2E8F0 !important; /* Sleek Metallic Light Titanium Bezel */
            border-radius: var(--radius-phone) !important;
            box-shadow: var(--shadow-phone) !important;
            padding: 12px 14px 20px 14px !important;
            box-sizing: border-box !important;
            position: relative !important;
            aspect-ratio: 440 / 956 !important;
            overflow: hidden !important; /* Containment: Lock all elements inside chassis */
        }

        /* The content viewport is the phone's only vertical scroll owner. */
        .st-key-mobile_scroll_viewport {
            height: 800px !important;
            max-height: 800px !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-right: 0 !important;
            padding-bottom: 84px !important;
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
            overscroll-behavior-y: contain !important;
            touch-action: pan-y !important;
            -webkit-overflow-scrolling: touch;
        }

        .st-key-mobile_scroll_viewport::-webkit-scrollbar {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }

        .st-key-mobile_scroll_viewport [data-testid="stVerticalBlock"] {
            max-height: none !important;
            overflow: visible !important;
        }

        /* Internal Scroll Viewport for Mobile App Content */
        .mobile-app-viewport {
            width: 100% !important;
            box-sizing: border-box !important;
        }

        /* Dynamic Island */
        .iphone-island-container {
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2px 0 10px 0;
            user-select: none;
        }

        .iphone-island {
            width: 110px;
            height: 24px;
            background: #000000;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        .island-camera {
            width: 8px;
            height: 8px;
            background: #1E293B;
            border-radius: 50%;
            border: 1px solid #334155;
        }

        .island-sensor {
            width: 6px;
            height: 6px;
            background: #0F172A;
            border-radius: 50%;
        }

        /* Home Indicator - Anchored at the absolute lowest bottom edge of the phone chassis ("dưới đít điện thoại") */
        .st-key-iphone_17_simulator [data-testid="stElementContainer"]:has(.iphone-home-indicator-container),
        .iphone-home-indicator-container {
            position: absolute !important;
            bottom: 8px !important;
            left: 0 !important;
            right: 0 !important;
            width: 100% !important;
            margin: 0 auto !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            padding: 0 !important;
            z-index: 99999 !important;
            pointer-events: none !important;
            user-select: none !important;
        }

        .iphone-home-indicator {
            width: 130px !important;
            height: 4px !important;
            background: #0F172A !important;
            border-radius: 2px !important;
            opacity: 0.3 !important;
            margin: 0 auto !important;
        }

        /* Mobile controls use a comfortable iOS-sized touch target. */
        .st-key-mobile_scroll_viewport div.stButton > button,
        .st-key-iphone_17_simulator div.stButton > button {
            white-space: nowrap !important;
            font-size: 0.78rem !important;
            min-height: 44px !important;
            padding: 0.45rem 0.5rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            line-height: 1 !important;
        }

        /* -------------------------------------------------------------
           MOBILE UI COMPONENTS & PATTERNS
           ------------------------------------------------------------- */
        .mobile-header-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0.75rem;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 0.75rem;
        }

        .mobile-header-brand {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .mobile-brand-icon {
            font-size: 1.2rem;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #0891B2, #06B6D4);
            border-radius: var(--radius-sm);
            color: white;
        }

        .mobile-brand-name {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 1rem;
            line-height: 1.15;
            color: var(--text-primary);
        }

        .mobile-brand-sub {
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        .mobile-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 3px 8px;
            border-radius: var(--radius-full);
            font-size: 0.725rem;
            font-weight: 600;
            background: var(--sev-safe-bg);
            border: 1px solid var(--sev-safe-border);
            color: var(--sev-safe-text);
        }

        /* Mobile tab bar stays outside the scrolling content. */
        .st-key-mobile_bottom_nav {
            position: absolute !important;
            left: 14px !important;
            right: 14px !important;
            bottom: 18px !important;
            z-index: 900 !important;
            padding: 7px !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 18px !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 -4px 18px rgba(15, 23, 42, 0.08) !important;
            backdrop-filter: blur(12px) !important;
        }

        .st-key-mobile_bottom_nav [data-testid="stRadio"] > div {
            flex-direction: row !important;
            background: var(--bg-surface-elevated) !important;
            border-radius: var(--radius-md) !important;
            padding: 3px !important;
            gap: 4px !important;
        }

        .st-key-mobile_bottom_nav [data-testid="stRadio"] label {
            flex: 1 1 0 !important;
            justify-content: center !important;
            min-height: 44px !important;
            padding: 8px 6px !important;
            border-radius: var(--radius-sm) !important;
            font-size: 0.76rem !important;
            font-weight: 600 !important;
            color: var(--text-secondary) !important;
            margin: 0 !important;
        }

        .st-key-mobile_bottom_nav label:has(input:checked) {
            background: var(--bg-surface) !important;
            color: var(--accent-brand) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        .st-key-mobile_bottom_nav label:focus-within {
            outline: 2px solid var(--accent-brand) !important;
            outline-offset: 2px !important;
        }

        /* Hide radio circles in mobile nav for clean segmented tabs */
        .st-key-mobile_bottom_nav [data-testid="stRadio"] input[type="radio"],
        .st-key-mobile_bottom_nav [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }

        /* Media Entry Hub Card (Zalo-Inspired Pattern) */
        .mobile-media-entry-hub {
            background: var(--bg-surface);
            border: 2px dashed var(--border-medium);
            border-radius: var(--radius-lg);
            padding: 1.75rem 1rem;
            text-align: center;
            margin-bottom: 0.85rem;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .mobile-media-entry-hub:hover {
            border-color: var(--accent-brand);
            background: var(--accent-brand-light);
        }

        .mobile-media-entry-icon {
            font-size: 2.25rem;
            margin-bottom: 0.4rem;
            line-height: 1;
        }

        .mobile-media-entry-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }

        .mobile-media-entry-sub {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        /* Media Chooser Grid (Zalo Pattern) */
        .media-chooser-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-bottom: 1rem;
        }

        .media-chooser-cell {
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 12px 6px;
            text-align: center;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .media-chooser-cell:hover {
            border-color: var(--accent-brand);
            background: var(--accent-brand-light);
        }

        .media-chooser-cell-icon {
            font-size: 1.5rem;
            margin-bottom: 4px;
        }

        .media-chooser-cell-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Mobile Safety Hero Banner */
        .mobile-safety-hero {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.85rem 1rem;
            border-radius: var(--radius-md);
            margin-bottom: 0.85rem;
            border: 1px solid;
            box-shadow: var(--shadow-sm);
        }

        .mobile-safety-hero.critical {
            background: var(--sev-critical-bg);
            border-color: var(--sev-critical-border);
            color: var(--sev-critical-text);
        }

        .mobile-safety-hero.moderate {
            background: var(--sev-moderate-bg);
            border-color: var(--sev-moderate-border);
            color: var(--sev-moderate-text);
        }

        .mobile-safety-hero.unresolved {
            background: var(--sev-unresolved-bg);
            border-color: var(--sev-unresolved-border);
            color: var(--sev-unresolved-text);
        }

        .mobile-safety-hero.safe {
            background: var(--sev-safe-bg);
            border-color: var(--sev-safe-border);
            color: var(--sev-safe-text);
        }

        .mobile-safety-icon {
            font-size: 1.5rem;
            line-height: 1;
        }

        .mobile-safety-title {
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 0.2rem;
            color: inherit;
        }

        .mobile-safety-desc {
            font-size: 0.8rem;
            line-height: 1.4;
            color: inherit;
            opacity: 0.95;
        }

        /* Mobile Pill Row (Touchable) */
        .mobile-pill-row {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.5rem;
            box-shadow: var(--shadow-sm);
            transition: all 0.15s ease;
        }

        .mobile-pill-row.active {
            border: 2px solid var(--accent-brand);
            box-shadow: 0 0 0 3px var(--accent-brand-light);
        }

        .mobile-pill-row-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.25rem;
        }

        .mobile-pill-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.725rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .mobile-pill-name {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 0.95rem;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
            word-wrap: break-word;
        }

        .mobile-pill-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.775rem;
            color: var(--text-muted);
        }

        .mobile-pill-conf {
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Mobile DDI Card */
        .mobile-ddi-card {
            background: var(--bg-surface);
            border-left: 4px solid var(--sev-critical);
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            padding: 0.85rem;
            margin-bottom: 0.65rem;
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            box-shadow: var(--shadow-sm);
        }

        .mobile-ddi-card.moderate {
            border-left-color: var(--sev-moderate);
        }

        .mobile-ddi-header {
            margin-bottom: 0.35rem;
        }

        .mobile-ddi-badge {
            display: inline-block;
            padding: 2px 7px;
            border-radius: var(--radius-full);
            font-size: 0.68rem;
            font-weight: 700;
        }

        .mobile-ddi-badge.critical {
            background: var(--sev-critical-bg);
            color: var(--sev-critical-text);
            border: 1px solid var(--sev-critical-border);
        }

        .mobile-ddi-badge.moderate {
            background: var(--sev-moderate-bg);
            color: var(--sev-moderate-text);
            border: 1px solid var(--sev-moderate-border);
        }

        .mobile-ddi-pair {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 0.95rem;
            color: var(--text-primary);
            margin-bottom: 0.4rem;
            word-wrap: break-word;
        }

        .mobile-ddi-section {
            font-size: 0.8rem;
            line-height: 1.45;
            margin-bottom: 0.35rem;
        }

        .mobile-ddi-label {
            font-size: 0.725rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            margin-bottom: 2px;
            color: var(--text-muted);
        }

        .mobile-ddi-text {
            color: var(--text-secondary);
        }

        .mobile-guidance-box {
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.75rem 0.85rem;
            margin-top: 0.75rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        .mobile-safe-box {
            background: var(--sev-safe-bg);
            border: 1px solid var(--sev-safe-border);
            color: var(--sev-safe-text);
            padding: 0.75rem 0.85rem;
            border-radius: var(--radius-md);
            font-size: 0.825rem;
            line-height: 1.45;
            margin-bottom: 0.75rem;
        }

        .mobile-report-text {
            font-size: 0.825rem;
            line-height: 1.6;
            color: var(--text-secondary);
        }

        /* Consumer mobile flow */
        .mobile-app-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 2px 2px 12px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 18px;
        }

        .mobile-app-kicker { color: var(--accent-brand); font-size: 0.64rem; font-weight: 800; letter-spacing: 0.12em; margin-bottom: 2px; }
        .mobile-app-bar h1 { font-size: 1.28rem !important; line-height: 1.15 !important; margin: 0 !important; }
        .mobile-app-mark { width: 34px; height: 34px; border-radius: 11px; display: grid; place-items: center; background: var(--accent-brand); color: #FFFFFF; font-size: 1.35rem; font-weight: 500; }

        .mobile-intro { padding: 12px 2px 18px; }
        .mobile-intro.compact { padding-bottom: 10px; }
        .mobile-intro h2 { font-size: 1.42rem !important; line-height: 1.18 !important; margin: 0 0 8px !important; max-width: 320px; }
        .mobile-intro p { color: var(--text-muted) !important; font-size: 0.88rem !important; line-height: 1.5 !important; margin: 0 !important; }

        .mobile-photo-guide,
        .mobile-image-warning,
        .mobile-medical-disclaimer,
        .mobile-empty-state {
            display: flex;
            flex-direction: column;
            gap: 5px;
            border-radius: var(--radius-md);
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .mobile-photo-guide { margin: 14px 0 10px; padding: 12px 14px; background: var(--bg-surface-elevated); color: var(--text-secondary); }
        .mobile-image-warning { margin-bottom: 12px; padding: 12px 14px; background: var(--sev-unresolved-bg); border: 1px solid var(--sev-unresolved-border); color: var(--sev-unresolved-text); }
        .mobile-empty-state { margin: 22px 0 14px; padding: 24px 18px; text-align: center; background: var(--bg-surface); border: 1px dashed var(--border-medium); color: var(--text-secondary); }
        .mobile-empty-state strong { font-size: 1rem; color: var(--text-primary); }
        .mobile-empty-state.error { border-style: solid; border-color: var(--sev-critical-border); }

        .mobile-safety-hero { padding: 16px !important; gap: 12px !important; margin-bottom: 22px !important; box-shadow: none !important; }
        .mobile-safety-symbol { flex: 0 0 32px; width: 32px; height: 32px; display: grid; place-items: center; border: 1.5px solid currentColor; border-radius: 50%; font-weight: 800; line-height: 1; }
        .mobile-safety-content { min-width: 0; }
        .mobile-safety-eyebrow { font-size: 0.67rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 4px; }
        .mobile-safety-content h2 { color: inherit !important; font-size: 1.08rem !important; line-height: 1.22 !important; margin: 0 0 6px !important; }
        .mobile-safety-content p { color: inherit !important; font-size: 0.82rem !important; line-height: 1.45 !important; margin: 0 0 10px !important; }
        .mobile-safety-action { display: flex; flex-direction: column; gap: 3px; padding: 10px 11px; border-radius: var(--radius-sm); background: rgba(255, 255, 255, 0.72); color: var(--text-primary); font-size: 0.8rem; line-height: 1.42; }
        .mobile-safety-action strong,
        .mobile-medical-disclaimer strong { color: inherit !important; }
        .mobile-completeness-note { margin-top: 8px; font-size: 0.75rem; font-weight: 700; }

        .mobile-section-heading { display: flex; align-items: center; justify-content: space-between; margin: 0 2px 10px; }
        .mobile-section-heading.simple { margin-top: 24px; }
        .mobile-section-heading span { display: block; color: var(--text-muted); font-size: 0.64rem; font-weight: 800; letter-spacing: 0.08em; margin-bottom: 2px; }
        .mobile-section-heading h2 { font-size: 1.02rem !important; margin: 0 !important; }
        .mobile-progress-count { min-width: 42px; padding: 6px 8px; border-radius: var(--radius-full); background: var(--accent-brand-light); color: var(--accent-brand); text-align: center; font-size: 0.78rem; font-weight: 800; }

        .mobile-pill-row { padding: 13px 14px !important; margin-bottom: 8px !important; box-shadow: none !important; }
        .mobile-pill-row.unresolved,
        .mobile-pill-row.ambiguous { border-color: var(--sev-unresolved-border); }
        .mobile-pill-index { color: var(--text-muted); font-size: 0.72rem; font-weight: 700; }
        .mobile-pill-confidence { color: var(--text-secondary); font-size: 0.78rem; font-weight: 600; margin-bottom: 3px; }
        .mobile-pill-evidence { color: var(--text-muted); font-size: 0.72rem; line-height: 1.35; }
        .pill-badge.manual { background: var(--sev-unresolved-bg); color: var(--sev-unresolved-text); border: 1px solid var(--sev-unresolved-border); }

        .st-key-mobile_scroll_viewport [data-testid="stForm"] { margin: -2px 0 10px; padding: 12px !important; border: 1px solid var(--sev-unresolved-border) !important; border-radius: var(--radius-md) !important; background: var(--sev-unresolved-bg) !important; }

        .mobile-ddi-card,
        .mobile-duplicate-card { padding: 14px !important; margin-bottom: 10px !important; box-shadow: none !important; }
        .mobile-duplicate-card { background: var(--sev-duplicate-bg); border: 1px solid var(--sev-duplicate-border); border-left: 4px solid var(--sev-duplicate); border-radius: 0 var(--radius-md) var(--radius-md) 0; }
        .mobile-ddi-badge.duplicate { background: #FFFFFF; color: var(--sev-duplicate-text); border: 1px solid var(--sev-duplicate-border); margin-bottom: 6px; }
        .mobile-ddi-action { display: flex; flex-direction: column; gap: 3px; padding: 10px 11px; border-radius: var(--radius-sm); background: var(--bg-surface-elevated); color: var(--text-primary); font-size: 0.8rem; line-height: 1.42; margin-bottom: 10px; }
        .mobile-ddi-action strong { color: var(--accent-brand) !important; }

        .mobile-medical-disclaimer { margin: 18px 0 10px; padding: 13px 14px; background: #F8FAFC; border: 1px solid var(--border-subtle); color: var(--text-muted); }

        /* -------------------------------------------------------------
           DESKTOP WORKSPACE COMPONENTS
           ------------------------------------------------------------- */
        .clinical-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.5rem;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            margin-bottom: 1.25rem;
            box-shadow: var(--shadow-sm);
        }

        .clinical-brand {
            display: flex;
            align-items: center;
            gap: 0.875rem;
        }

        .clinical-brand-icon {
            width: 42px;
            height: 42px;
            border-radius: var(--radius-md);
            background: linear-gradient(135deg, #0891B2, #06B6D4);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            color: #FFFFFF;
            box-shadow: 0 2px 8px rgba(8, 145, 178, 0.25);
        }

        .clinical-brand-title {
            font-size: 1.35rem !important;
            margin: 0 !important;
            line-height: 1.2 !important;
            color: var(--text-primary) !important;
        }

        .clinical-brand-subtitle {
            font-size: 0.825rem;
            color: var(--text-muted) !important;
            margin: 0;
        }

        .clinical-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: var(--shadow-sm);
        }

        .severity-banner {
            border-radius: var(--radius-md);
            padding: 1rem 1.25rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            border: 1px solid;
            box-shadow: var(--shadow-sm);
        }

        .severity-banner.critical {
            background: var(--sev-critical-bg);
            border-color: var(--sev-critical-border);
            color: var(--sev-critical-text);
        }

        .severity-banner.moderate {
            background: var(--sev-moderate-bg);
            border-color: var(--sev-moderate-border);
            color: var(--sev-moderate-text);
        }

        .severity-banner.safe {
            background: var(--sev-safe-bg);
            border-color: var(--sev-safe-border);
            color: var(--sev-safe-text);
        }

        .severity-banner.unresolved {
            background: var(--sev-unresolved-bg);
            border-color: var(--sev-unresolved-border);
            color: var(--sev-unresolved-text);
        }

        .pill-card-item {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1rem 1.15rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-sm);
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        .pill-card-item.active {
            border: 2px solid var(--accent-brand);
            box-shadow: 0 0 0 3px var(--accent-brand-light);
        }

        /* Compact desktop medication cards prevent long medication names from creating a long page. */
        .pill-card-top {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.45rem;
        }

        .pill-instance-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .pill-drug-name {
            margin: 0.15rem 0 0.7rem !important;
            font-size: 1.08rem !important;
            line-height: 1.3 !important;
            overflow-wrap: anywhere;
        }

        .pill-meta-row {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            padding: 0.32rem 0;
            border-top: 1px solid var(--border-subtle);
            font-size: 0.82rem;
            line-height: 1.35;
        }

        .pill-meta-row span:first-child {
            color: var(--text-muted);
            flex: 0 0 auto;
        }

        .pill-meta-row span:last-child {
            color: var(--text-primary);
            font-weight: 600;
            min-width: 0;
            overflow-wrap: anywhere;
            text-align: right;
        }

        .pill-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: var(--radius-full);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .pill-badge.accepted { background: var(--sev-safe-bg); color: var(--sev-safe-text); border: 1px solid var(--sev-safe-border); }
        .pill-badge.ambiguous { background: var(--sev-moderate-bg); color: var(--sev-moderate-text); border: 1px solid var(--sev-moderate-border); }
        .pill-badge.unresolved { background: var(--sev-critical-bg); color: var(--sev-critical-text); border: 1px solid var(--sev-critical-border); }

        .ddi-card {
            background: var(--bg-surface);
            border-left: 4px solid var(--sev-critical);
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            box-shadow: var(--shadow-sm);
        }

        .ddi-card.moderate {
            border-left-color: var(--sev-moderate);
        }

        .clinical-disclaimer {
            background: var(--bg-surface-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.875rem 1.15rem;
            margin-top: 1.5rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.45;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }

        /* Streamlit Buttons & Touch Ergonomics */
        div.stButton > button {
            border-radius: var(--radius-md) !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.15s ease !important;
            border: 1px solid var(--border-medium) !important;
            background-color: var(--bg-surface) !important;
            color: var(--text-primary) !important;
            min-height: 42px !important;
        }

        div.stButton > button:hover {
            border-color: var(--accent-brand) !important;
            color: var(--accent-brand) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        div.stButton > button[kind="primary"] {
            background: var(--accent-brand) !important;
            color: #FFFFFF !important;
            border: none !important;
            min-height: 46px !important;
            font-size: 0.95rem !important;
            box-shadow: 0 4px 12px rgba(8, 145, 178, 0.25) !important;
        }

        div.stButton > button[kind="primary"]:hover {
            background: var(--accent-brand-hover) !important;
            box-shadow: 0 6px 16px rgba(8, 145, 178, 0.35) !important;
            transform: translateY(-1px);
        }

        /* -------------------------------------------------------------
           REAL SMARTPHONE RESPONSIVE OVERRIDES (max-width: 600px)
           ------------------------------------------------------------- */
        @media (max-width: 600px) {
            .preview-toolbar-stage {
                display: none !important;
            }
            .st-key-iphone_17_simulator {
                max-width: 100% !important;
                margin: 0 !important;
                border: none !important;
                border-radius: 0 !important;
                box-shadow: none !important;
                background: #FFFFFF !important;
                padding: 0 !important;
                aspect-ratio: auto !important;
            }
            .iphone-island-container {
                display: none !important;
            }
            .iphone-home-indicator-container {
                display: none !important;
            }
            .st-key-mobile_scroll_viewport {
                height: auto !important;
                max-height: none !important;
                overflow: visible !important;
                padding-bottom: 96px !important;
            }
            .st-key-mobile_bottom_nav {
                position: fixed !important;
                left: 8px !important;
                right: 8px !important;
                bottom: max(8px, env(safe-area-inset-bottom)) !important;
            }
            html, body, .stApp, [data-testid="stAppViewContainer"] {
                scrollbar-width: none !important;
                -ms-overflow-style: none !important;
            }
            html::-webkit-scrollbar,
            body::-webkit-scrollbar,
            .stApp::-webkit-scrollbar,
            [data-testid="stAppViewContainer"]::-webkit-scrollbar {
                display: none !important;
                width: 0 !important;
            }
            .block-container {
                padding-top: 3.5rem !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
