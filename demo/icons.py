"""
IASTAM Edge Demo — Professional Aerospace Vector Icons & Badges
================================================================
Uses Font Awesome 6 (industry standard open-source library)
and semantic status badges matching the IASTAM 6.0 aerospace identity.
"""

def fa_icon(fa_class: str, color: str = "#38bdf8", size: str = "1.05em", extra_style: str = "") -> str:
    """Generates a crisp Font Awesome vector icon tag."""
    return f'<i class="{fa_class}" style="color:{color}; font-size:{size}; vertical-align:-1px; margin-right:8px; {extra_style}"></i>'


# ── Primary Aerospace UI Icons (Font Awesome 6 Solid) ──
ICON_SLIDERS = fa_icon("fa-solid fa-sliders", color="#10b981")
ICON_SATELLITE = fa_icon("fa-solid fa-satellite", color="#38bdf8")
ICON_ACTIVITY = fa_icon("fa-solid fa-wave-square", color="#10b981")
ICON_PACKAGE = fa_icon("fa-solid fa-box-archive", color="#38bdf8")
ICON_TARGET = fa_icon("fa-solid fa-crosshairs", color="#38bdf8")
ICON_TERMINAL = fa_icon("fa-solid fa-terminal", color="#38bdf8")
ICON_GAUGE = fa_icon("fa-solid fa-gauge-high", color="#f59e0b")
ICON_FLAME = fa_icon("fa-solid fa-fire", color="#f97316")
ICON_SHIELD_CHECK = fa_icon("fa-solid fa-shield-halved", color="#10b981")
ICON_SHIELD_ALERT = fa_icon("fa-solid fa-triangle-exclamation", color="#ef4444")
ICON_INFO = fa_icon("fa-solid fa-circle-info", color="#38bdf8")
ICON_MICROCHIP = fa_icon("fa-solid fa-microchip", color="#10b981")

# Backward-compatible aliases for existing callers
SVG_SLIDERS = ICON_SLIDERS
SVG_ACTIVITY = ICON_ACTIVITY
SVG_PACKAGE = ICON_PACKAGE
SVG_TARGET = ICON_TARGET
SVG_TERMINAL = ICON_TERMINAL
SVG_GAUGE = ICON_GAUGE
SVG_FLAME = ICON_FLAME
SVG_SHIELD_CHECK = ICON_SHIELD_CHECK
SVG_SHIELD_ALERT = ICON_SHIELD_ALERT
SVG_INFO = ICON_INFO
SVG_MICROCHIP = ICON_MICROCHIP


def badge(text: str, variant: str = "nominal") -> str:
    """Generates an aerospace telemetry status pill with a crisp Font Awesome icon."""
    icons = {
        "optimal": '<i class="fa-solid fa-circle-check" style="margin-right:5px;"></i>',
        "passed": '<i class="fa-solid fa-circle-check" style="margin-right:5px;"></i>',
        "detected": '<i class="fa-solid fa-circle-check" style="margin-right:5px;"></i>',
        "margin": '<i class="fa-solid fa-triangle-exclamation" style="margin-right:5px;"></i>',
        "partial": '<i class="fa-solid fa-triangle-exclamation" style="margin-right:5px;"></i>',
        "alert": '<i class="fa-solid fa-circle-xmark" style="margin-right:5px;"></i>',
        "error": '<i class="fa-solid fa-circle-xmark" style="margin-right:5px;"></i>',
        "nominal": '<i class="fa-solid fa-satellite" style="margin-right:5px;"></i>',
        "scene": '<i class="fa-solid fa-layer-group" style="margin-right:5px;"></i>',
    }
    variant_classes = {
        "optimal": "badge-status-optimal",
        "passed": "badge-status-optimal",
        "detected": "badge-status-optimal",
        "margin": "badge-status-margin",
        "partial": "badge-status-margin",
        "alert": "badge-status-alert",
        "error": "badge-status-alert",
        "nominal": "badge-status-nominal",
        "scene": "badge-status-scene",
    }
    cls = variant_classes.get(variant.lower(), "badge-status-nominal")
    ic = icons.get(variant.lower(), "")
    return f'<span class="telemetry-badge {cls}">{ic}{text}</span>'
