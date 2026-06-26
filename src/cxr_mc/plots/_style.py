"""_style

Colour palette + per-energy colour map shared across every figure.
"""

COLORS = ["r", "y", "g", "b", "m", "c", "k", "orange", "purple", "brown"]

# Beam energy -> colour, CONSISTENT across every figure: a given E0 always gets
# the same colour (keyed to its rank in the sorted energy set, so e.g. 30/45/60
# keV map to the same three colours everywhere), and the palette stays readable
# on white (no low-contrast yellow). Pass the FULL set of energies present in
# the figure so the rank -- hence the colour -- is stable panel to panel.
_ENERGY_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


def energy_color(E0, energies):
    """Stable colour for beam energy ``E0`` given the figure's set of energies."""
    order = sorted({float(e) for e in energies})
    try:
        i = order.index(float(E0))
    except ValueError:
        i = 0
    return _ENERGY_PALETTE[i % len(_ENERGY_PALETTE)]
