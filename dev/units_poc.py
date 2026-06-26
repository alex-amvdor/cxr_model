"""
units_poc.py  (dev/, author-only)

Reproducible evidence + proof-of-concept behind docs/units-evaluation.md
(TODO P3 #7): does cxr_mc want a units library (Pint) project-wide?

Pint is NOT a project dependency. Run this with it pulled ephemerally:

    uv run --with pint python dev/units_poc.py

It (1) measures the costs that drive the "no project-wide refactor"
recommendation -- vectorized vs scalar overhead, the hbar=c=1 dimensional
conflict -- and (2) sketches the only adoption that would be worth it: a
lightweight, boundary-only checked constructor (no hot-path wrapping).
"""

import time


def _require_pint():
    try:
        import pint
    except ImportError as exc:
        raise SystemExit(
            "Pint is not installed. This PoC is meant to be run as:\n"
            "    uv run --with pint python dev/units_poc.py\n"
            "(Pint is deliberately NOT a project dependency -- see docs/units-evaluation.md.)"
        ) from exc
    return pint


def measure(ureg):
    import numpy as np

    # 1) vectorized: Pint delegates to numpy ufuncs -> ~free
    a = np.random.rand(2_000_000)
    q = a * ureg.eV
    n = 20
    t = time.perf_counter()
    for _ in range(n):
        _ = a * 2.0 + 1.0
    t_np = (time.perf_counter() - t) / n
    t = time.perf_counter()
    for _ in range(n):
        _ = q * 2.0 + 1.0 * ureg.eV
    t_pint = (time.perf_counter() - t) / n
    print(
        f"vectorized (2M): numpy {t_np * 1e3:.2f} ms, pint {t_pint * 1e3:.2f} ms "
        f"-> overhead x{t_pint / t_np:.1f}"
    )

    # 2) scalar: the amplitude sites are scalar-complex; this is the dealbreaker
    n = 200_000
    x = 1.5
    t = time.perf_counter()
    for _ in range(n):
        x = x * 1.0000001 + 0.5
    t_f = (time.perf_counter() - t) / n
    qs = 1.5 * ureg.eV
    t = time.perf_counter()
    for _ in range(n):
        qs = qs * 1.0000001 + 0.5 * ureg.eV
    t_q = (time.perf_counter() - t) / n
    print(
        f"scalar op: float {t_f * 1e9:.0f} ns, pint {t_q * 1e9:.0f} ns "
        f"-> overhead x{t_q / t_f:.0f}  (the amplitude hot path is scalar)"
    )

    # 3) complex magnitudes are fine
    z = (1 + 2j) * ureg.dimensionless
    print(f"complex magnitude ok: |1+2j| = {abs(z.magnitude):.4f}")

    # 4) hbar=c=1 conflict: the core mixes eV and 1/Angstrom on purpose
    pint = _require_pint()
    try:
        _ = (1.0 * ureg.eV) + (1.0 / ureg.angstrom)
        print("eV + 1/Ang allowed (no natural-unit conflict)")
    except pint.DimensionalityError:
        print("eV + 1/Ang -> DimensionalityError (hbar=c=1 core fights pint)")


# ---- the only adoption worth it: a boundary-only checked constructor ---------


class Quantity:
    """A 12-line stand-in for the recommended boundary check: assert the unit at
    construction, then carry a bare float into the hot path. No per-op overhead,
    no dependency, no hbar=c=1 conflict -- just catches "I passed keV where eV was
    expected" at the I/O edge. (Pint would also work here; this shows the floor.)"""

    __slots__ = ("value", "unit")
    _ALLOWED = {"eV", "keV", "angstrom", "cm", "sr", "rad"}

    def __init__(self, value, unit):
        if unit not in self._ALLOWED:
            raise ValueError(f"unknown unit {unit!r}; expected one of {sorted(self._ALLOWED)}")
        self.value, self.unit = float(value), unit

    def to_eV(self):
        if self.unit == "eV":
            return self.value
        if self.unit == "keV":
            return self.value * 1e3
        raise ValueError(f"cannot convert {self.unit} to eV")


def boundary_example():
    beam = Quantity(25.0, "keV")
    print(
        f"\nboundary check: beam {beam.value} {beam.unit} -> {beam.to_eV():.0f} eV "
        "(checked once, then a bare float enters mc_spectrum)"
    )
    try:
        Quantity(25.0, "kev")  # typo
    except ValueError as e:
        print(f"typo caught at the boundary: {e}")


def main():
    pint = _require_pint()
    ureg = pint.UnitRegistry()
    print(f"pint {pint.__version__}\n")
    measure(ureg)
    boundary_example()
    print(
        "\n=> see docs/units-evaluation.md: no project-wide refactor; "
        "boundary-only checking at most."
    )


if __name__ == "__main__":
    main()
