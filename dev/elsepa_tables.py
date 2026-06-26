"""
elsepa_tables.py  (dev/, author-only tooling)

Regenerate the NIST Mott TRANSPORT cross-section tables
(``src/cxr_mc/data/mott_transport_cross_sections/DisplayCalcTCSTableFor<El>.csv``)
from ELSEPA -- the Salvat-Jablonski-Powell relativistic Dirac partial-wave elastic
code -- via the ``pyelsepa`` Docker wrapper, so the hardcoded NIST SRD-64 dataset
can be reproduced or extended on demand (TODO P2 #6).

ELSEPA's ``tcstable`` carries the 1st transport cross section sigma_tr1(E), which
is exactly what ``montecarlo._load_mott_transport`` consumes to calibrate the
screened-Rutherford screening alpha(E). Writing it back in the NIST CSV layout
makes the output a drop-in replacement for the committed table.

GATING / status: the actual ELSEPA run needs Docker + the ``elsepa`` image (built
from the paywalled ``adus_v1_0.tar.gz`` Fortran source -- see
``C:/dev/pyelsepa/docker``) + ``pyelsepa`` installed. As of this writing the image
is NOT built on this box, so :func:`elsepa_transport_cross_section` raises a clear,
actionable error and the committed NIST tables + the analytic screened-Rutherford
fallback remain in force -- nothing in the model depends on this tool at runtime.
The CSV writer/reader and the comparison harness are pure and run without Docker;
they are unit-tested in ``tests/test_elsepa_tables.py``. See
``docs/elsepa-transport.md`` for the full evaluation.

Usage (on a box where the ``elsepa`` image is built and ``pyelsepa`` is installed):
    python dev/elsepa_tables.py --element C --Z 6              # regenerate the CSV
    python dev/elsepa_tables.py --element C --Z 6 --compare    # vs the NIST table
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cxr_mc import DATA_DIR  # noqa: E402

A0_SQ_CM2 = 2.8002852e-17  # Bohr radius squared [cm^2] (NIST SRD-64 table unit)
MOTT_DIR = os.path.join(str(DATA_DIR), "mott_transport_cross_sections")

# ELSEPA tcstable column order (tcstable.dat): E, total elastic, 1st transport,
# 2nd transport. The model wants the 1st transport cross section -> column 2.
ELSEPA_SIGMA_TR1_COL = 2


def default_energy_grid_eV(n=401, e_min=50.0, e_max=3.0e5):
    """The NIST SRD-64 grid: ``n`` linearly spaced points 50 eV - 300 keV
    (the committed tables use 401, step ~750 eV)."""
    return np.linspace(e_min, e_max, n)


# ---- pure CSV I/O (no Docker; the drop-in interop) ---------------------------


def write_mott_transport_csv(path, Z, E_eV, sigma_tr_cm2):
    """Write sigma_tr(E) in the NIST ``DisplayCalcTCSTableFor<El>.csv`` layout that
    ``montecarlo._load_mott_transport`` parses: a header block, then numbered
    ``i, E_eV,  sigma`` rows with sigma in a0**2 units (cm^2 / a0^2). Returns
    ``path``."""
    E_eV = np.asarray(E_eV, float)
    sig_a0 = np.asarray(sigma_tr_cm2, float) / A0_SQ_CM2
    if E_eV.shape != sig_a0.shape:
        raise ValueError("E_eV and sigma_tr_cm2 must have the same shape")
    header = [
        "RELATIVISTIC",
        "TRANSPORT CROSS SECTIONS",
        "in units of a0**2",
        "a0**2 = 2.8002852E-21 m**2",
        "",
        f"Atomic number: {int(Z)}",
        "",
        "No, Energy,Transport cross section",
        ", eV,a0**2",
    ]
    rows = [f"{i}, {e:.6g},  {s:.6E}" for i, (e, s) in enumerate(zip(E_eV, sig_a0, strict=True), 1)]
    with open(path, "w") as f:
        f.write("\n".join(header + rows) + "\n")
    return path


def read_mott_transport_csv(path):
    """Parse a NIST/ELSEPA transport-table CSV exactly as
    ``montecarlo._load_mott_transport`` does. Returns ``(E_eV, sigma_tr_cm2)``."""
    E, sig = [], []
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                E.append(float(parts[1]))
                sig.append(float(parts[2]))
    return np.array(E), np.array(sig) * A0_SQ_CM2


def compare_tables(E1, s1, E2, s2):
    """Max relative difference of two sigma_tr(E) tables on E1's points (E2/s2 are
    log-log interpolated onto E1). Returns ``(max_rel, median_rel)``."""
    E1 = np.asarray(E1, float)
    s2_on_1 = np.exp(
        np.interp(np.log(E1), np.log(np.asarray(E2, float)), np.log(np.asarray(s2, float)))
    )
    rel = np.abs(np.asarray(s1, float) - s2_on_1) / np.abs(s2_on_1)
    return float(np.max(rel)), float(np.median(rel))


# ---- ELSEPA driver (gated: needs Docker + the elsepa image + pyelsepa) -------


def _extract_sigma_tr1_cm2(tcstable, col=ELSEPA_SIGMA_TR1_COL):
    """Pull (E_eV, sigma_tr1_cm2) from a parsed pyelsepa ``tcstable``.

    pyelsepa returns a ``cslib.DataFrame`` carrying Pint units. We read column 0
    as energy and ``col`` as the 1st transport cross section, converting both to
    eV / cm^2 when the units are available. Defensive to the exact container type
    (structured ndarray, 2-D ndarray, or DataFrame-like); VERIFY the column index
    against your ELSEPA build the first time (pass ``col=`` to override)."""
    data = getattr(tcstable, "data", tcstable)
    arr = np.asarray(data)
    if arr.dtype.names:  # structured array -> stack the named columns in order
        names = arr.dtype.names
        cols = np.column_stack([arr[n].astype(float) for n in names])
    else:
        cols = np.atleast_2d(arr).astype(float)
    E = cols[:, 0]
    sig = cols[:, col]
    # unit-convert via Pint if pyelsepa attached units; else assume eV and a0**2
    units = getattr(tcstable, "units", None)
    if units is not None:
        try:
            E = (E * units[0]).to("eV").magnitude
            sig = (sig * units[col]).to("cm**2").magnitude
            return np.asarray(E, float), np.asarray(sig, float)
        except Exception:
            pass
    return E, sig * A0_SQ_CM2  # fall back: ELSEPA tcstable is in a0**2


def elsepa_transport_cross_section(Z, energies_eV=None, *, col=ELSEPA_SIGMA_TR1_COL):
    """Run ELSEPA via pyelsepa for atomic number ``Z`` and return
    ``(E_eV, sigma_tr1_cm2)``. GATED: needs ``pyelsepa`` importable, Docker
    running, and the ``elsepa`` image built. Raises ``RuntimeError`` with guidance
    when any is missing -- the model never calls this at runtime."""
    if energies_eV is None:
        energies_eV = default_energy_grid_eV()
    try:
        from elsepa import Settings, elscata, units
    except ImportError as exc:
        raise RuntimeError(
            "pyelsepa is not importable. Install it (`pip install C:/dev/pyelsepa`) and "
            "build the `elsepa` Docker image (C:/dev/pyelsepa/docker; needs the paywalled "
            "adus_v1_0.tar.gz). Until then the committed NIST tables + the screened-"
            "Rutherford fallback (elastic_model='sr') remain in force."
        ) from exc

    energies = np.asarray(energies_eV, float)
    # ELSEPA settings mirror pyelsepa's example: free-atom Dir-Fock, default
    # exchange/polarization; IELEC=-1 (electrons). MNUCL/MELEC etc. as upstream.
    settings = Settings(
        IZ=int(Z),
        MNUCL=3,
        MELEC=4,
        MUFFIN=0,
        IELEC=-1,
        MEXCH=1,
        MCPOL=2,
        IHEF=0,
        MABS=0,
        EV=energies * units.eV,
    )
    result = elscata(settings)
    if "tcstable" not in result:
        raise RuntimeError(f"ELSEPA returned no tcstable (keys: {list(result)})")
    return _extract_sigma_tr1_cm2(result["tcstable"], col=col)


def regenerate(element, Z, out_dir=MOTT_DIR, energies_eV=None, **kw):
    """ELSEPA -> ``<out_dir>/DisplayCalcTCSTableFor<element>.csv`` (gated). Returns
    the written path."""
    E, sig = elsepa_transport_cross_section(Z, energies_eV, **kw)
    path = os.path.join(out_dir, f"DisplayCalcTCSTableFor{element}.csv")
    write_mott_transport_csv(path, Z, E, sig)
    print(f"wrote {path} ({E.size} points, {E.min():.0f}-{E.max():.0f} eV)")
    return path


def compare_to_nist(element, Z, energies_eV=None, **kw):
    """Run ELSEPA and report max/median relative difference vs the committed NIST
    table for ``element`` (gated)."""
    E_el, sig_el = elsepa_transport_cross_section(Z, energies_eV, **kw)
    nist_path = os.path.join(MOTT_DIR, f"DisplayCalcTCSTableFor{element}.csv")
    E_ni, sig_ni = read_mott_transport_csv(nist_path)
    max_rel, med_rel = compare_tables(E_el, sig_el, E_ni, sig_ni)
    print(f"{element} (Z={Z}) ELSEPA vs NIST sigma_tr: max rel {max_rel:.2%}, median {med_rel:.2%}")
    return max_rel, med_rel


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate NIST Mott transport tables from ELSEPA")
    ap.add_argument("--element", required=True, help="element symbol, e.g. C")
    ap.add_argument("--Z", type=int, required=True, help="atomic number")
    ap.add_argument("--out-dir", default=MOTT_DIR, help="output directory")
    ap.add_argument(
        "--compare", action="store_true", help="compare to the NIST table instead of writing"
    )
    ap.add_argument(
        "--col", type=int, default=ELSEPA_SIGMA_TR1_COL, help="tcstable transport column index"
    )
    args = ap.parse_args(argv)
    if args.compare:
        compare_to_nist(args.element, args.Z, col=args.col)
    else:
        regenerate(args.element, args.Z, out_dir=args.out_dir, col=args.col)


if __name__ == "__main__":
    main()
