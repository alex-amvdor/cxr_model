"""
sweep.py
============

Define a parameter sweep and expand it into the per-case dicts that
``montecarlo.run_case`` consumes.

The driver notebook sets ONE :class:`Sweep`. Every physical parameter accepts
**either a single value (fixed) or a sequence/array (swept)**; :func:`build_cases`
takes the Cartesian product over whatever is swept. So

    Sweep(tilt_deg=-30.0,              tilt_azim_deg=0.0)          # one case geometry
    Sweep(tilt_deg=np.linspace(-36,-1,14), tilt_azim_deg=[-9,0,9])  # 14*3 geometries

are both valid and need no other code changes.

Crystallography (composition, dominant reflections, zone axis, B-factor, default
energy grid) is looked up per material; the detector geometry defaults to the
2x2 Timepix3 quad. Only ``crystallography`` is imported here (no GPU), so
this module is cheap to import and test.
"""

from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence, Union

import numpy as np

from .crystallography import CRYSTALS, dominant_reflections

# ---- Timepix3 quad geometry (fixed hardware) --------------------------------
TIMEPIX3_PIXEL_PITCH_M = 55e-6
TIMEPIX3_CHIP_WIDTH_M = 256 * TIMEPIX3_PIXEL_PITCH_M
TIMEPIX3_DISTANCE_M = 0.4
TIMEPIX3_DTHETA_OBS_DEG = float(
    np.degrees(2 * np.arctan((TIMEPIX3_CHIP_WIDTH_M / 2) / TIMEPIX3_DISTANCE_M))
)
TIMEPIX3_DOMEGA_SR = float(TIMEPIX3_CHIP_WIDTH_M**2 / TIMEPIX3_DISTANCE_M**2)

# pretty labels for the material column / config names
MATERIAL_LABELS = {
    "mose2": "MoSe2",
    "wse2": "WSe2",
    "mote2": "MoTe2",
    "mos2": "MoS2",
    "ws2": "WS2",
    "ptse2": "PtSe2",
    "hfse2": "HfSe2",
    "zrse2": "ZrSe2",
    "hopg": "HOPG",
    "diamond": "diamond",
    "silicon": "silicon",
}

ScalarOrSeq = Union[float, Sequence[float], np.ndarray]


def fmt_thickness(t_ang):
    """Compact human thickness label from Angstroms: 316A / 31.6nm / 17um / 1mm."""
    if t_ang < 1e2:
        return f"{t_ang:g}A"
    if t_ang < 1e4:
        return f"{t_ang / 10:g}nm"
    if t_ang < 1e7:
        return f"{t_ang / 1e4:g}um"
    return f"{t_ang / 1e7:g}mm"


def n_of(crystal, element):
    """Number density [1/Ang^3] of one element in a crystal's unit cell."""
    info = CRYSTALS[crystal]
    count = sum(1 for el, _ in info["basis"] if el == element)
    return count / info["V_cell"]


def pm(*hkls):
    """A list of reflections together with their negatives."""
    out = []
    for h in hkls:
        out += [tuple(h), tuple(-x for x in h)]
    return out


# amorphous substrate number densities [1/Ang^3], from bulk mass density:
#   n_formula = rho[g/cc] * 0.602214 / M[g/mol], then * per-element stoichiometry
_SUBSTRATE_COMP = {
    "sio2": [("Si", 0.02205), ("O", 0.04410)],   # fused silica, rho=2.20, M=60.08
    "al2o3": [("Al", 0.04702), ("O", 0.07053)],  # sapphire,     rho=3.98, M=101.96
}


def substrate_composition(substrate):
    """Number-density composition [(element, n_per_Ang3), ...] for a substrate.
    Amorphous presets ('sio2', 'al2o3') come from bulk density; a crystalline
    substrate already in CRYSTALS (e.g. 'silicon') uses its unit-cell density."""
    if substrate.lower() in _SUBSTRATE_COMP:
        return [(el, n) for el, n in _SUBSTRATE_COMP[substrate.lower()]]
    if substrate in CRYSTALS:
        from collections import Counter

        info = CRYSTALS[substrate]
        counts = Counter(el for el, _ in info["basis"])
        return [(el, c / info["V_cell"]) for el, c in counts.items()]
    raise ValueError(
        f"unknown substrate {substrate!r}; use one of {list(_SUBSTRATE_COMP)} "
        f"or a crystal key in {list(CRYSTALS)}"
    )


def film_on_substrate_layers(
    film_composition, film_thickness_ang, substrate, substrate_thickness_ang
):
    """Absorber stack [(z_top, z_bot, composition), ...] for a film at the
    entrance face (z=0..t_film) on a substrate (t_film..t_film+t_sub). Attach as
    a case's ``abs_layers`` so emitted lines/brem are attenuated by the WHOLE
    stack (the film still RADIATES via the case's crystal/hkl_list). Beam enters
    the film side; with negative tilt (front exit) the substrate sits BEHIND the
    emission and does not attenuate -- it bites the back-exit / transmission
    geometry. See docs/multilayer-materials.md."""
    t_f = float(film_thickness_ang)
    t_s = float(substrate_thickness_ang)
    return [
        (0.0, t_f, [(el, float(n)) for el, n in film_composition]),
        (t_f, t_f + t_s, substrate_composition(substrate)),
    ]


def crystal_params(material, n_families=4):
    """Fixed crystallography for a material: composition, the dominant
    reflections, the beam zone axis [uvw], the (isotropic) B-factor, and a
    sensible default photon-energy grid. Override the grid via Sweep.e_grid_eV."""
    if material == "mose2":
        return dict(
            crystal="mose2",
            composition=[("Mo", n_of("mose2", "Mo")), ("Se", n_of("mose2", "Se"))],
            hkl_list=dominant_reflections("mose2", n_families=n_families, B_ang2=0.6),
            beam_uvw=(0, 0, 2),
            B_ang2=0.6,
            E_grid=np.arange(350.0, 1750.0, 3.0),
        )
    if material == "wse2":
        # isostructural with MoSe2; W has no NIST Mott table so transport falls
        # back to analytic screened-Rutherford screening for W (see montecarlo).
        return dict(
            crystal="wse2",
            composition=[("W", n_of("wse2", "W")), ("Se", n_of("wse2", "Se"))],
            hkl_list=dominant_reflections("wse2", n_families=n_families, B_ang2=0.6),
            beam_uvw=(0, 0, 2),
            B_ang2=0.6,
            E_grid=np.arange(350.0, 2500.0, 3.0),
        )
    if material == "mote2":
        # 2H-MoTe2 (alpha), isostructural with MoSe2. Te has no NIST Mott table
        # -> transport falls back to analytic screened-Rutherford screening for Te
        # (see montecarlo), as for W/S/Pt/Hf/Zr.
        return dict(
            crystal="mote2",
            composition=[("Mo", n_of("mote2", "Mo")), ("Te", n_of("mote2", "Te"))],
            hkl_list=dominant_reflections("mote2", n_families=n_families, B_ang2=0.6),
            beam_uvw=(0, 0, 2),
            B_ang2=0.6,
            E_grid=np.arange(350.0, 2500.0, 3.0),
        )
    if material in ("ws2", "mos2"):
        # 2H disulfides, isostructural with WSe2/MoSe2 (small in-plane a -> bright).
        # S has no NIST Mott table -> analytic SR screening fallback.
        metal = {"ws2": "W", "mos2": "Mo"}[material]
        return dict(
            crystal=material,
            composition=[(metal, n_of(material, metal)), ("S", n_of(material, "S"))],
            hkl_list=dominant_reflections(material, n_families=n_families, B_ang2=0.6),
            beam_uvw=(0, 0, 2),
            B_ang2=0.6,
            E_grid=np.arange(350.0, 2500.0, 3.0),
        )
    if material in ("ptse2", "hfse2", "zrse2"):
        # 1T (CdI2-type): heavy metal at the ORIGIN -> every (00l) stays strong,
        # so the bright basal series marches up in energy with the tight c. These
        # metals (Pt/Hf/Zr) have no NIST Mott table -> analytic SR screening.
        metal = {"ptse2": "Pt", "hfse2": "Hf", "zrse2": "Zr"}[material]
        return dict(
            crystal=material,
            composition=[(metal, n_of(material, metal)), ("Se", n_of(material, "Se"))],
            hkl_list=dominant_reflections(material, n_families=n_families, B_ang2=0.6),
            beam_uvw=(0, 0, 1),  # c-axis along the beam (1T: 1 layer/cell)
            B_ang2=0.6,
            E_grid=np.arange(350.0, 3500.0, 3.0),
        )
    if material == "diamond":
        return dict(
            crystal="diamond",
            composition=[("C", n_of("diamond", "C"))],
            hkl_list=dominant_reflections(
                "diamond", n_families=n_families, B_ang2=0.21
            ),
            beam_uvw=(4, 0, 0),
            B_ang2=0.21,
            E_grid=np.arange(100.0, 5000.0, 2.0),
        )
    if material == "silicon":
        return dict(
            crystal="silicon",
            composition=[("Si", n_of("silicon", "Si"))],
            hkl_list=dominant_reflections(
                "silicon", n_families=n_families, B_ang2=0.46
            ),
            beam_uvw=(4, 4, 0),
            B_ang2=0.46,
            E_grid=np.arange(100.0, 5000.0, 3.0),
        )
    if material == "hopg":
        # HOPG is fiber-textured: only the (00l) c-axis reflections are coherent.
        return dict(
            crystal="hopg",
            composition=[("C", n_of("hopg", "C"))],
            hkl_list=pm((0, 0, 2), (0, 0, 4)),
            beam_uvw=(0, 0, 1),
            B_ang2=0.8,
            E_grid=np.arange(100.0, 5000.0, 3.0),
        )
    raise ValueError(f"unknown material {material!r} (have {list(MATERIAL_LABELS)})")


@dataclass
class Sweep:
    """One parameter sweep.

    Each of ``thickness_ang``, ``energy_keV``, ``tilt_deg`` and
    ``tilt_azim_deg`` is either a single number (fixed) or a sequence/array
    (swept); build_cases() takes the product. The remaining fields are fixed
    setup that rarely changes per run.
    """

    material: str  # required: no default, so a Sweep can't silently load MoSe2
    thickness_ang: ScalarOrSeq = 2e4
    energy_keV: ScalarOrSeq = (30.0, 45.0, 60.0)
    tilt_deg: ScalarOrSeq = -30.0
    tilt_azim_deg: ScalarOrSeq = 0.0
    # fixed setup (single values) ------------------------------------------
    theta_obs_deg: float = 90.0
    n_families: int = 4
    # two independent photon-energy grids (None -> per-material defaults):
    #   E_grid_line : fine + NARROW; where the coherent lines are evaluated (the
    #       expensive sinc^2). Lines are kinematically capped at a few keV, so it
    #       need not extend past ~4 keV.
    #   E_grid_brem : coarse + WIDE; where the smooth bremsstrahlung is evaluated
    #       (cheap). Extend to 20-40 keV / the beam energy to model the full
    #       measured spectrum without inflating the line cost. Default spans the
    #       line start up to the highest beam energy at a 50 eV step.
    E_grid_line: Optional[np.ndarray] = None
    E_grid_brem: Optional[np.ndarray] = None
    e_grid_eV: Optional[np.ndarray] = None  # deprecated: alias for E_grid_line
    dtheta_obs_deg: Optional[float] = None  # None -> Timepix3 default
    domega_sr: Optional[float] = None  # None -> Timepix3 default
    beam_uvw: Optional[tuple] = None  # None -> per-material default
    # GPU memory knobs: segments per matmul in the spectrum / brem kernels (None
    # -> 40000 / 20000 defaults). Lower them (e.g. 4000) to cap peak GPU memory on
    # a busy or shared device; the cost is only a little extra loop overhead.
    spec_chunk: Optional[int] = None
    brem_chunk: Optional[int] = None
    # crystal mosaicity (the INITIAL ANALYTIC broadening; switchable per run).
    #   mosaic=False (default) -> OFF: perfect crystal, an exact no-op.
    #   mosaic=True            -> apply the per-crystal mosaic_fwhm_deg from
    #       crystal_structures.toml. Crystals without a value (diamond, silicon, the
    #       TMDs) stay perfect even with mosaic=True, so this is the "optional,
    #       excluding crystals which lack such data" switch -- HOPG has a value (0.8
    #       deg) and so DOES broaden when mosaic=True.
    #   mosaic_fwhm_deg        -> override the per-crystal value for ALL crystals in
    #       the sweep (e.g. HOPG ZYA 0.4 / ZYB 0.8 / ZYH 3.5), or supply one for a
    #       crystal that has none. Ignored unless mosaic=True.
    mosaic: bool = False
    mosaic_fwhm_deg: Optional[float] = None
    # film-on-substrate stack (optional, FIRST SLICE of the multilayer feature --
    # docs/multilayer-materials.md). substrate=None -> free-standing film
    # (unchanged). Otherwise the film's emitted lines + brem are attenuated by the
    # substrate via each case's abs_layers; transport + radiation still use the
    # film material (substrate backscatter / substrate emission are deferred).
    substrate: Optional[str] = None  # "sio2" | "al2o3" | a crystal key e.g. "silicon"
    substrate_thickness_ang: float = 5e6  # 0.5 mm default


def _seq(x):
    """Normalize a scalar-or-sequence into a 1-D float array, order preserved."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def build_cases(sweep: Sweep, n_electrons=450, n_electrons_brem=100):
    """Expand a :class:`Sweep` into a list of run_case dicts (the Cartesian
    product over the swept thickness / tilt / azimuth, each crossed with every
    beam energy). Returns the ``cases`` list; preview it with
    :func:`geometry_table`."""
    cp = crystal_params(sweep.material, sweep.n_families)
    # line grid: fine + narrow (per-material default, E_grid_line, or the
    # deprecated e_grid_eV alias). brem grid: coarse + wide -- default spans the
    # line start up to the highest beam energy at 50 eV (brem cuts off at the
    # beam energy, so that's the full physical range); override via E_grid_brem.
    line_src = sweep.E_grid_line if sweep.E_grid_line is not None else sweep.e_grid_eV
    line_grid = cp["E_grid"] if line_src is None else np.asarray(line_src, float)
    energies = _seq(sweep.energy_keV)
    if sweep.E_grid_brem is not None:
        brem_grid = np.asarray(sweep.E_grid_brem, float)
    else:
        brem_grid = np.arange(
            float(line_grid[0]), float(energies.max()) * 1e3 + 50.0, 50.0
        )

    dtheta = (
        TIMEPIX3_DTHETA_OBS_DEG
        if sweep.dtheta_obs_deg is None
        else sweep.dtheta_obs_deg
    )
    domega = TIMEPIX3_DOMEGA_SR if sweep.domega_sr is None else sweep.domega_sr
    beam_uvw = cp["beam_uvw"] if sweep.beam_uvw is None else sweep.beam_uvw
    label = MATERIAL_LABELS.get(sweep.material, sweep.material)

    # crystal mosaicity (analytic, optional): None unless the run enables it AND the
    # crystal has a mosaic_fwhm_deg (or the Sweep overrides it). None -> perfect
    # crystal, so store_result adds no mosaic term (exact no-op).
    mosaic_deg = None
    if sweep.mosaic:
        mosaic_deg = (
            sweep.mosaic_fwhm_deg
            if sweep.mosaic_fwhm_deg is not None
            else CRYSTALS[cp["crystal"]].get("mosaic_fwhm_deg")
        )
    mosaic_fwhm_rad = float(np.deg2rad(mosaic_deg)) if mosaic_deg else None

    def _triple(g):
        """(start, stop, step) so np.arange(*triple) reproduces grid g."""
        step = float(g[1] - g[0])
        return (float(g[0]), float(g[-1]) + step, step)

    line_triple, brem_triple = _triple(line_grid), _triple(brem_grid)

    cases = []
    for i_c, (thickness, tilt, azim) in enumerate(
        product(
            _seq(sweep.thickness_ang), _seq(sweep.tilt_deg), _seq(sweep.tilt_azim_deg)
        )
    ):
        name = f"{label} {fmt_thickness(thickness)} pol={tilt:g} az={azim:g}"
        abs_layers = None
        if sweep.substrate is not None:
            name = f"{name} on {sweep.substrate}"
            abs_layers = film_on_substrate_layers(
                cp["composition"],
                thickness,
                sweep.substrate,
                sweep.substrate_thickness_ang,
            )
        for i_e, E0 in enumerate(energies):
            cases.append(
                dict(
                    name=name,
                    crystal=cp["crystal"],
                    composition=cp["composition"],
                    hkl_list=cp["hkl_list"],
                    B_ang2=cp["B_ang2"],
                    E0_keV=float(E0),
                    thickness_ang=float(thickness),
                    E_grid=line_triple,  # legacy key (== line grid)
                    E_grid_line=line_triple,
                    E_grid_brem=brem_triple,
                    theta_obs_rad=np.deg2rad(sweep.theta_obs_deg),
                    tilt_deg=float(tilt),
                    tilt_azim_deg=float(azim),
                    beam_uvw=beam_uvw,
                    mosaic_fwhm_rad=mosaic_fwhm_rad,  # None -> perfect crystal
                    abs_layers=abs_layers,  # None -> single slab; else film-on-substrate stack
                    brem_file=None,
                    Ne=n_electrons,
                    Ne_brem=n_electrons_brem,
                    seed=1000 * i_c + 10 * i_e + 1,
                    spec_chunk=sweep.spec_chunk,  # GPU rows/matmul (None -> run_case default)
                    brem_chunk=sweep.brem_chunk,
                    # used downstream (detector model / unit scaling); ignored by run_case:
                    dtheta_obs_rad=np.deg2rad(dtheta),
                    domega_sr=domega,
                )
            )
    return cases


def geometry_table(cases):
    """A one-row-per-config DataFrame summarizing the geometry of a case list,
    for a quick sanity check before running."""
    import pandas as pd

    def _grid(triple):
        """'0.05-4 keV @ 3 eV' label from a (start, stop, step) grid triple."""
        if triple is None:
            return "-"
        s0, s1, ds = triple
        return f"{s0 / 1e3:g}-{(s1 - ds) / 1e3:g} keV @ {ds:g} eV"

    rows, seen = [], set()
    for c in cases:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        same = [k for k in cases if k["name"] == c["name"]]
        rows.append(
            {
                "config": c["name"],
                "refl": len(c["hkl_list"]),
                "t [um]": c["thickness_ang"] / 1e4,
                "polar [deg]": round(c["tilt_deg"], 2),
                "azim [deg]": round(c["tilt_azim_deg"], 2),
                "energies [keV]": [k["E0_keV"] for k in same],
                "line grid": _grid(c.get("E_grid_line", c["E_grid"])),
                "brem grid": _grid(c.get("E_grid_brem")),
                "theta_obs [deg]": round(np.degrees(c["theta_obs_rad"]), 1),
                "dOmega [sr]": c["domega_sr"],
            }
        )
    return pd.DataFrame(rows)
