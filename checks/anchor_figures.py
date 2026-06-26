"""
anchor_figures.py  (checks/)

Publication validation figures: the Monte-Carlo model's PXR+CBS spectra
overlaid against first-principles THEORY anchors, for the Zhai et al.,
Nat. Commun. 16, 11218 (2025), Fig. 1c geometry -- tunable X-rays from a
1 mm HOPG bulk crystal under 17.5 / 20 / 22.5 / 25 keV electrons observed at
theta_obs = 119 deg into 0.066 sr, plus the bulk vs 29 nm thin-film
enhancement at 25 keV.

This is the "model versus the literature" deliverable (TODO P1 #2). With no
digitized Fig 1c curve in the repo, the model is anchored against EXACT theory:

  1. Dispersion-relation line energies  E = hbar c beta g / (1 - beta cos theta)
     (Feranchuk-Spence Eq. 10, the zero-scattering resonance): the MC peak
     positions must land on these. Drawn as vertical markers per beam energy.
  2. Feranchuk-Spence closed-form absolute line flux (Eq. 12, photons/electron
     into dOmega): a SINGLE straight segment integrated through mc_spectrum
     reproduces it to <1% (the lineshape-normalization anchor); the full
     transport-broadened MC line then shows the transport correction.
  3. Bulk vs 29 nm film enhancement (MC), with the analytic effective-length
     ratio L_eff = L_abs (1 - exp(-L_z / L_abs)) as the no-transport ceiling.

If a digitized Fig 1c curve is provided in reference_data/zhai_fig1c.csv
(schema in reference_data/README.md), the spectra figure overlays it
automatically -- so the same figure/notebook becomes a true model-vs-measured
plot the moment real data lands, with no code change.

Backend module: the functions return plain data + matplotlib Figures; main()
runs the (slow) MC, writes figures/, and prints the validation tables. The
companion notebook checks/zhai_fig1c_validation.ipynb is a thin viz wrapper.

Run (CPU-force on a box with the cupy wheel but no CUDA device):
  uv run python -c "import sys;sys.modules['cupy']=None;sys.path.insert(0,'checks');import runpy;runpy.run_path('checks/anchor_figures.py',run_name='__main__')"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tabulate import tabulate

# checks/ siblings (feranchuk_spence) and ../src (cxr_mc) on the path,
# regardless of CWD.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feranchuk_spence import photons_per_electron  # noqa: E402

from cxr_mc.crystallography import (  # noqa: E402
    CRYSTALS,
    HBARC_EV_ANG,
    absorption_length_ang,
    reciprocal_g_vector,
)
from cxr_mc.montecarlo import (  # noqa: E402
    aperture_fwhm_eV,
    beta_from_keV,
    convolve_detector,
    eds_fwhm_eV,
    mc_brem_spectrum,
    mc_spectrum,
    simulate_trajectories,
)

GRAPHITE_B_002 = 0.8  # graphite c-axis Debye-Waller B-factor [Ang^2], approx (Zhai SI)


@dataclass(frozen=True)
class ZhaiAnchor:
    """Experimental conditions of Zhai et al. Fig. 1c (SEM/EDS take-off geometry).

    All defaults reproduce the published setup; override fields to explore other
    crystals, energies or detector geometries with the same machinery.
    """

    crystal: str = "hopg"
    hkl: tuple[int, int, int] = (0, 0, 2)
    hkl_list: tuple[tuple[int, int, int], ...] = ((0, 0, 2), (0, 0, -2))
    energies_keV: tuple[float, ...] = (17.5, 20.0, 22.5, 25.0)
    theta_obs_rad: float = float(np.deg2rad(119.0))
    dtheta_obs_rad: float = float(np.deg2rad(16.6))
    domega_sr: float = 0.066
    per_nA: float = 6.2415e9  # electrons/s at 1 nA
    thick_bulk_ang: float = 1e7  # 1 mm
    thick_film_ang: float = 290.0  # 29 nm
    B_ang2: float = GRAPHITE_B_002
    e_min_eV: float = 500.0
    e_max_eV: float = 1250.0
    de_eV: float = 1.0

    @property
    def E_grid(self) -> np.ndarray:
        return np.arange(self.e_min_eV, self.e_max_eV, self.de_eV)

    @property
    def n_atoms_per_ang3(self) -> float:
        info = CRYSTALS[self.crystal]
        return len(info["basis"]) / info["V_cell"]


# ---- theory anchors (cheap, analytic) ----------------------------------------


def line_energy_eV(anchor: ZhaiAnchor, E0_keV: float) -> float:
    """Zero-scattering resonance energy, Feranchuk-Spence Eq. (10):

        E = hbar c beta g_z / (1 - beta cos theta_obs),

    for beam || g (HOPG c-axis), so g_z = |g|. The MC peak must land here.
    """
    info = CRYSTALS[anchor.crystal]
    beta = beta_from_keV(E0_keV)
    _, g = reciprocal_g_vector(anchor.hkl, info["lattice"])
    return HBARC_EV_ANG * beta * g / (1.0 - beta * np.cos(anchor.theta_obs_rad))


def theory_line_energies(anchor: ZhaiAnchor) -> dict[float, float]:
    """{beam energy [keV]: dispersion-relation line energy [eV]}."""
    return {E0: line_energy_eV(anchor, E0) for E0 in anchor.energies_keV}


def feranchuk_line_flux(anchor: ZhaiAnchor, E0_keV: float, thickness_ang: float) -> float:
    """Feranchuk-Spence Eq. (12) closed-form line flux [photons / electron into
    dOmega], absorption-limited, at the dispersion-relation line energy."""
    beta = beta_from_keV(E0_keV)
    E_line = line_energy_eV(anchor, E0_keV)
    L_abs = absorption_length_ang("C", E_line, anchor.n_atoms_per_ang3)
    return photons_per_electron(
        anchor.crystal,
        anchor.hkl,
        E_line,
        anchor.theta_obs_rad,  # geometry="lif": this slot carries theta_obs
        beta,
        L_z_ang=thickness_ang,
        L_abs_ang=L_abs,
        dOmega_sr=anchor.domega_sr,
        polarization="both",
        B_ang2=anchor.B_ang2,
        use_henke=True,
        geometry="lif",
    )


def single_segment_anchor(
    anchor: ZhaiAnchor, E0_keV: float, L_seg_ang: float = 290.0
) -> tuple[float, float, float]:
    """The cleanest analytic<->MC anchor. A single straight segment of length
    L_seg pushed through mc_spectrum, integrated over energy, equals the
    Eq. (12) closed form -- this isolates the finite-segment lineshape
    normalization from electron transport. Returns (mc_integral, closed_form,
    ratio); ratio should be ~1.
    """
    beta = beta_from_keV(E0_keV)
    E_line = line_energy_eV(anchor, E0_keV)
    fake = {
        "r_mid": np.array([[0.0, 0.0, 0.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([L_seg_ang]),
        "E_keV": np.array([E0_keV]),
        "Ne": 1,
        "thickness_ang": anchor.thick_bulk_ang,
    }
    spec = mc_spectrum(
        fake,
        anchor.E_grid,
        crystal=anchor.crystal,
        hkl_list=(anchor.hkl,),
        theta_obs_rad=anchor.theta_obs_rad,
        B_ang2=anchor.B_ang2,
    )
    mc_int = float(np.trapezoid(spec, anchor.E_grid))  # photons / electron / sr
    # Closed form per electron per sr: dOmega=1, L_abs huge so L_eff -> L_seg.
    closed = float(
        photons_per_electron(
            anchor.crystal,
            anchor.hkl,
            E_line,
            anchor.theta_obs_rad,
            beta,
            L_z_ang=L_seg_ang,
            L_abs_ang=1e12,
            dOmega_sr=1.0,
            polarization="both",
            B_ang2=anchor.B_ang2,
            use_henke=True,
            geometry="lif",
        )
    )
    return mc_int, closed, mc_int / closed


# ---- the (slow) Monte-Carlo model spectra ------------------------------------


def model_spectra(anchor: ZhaiAnchor, ne: int = 500, ne_brem: int = 200) -> dict:
    """Run the MC transport + PXR/CBS spectrum for each beam energy (1 mm bulk),
    plus the 29 nm film at the top energy. SLOW (CPU minutes for ne~500).

    Returns a dict keyed by beam energy [keV], each with the intrinsic line
    spectrum, detector-convolved line + brem, peak energy, FWHM, and the
    per-electron integrated line flux into dOmega; plus a "film" entry.
    """
    n_atoms = anchor.n_atoms_per_ang3
    E_grid = anchor.E_grid
    out: dict = {}
    for E0 in anchor.energies_keV:
        seed = int(E0 * 10)
        segs = simulate_trajectories(
            E0,
            ne,
            anchor.thick_bulk_ang,
            element="C",
            n_atoms_per_ang3=n_atoms,
            E_cut_keV=5.0,
            seed=seed,
        )
        spec = mc_spectrum(
            segs,
            E_grid,
            crystal=anchor.crystal,
            hkl_list=anchor.hkl_list,
            theta_obs_rad=anchor.theta_obs_rad,
            B_ang2=anchor.B_ang2,
        )
        beta = beta_from_keV(E0)
        E_pk = float(E_grid[np.argmax(spec)])
        fwhm = float(
            np.hypot(
                eds_fwhm_eV(E_pk),
                aperture_fwhm_eV(E_pk, beta, anchor.theta_obs_rad, anchor.dtheta_obs_rad),
            )
        )
        spec_det = convolve_detector(E_grid, spec, fwhm)
        segs_b = simulate_trajectories(
            E0,
            ne_brem,
            anchor.thick_bulk_ang,
            element="C",
            n_atoms_per_ang3=n_atoms,
            E_cut_keV=1.0,
            seed=seed + 1,
        )
        brem = mc_brem_spectrum(
            segs_b,
            E_grid,
            element="C",
            n_atoms_per_ang3=n_atoms,
            theta_obs_rad=anchor.theta_obs_rad,
        )
        brem_det = convolve_detector(E_grid, brem, fwhm)
        out[E0] = {
            "spec": spec,
            "spec_det": spec_det,
            "brem": brem,
            "brem_det": brem_det,
            "E_peak": E_pk,
            "fwhm": fwhm,
            "line_flux_per_e": float(np.trapezoid(spec, E_grid) * anchor.domega_sr),
            "backscatter": float(segs["n_backscattered"] / segs["Ne"]),
        }
    # 29 nm film at the top energy, same detector FWHM
    E_top = anchor.energies_keV[-1]
    segs_f = simulate_trajectories(
        E_top,
        ne,
        anchor.thick_film_ang,
        element="C",
        n_atoms_per_ang3=n_atoms,
        E_cut_keV=5.0,
        seed=7,
    )
    spec_f = mc_spectrum(
        segs_f,
        E_grid,
        crystal=anchor.crystal,
        hkl_list=anchor.hkl_list,
        theta_obs_rad=anchor.theta_obs_rad,
        B_ang2=anchor.B_ang2,
    )
    spec_f_det = convolve_detector(E_grid, spec_f, out[E_top]["fwhm"])
    out["film"] = {
        "E0_keV": E_top,
        "spec": spec_f,
        "spec_det": spec_f_det,
        "line_flux_per_e": float(np.trapezoid(spec_f, E_grid) * anchor.domega_sr),
        "n_transmitted": int(segs_f.get("n_transmitted", 0)),
    }
    return out


# ---- optional digitized reference (model-vs-measured hook) -------------------


def reference_curve(anchor: ZhaiAnchor | None = None, path: str | Path | None = None):
    """Load a digitized Zhai Fig 1c curve if present, else return None.

    Schema (see reference_data/README.md): a CSV with columns
        series, energy_eV, intensity
    where `series` labels the beam energy (e.g. "25keV"). Returns
        {series_label: (energy_eV[np], intensity[np])}
    or None when no file exists, so callers degrade gracefully to theory-only.
    """
    import pandas as pd

    if path is None:
        path = _HERE / "reference_data" / "zhai_fig1c.csv"
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    ecol = cols.get("energy_ev") or cols.get("energy")
    icol = cols.get("intensity") or cols.get("counts")
    scol = cols.get("series") or cols.get("label")
    if ecol is None or icol is None:
        raise ValueError(
            f"reference CSV {path} needs energy_eV and intensity columns; got {list(df.columns)}"
        )
    if scol is None:
        df = df.assign(_series="measured")
        scol = "_series"
    return {str(s): (g[ecol].to_numpy(float), g[icol].to_numpy(float)) for s, g in df.groupby(scol)}


def _match_series(reference: dict, E0_keV: float) -> str | None:
    """Find the reference series whose label parses to ~E0_keV (tolerant of
    '25', '25keV', '25.0 keV', etc.)."""
    for key in reference:
        digits = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in key)
        for tok in digits.split():
            try:
                if abs(float(tok) - E0_keV) < 0.25:
                    return key
            except ValueError:
                continue
    return None


# ---- figures -----------------------------------------------------------------


def figure_spectra(anchor: ZhaiAnchor, model: dict, reference: dict | None = None):
    """Fig 1c analog vs theory: model spectra (intrinsic + detector-convolved
    with bremsstrahlung) with the Eq.(10) dispersion-relation line energies as
    vertical markers, the 29 nm film overlay, and -- if provided -- the
    digitized measured curve scaled to the model peak per beam energy."""
    import matplotlib.pyplot as plt

    scale = anchor.domega_sr * anchor.per_nA  # per e/sr/eV -> Phs/eV/s/nA
    lines = theory_line_energies(anchor)
    fig, (ax_i, ax_d) = plt.subplots(1, 2, figsize=(13, 5))
    for i, E0 in enumerate(anchor.energies_keV):
        m = model[E0]
        c = f"C{i}"
        ax_i.plot(anchor.E_grid, m["spec"] * scale, color=c, label=f"{E0:g} keV")
        ax_d.plot(
            anchor.E_grid,
            (m["spec_det"] + m["brem_det"]) * scale,
            color=c,
            label=f"{E0:g} keV (bulk)",
        )
        ax_d.plot(anchor.E_grid, m["brem_det"] * scale, color=c, ls="--", lw=0.9)
        for ax in (ax_i, ax_d):
            ax.axvline(lines[E0], color=c, ls=":", lw=1.2, alpha=0.7)
    if "film" in model:
        ax_d.plot(
            anchor.E_grid,
            model["film"]["spec_det"] * scale,
            "k-",
            lw=1.0,
            label=f"{model['film']['E0_keV']:g} keV, 29 nm film",
        )
    if reference:
        for i, E0 in enumerate(anchor.energies_keV):
            key = _match_series(reference, E0)
            if key is None:
                continue
            e, inten = reference[key]
            peak = float((model[E0]["spec_det"] + model[E0]["brem_det"]).max() * scale)
            y = inten / np.nanmax(inten) * peak  # scale measured shape to model peak
            ax_d.scatter(
                e,
                y,
                s=14,
                facecolors="none",
                edgecolors=f"C{i}",
                alpha=0.8,
                label=f"{E0:g} keV (measured)",
            )
    ax_i.set_title("Intrinsic PXR+CBS (1 mm HOPG)\ndotted = Eq.(10) line energy")
    ax_d.set_title("EDS-convolved: peaks + brem (dashed)\ndotted = Eq.(10) line energy")
    for ax in (ax_i, ax_d):
        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel("Intensity (Phs/eV/s/nA)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(r"Model vs theory: PXR+CBS from HOPG, $\theta_{obs}$=119$\degree$, 0.066 sr")
    fig.tight_layout()
    return fig


def figure_flux_anchor(anchor: ZhaiAnchor, model: dict):
    """Absolute-flux anchor. Left: the single straight-segment MC/closed-form
    ratio per beam energy (must hover at 1 -- validates the lineshape
    normalization). Right: per-electron integrated line flux, MC full transport
    (bulk) vs the Feranchuk Eq.(12) bulk estimate, log-y."""
    import matplotlib.pyplot as plt

    E0s = np.array(anchor.energies_keV, float)
    ratios = np.array([single_segment_anchor(anchor, E0)[2] for E0 in anchor.energies_keV])
    mc_bulk = np.array([model[E0]["line_flux_per_e"] for E0 in anchor.energies_keV])
    fer_bulk = np.array(
        [feranchuk_line_flux(anchor, E0, anchor.thick_bulk_ang) for E0 in anchor.energies_keV]
    )

    fig, (ax_r, ax_f) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_r.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
    ax_r.plot(E0s, ratios, "o-", color="C3")
    ax_r.set_ylim(0.9, 1.1)
    ax_r.set_xlabel("Beam energy (keV)")
    ax_r.set_ylabel("MC / closed-form (single segment)")
    ax_r.set_title("Lineshape-normalization anchor\n(Eq. 12, should be 1)")
    ax_r.grid(alpha=0.3)

    ax_f.semilogy(E0s, mc_bulk, "o-", color="C0", label="MC (full transport, bulk)")
    ax_f.semilogy(E0s, fer_bulk, "s--", color="C1", label="Feranchuk Eq.(12), bulk")
    ax_f.set_xlabel("Beam energy (keV)")
    ax_f.set_ylabel("Line flux (photons / electron into 0.066 sr)")
    ax_f.set_title("Absolute line flux: MC vs analytic\n(transport vs escape-limited)")
    ax_f.grid(alpha=0.3, which="both")
    ax_f.legend(fontsize=8)
    fig.tight_layout()
    return fig


def figure_enhancement(anchor: ZhaiAnchor, model: dict):
    """Bulk vs 29 nm film at the top beam energy (EDS-convolved), annotated with
    the MC enhancement factor and the analytic no-transport ceiling
    L_eff(bulk)/L_eff(film)."""
    import matplotlib.pyplot as plt

    E0 = anchor.energies_keV[-1]
    scale = anchor.domega_sr * anchor.per_nA
    bulk = model[E0]["spec_det"] * scale
    film = model["film"]["spec_det"] * scale
    mc_enh = float(bulk.max() / film.max())

    E_line = line_energy_eV(anchor, E0)
    L_abs = absorption_length_ang("C", E_line, anchor.n_atoms_per_ang3)
    leff_bulk = L_abs * (1.0 - np.exp(-anchor.thick_bulk_ang / L_abs))
    leff_film = L_abs * (1.0 - np.exp(-anchor.thick_film_ang / L_abs))
    ceiling = float(leff_bulk / leff_film)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(anchor.E_grid, bulk, color="C0", label=f"{E0:g} keV, 1 mm bulk")
    ax.plot(anchor.E_grid, film, color="k", label=f"{E0:g} keV, 29 nm film")
    ax.axvline(E_line, color="C3", ls=":", lw=1.2, alpha=0.7, label="Eq.(10) line energy")
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Intensity (Phs/eV/s/nA)")
    ax.set_title(
        f"Bulk vs thin-film enhancement\nMC peak ratio = {mc_enh:.1f}x  "
        f"(no-transport ceiling {ceiling:.0f}x)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ---- validation table + CLI --------------------------------------------------


def validation_table(anchor: ZhaiAnchor, model: dict) -> list[list]:
    """Rows: per beam energy, the MC peak position vs the Eq.(10) line energy,
    the single-segment MC/closed-form ratio, and the MC bulk line flux."""
    lines = theory_line_energies(anchor)
    rows = []
    for E0 in anchor.energies_keV:
        m = model[E0]
        _, _, ratio = single_segment_anchor(anchor, E0)
        rows.append(
            [
                f"{E0:g}",
                f"{m['E_peak']:.0f}",
                f"{lines[E0]:.0f}",
                f"{m['E_peak'] - lines[E0]:+.0f}",
                f"{ratio:.3f}",
                f"{m['line_flux_per_e']:.3e}",
                f"{m['backscatter']:.3f}",
            ]
        )
    return rows


def main(outdir: str = "figures", ne: int = 500, ne_brem: int = 200) -> None:
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass

    anchor = ZhaiAnchor()
    print(f"Running MC model spectra (ne={ne}, ne_brem={ne_brem}); slow on CPU...")
    model = model_spectra(anchor, ne=ne, ne_brem=ne_brem)
    reference = reference_curve(anchor)
    print("reference data:", "LOADED" if reference else "none (theory-only overlay)")

    print()
    print(
        tabulate(
            validation_table(anchor, model),
            headers=[
                "E0\n[keV]",
                "MC peak\n[eV]",
                "Eq.10\n[eV]",
                "diff\n[eV]",
                "MC/closed\n(1 seg)",
                "line flux\n[ph/e/0.066sr]",
                "backscatter",
            ],
            tablefmt="github",
        )
    )
    E0 = anchor.energies_keV[-1]
    enh = model[E0]["spec_det"].max() / model["film"]["spec_det"].max()
    print(
        f"\n25 keV bulk vs 29 nm film enhancement: {enh:.1f}x  "
        f"(film transmitted {model['film']['n_transmitted']} electrons)"
    )

    outpath = _HERE.parent / outdir
    outpath.mkdir(exist_ok=True)
    figs = {
        "zhai_fig1c_spectra_vs_theory": figure_spectra(anchor, model, reference),
        "zhai_flux_anchor": figure_flux_anchor(anchor, model),
        "zhai_bulk_vs_film_enhancement": figure_enhancement(anchor, model),
    }
    for name, fig in figs.items():
        for ext in ("png", "pdf"):
            fig.savefig(outpath / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        print("wrote", (outpath / f"{name}.png").relative_to(_HERE.parent))


if __name__ == "__main__":
    main()
