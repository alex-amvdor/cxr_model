"""
eaglexo_response.py
===================

Forward model of a Raptor Photonics **Eagle XO** camera *recording* an incident
X-ray spectrum. Everything upstream (montecarlo / crystallography)
predicts the photons that LEAVE the sample; this module predicts what the Eagle
XO collects, which is set -- as the datasheet makes plain and unlike the
Timepix3 quad -- by just two things:

  1. the SOLID ANGLE it subtends (sensor size / source distance), which scales
     the flux that lands on it, and
  2. the QUANTUM EFFICIENCY QE(E), which shapes that flux with energy.

There is no counting threshold, no charge-sharing tail, no Time-over-Threshold
energy noise to model: the Eagle XO is a deep-cooled, back-illuminated,
direct-detection CCD with an open (windowless) front end, so a photon that is
absorbed in the active silicon is collected. That makes this model a clean
``solid_angle x QE(E)`` operator rather than the per-photon Monte Carlo the
Timepix needed -- see timepix_response.py for the counting-detector machinery.


The two knobs in detail
-----------------------
**Solid angle** -- :func:`solid_angle_sr` is the exact on-axis solid angle of a
rectangular sensor of full size (w, h) at distance d,
    Omega = 4 * arctan( (w/2)(h/2) / (d * sqrt((w/2)^2 + (h/2)^2 + d^2)) ),
which reduces to the familiar A/d^2 when the sensor is small vs the distance.
:func:`geometry` packages this with the polar-angle span dtheta_obs (used by the
line-broadening model in montecarlo.aperture_fwhm_eV) for either sensor
variant, ready to splat into a ``sweep.Sweep``. The flux scaling itself is
applied UPSTREAM via ``case['domega_sr']`` -> ``r['scale']`` (see
results.store_result), exactly as for the Timepix; this module only needs to
supply the number.

**Quantum efficiency** -- :func:`qe` returns QE(E) for the chosen sensor
coating. The canonical curve is the manufacturer's measured QE, digitized from
the datasheet figure into ``data/eaglexo_qe.csv`` (12 eV - 29 keV); it carries
the real soft-X-ray structure (the EUV peak, the ~150 eV dip, the rise to ~95%
near 1 keV), the **Si-K edge notch at ~1.84 keV**, and the high-energy roll-off
as the ~15 um back-thinned silicon turns transparent (QE ~ 2% by 20 keV, so the
camera is efficient for the soft PXR lines but POOR for hard bremsstrahlung).
Above the table's 29 keV ceiling QE is continued by the optically-thin silicon
limit QE ~ 1/L_abs(E) (anchored to the last tabulated point) rather than clipped
to zero, so detected brem out to a 30-60 keV beam energy stays finite -- the
same lesson as the window-QE fix for the SDD model.


Coatings
--------
The datasheet plots two back-illuminated options:
  * ``"BN"``  -- back-illuminated, NO anti-reflection coating (the headline
                 "back illuminated with no coating" option). The default.
  * ``"BEN"`` -- back-illuminated, enhanced: better soft-X-ray/EUV response
                 below ~1 keV; identical to BN at and above 1 keV (the two
                 datasheet curves overlap there).


Optional photon-counting (energy-resolving) mode
------------------------------------------------
A CCD integrates charge and does not, by itself, return a spectrum. But a
direct-detection CCD run at low occupancy (<~1 photon per pixel cluster per
frame) can be used as an energy-dispersive single-photon counter -- the listed
XRF / source-characterization / spectroscopy applications. In that mode each
collected photon's energy is measured to the Fano + read-noise limit,
    FWHM(E) = 2.3548 * W_Si * sqrt( F (E / W_Si) + n_pix * sigma_read^2 ),
which is ~60 eV at 1 keV -- far sharper than the Timepix ToT (~1 keV) but real.
:meth:`EagleResponse.apply` leaves the spectrum un-blurred by default (the
``solid_angle x QE`` view the datasheet's two parameters give you); pass
``resolve_energy=True`` for the photon-counting line shape.


Units
-----
Energies in eV; sensor dimensions and distance in mm; solid angle in sr;
QE dimensionless in [0, 1]; charge in electrons (e-). The photon<->charge bridge
is W_Si = 3.65 eV/pair (each absorbed photon of energy E makes E/W_Si electrons).

### The working DISTANCE and the active-silicon thickness are flagged '### FILL
### IN'. The Eagle XO is an open-front camera bolted to a vacuum chamber, so the
### source-to-sensor distance -- which, with the sensor size, sets the whole
### solid angle -- is experiment-specific. Set DEFAULT_DISTANCE_M (or pass
### distance_m=) to your geometry before trusting absolute rates.
"""

from pathlib import Path

import numpy as np

from crystallography import absorption_length_ang

# ---- silicon sensor physics (fixed material constants) -----------------------
W_EHP_EV = 3.65  # mean energy to make one electron-hole pair [eV]
FANO_SI = 0.115  # Fano factor: Var(N) = F * N (sub-Poisson) for Si
SI_DENSITY_G_CM3 = 2.329
SI_A = 28.085  # Si molar mass [g/mol]
# number density n = rho/A * N_A, converted cm^-3 -> Ang^-3 (the unit
# absorption_length_ang wants). 0.602214076 = N_A * 1e-24.
SI_N_PER_ANG3 = SI_DENSITY_G_CM3 / SI_A * 0.602214076

# ---- sensor variants (fixed, from the Eagle XO datasheet) --------------------
# Active area and pixel pitch for the two CCD options; the active area (with the
# source distance) is what sets the solid angle.
SENSORS = {
    "4240": dict(
        sensor="E2V 42-40", n_pix=(2048, 2048), pixel_um=13.5, active_mm=(27.6, 27.6)
    ),  # the larger sensor
    "4710": dict(
        sensor="E2V 47-10", n_pix=(1024, 1024), pixel_um=13.0, active_mm=(13.3, 13.3)
    ),
}
DEFAULT_SENSOR = "4240"

# ---- operating point (### FILL IN with your geometry / readout) --------------
DEFAULT_DISTANCE_M = 0.4  ### FILL IN -- source-to-sensor distance [m] (placeholder,
#   matched to the Timepix default for a like-for-like
#   comparison; this sets the solid angle, the key knob)
ACTIVE_SI_UM = 16.0  ### FILL IN -- active (depleted) Si thickness [um]; only
#   shapes the QE roll-off extrapolation ABOVE the 29 keV
#   datasheet ceiling, fitted to that roll-off (~14-19 um)

# ---- readout / noise (datasheet; used only by the 'measured' realization) -----
READ_NOISE_E = 2.3  # read noise [e- RMS], 75 kHz typical (9.0 e- @ 2 MHz)
DARK_CURRENT_E_PER_S = 0.0005  # dark current [e-/pixel/s], deep-cooled (<0.0005)
FULL_WELL_E = 100_000.0  # full-well capacity [e-] (100 ke- typical)

# QE table: digitized datasheet curve (energy_eV, QE_BN_%, QE_BEN_%)
_QE_PATH = Path(__file__).parent.parent / "data" / "eaglexo_qe.csv"


# ---- geometry: the solid angle (knob 1) --------------------------------------
def solid_angle_sr(width_mm, height_mm, distance_mm):
    """Exact on-axis solid angle [sr] of a rectangular sensor of full size
    (width, height) whose centre is a distance ``distance_mm`` from the source,
    facing it:

        Omega = 4 * arctan( a b / (d sqrt(a^2 + b^2 + d^2)) ),  a=w/2, b=h/2.

    This is the rigorous result for a centred rectangle (not the small-angle
    A/d^2, though it reduces to it when a, b << d: the Eagle's 27.6 mm sensor at
    0.4 m differs from A/d^2 by <0.1%, but at a few cm the exact form matters).
    Scalar in, scalar out."""
    a, b, d = 0.5 * width_mm, 0.5 * height_mm, float(distance_mm)
    return 4.0 * np.arctan(a * b / (d * np.sqrt(a * a + b * b + d * d)))


def geometry(sensor=DEFAULT_SENSOR, distance_m=None):
    """Solid angle + polar-angle span for one sensor variant at a given distance.

    Returns a dict ready to splat the geometry into a sweep::

        geo = eaglexo_response.geometry("4240", distance_m=0.25)
        sweep = Sweep(..., domega_sr=geo["domega_sr"],
                           dtheta_obs_deg=geo["dtheta_obs_deg"])

    Fields: ``domega_sr`` (knob 1, the solid angle), ``dtheta_obs_deg`` (the
    polar angular span 2 arctan((h/2)/d), used by the line-broadening model),
    plus the descriptive ``sensor``, ``active_mm`` and ``distance_m`` echoed back.
    """
    if sensor not in SENSORS:
        raise ValueError(f"unknown sensor {sensor!r} (have {list(SENSORS)})")
    s = SENSORS[sensor]
    d_m = DEFAULT_DISTANCE_M if distance_m is None else float(distance_m)
    w_mm, h_mm = s["active_mm"]
    d_mm = d_m * 1e3
    return dict(
        sensor=s["sensor"],
        sensor_key=sensor,
        active_mm=s["active_mm"],
        distance_m=d_m,
        domega_sr=solid_angle_sr(w_mm, h_mm, d_mm),
        dtheta_obs_deg=float(np.degrees(2.0 * np.arctan((h_mm / 2.0) / d_mm))),
    )


def sweep_geometry(sensor=DEFAULT_SENSOR, distance_m=None):
    """Just the two ``sweep.Sweep`` geometry overrides for this camera, ready
    to splat -- this is how you point a sweep at the Eagle XO solid angle instead
    of the default Timepix quad::

        sweep = Sweep(material="mose2", thickness_ang=2e4,
                      **eaglexo_response.sweep_geometry("4240", distance_m=0.25))

    Returns ``{"domega_sr": ..., "dtheta_obs_deg": ...}`` only (no extra keys, so
    it splats without tripping the Sweep constructor)."""
    g = geometry(sensor, distance_m)
    return {"domega_sr": g["domega_sr"], "dtheta_obs_deg": g["dtheta_obs_deg"]}


# ---- quantum efficiency: QE(E) (knob 2) --------------------------------------
_QE_CACHE = None


def _tail_slope(Et, col, lo=4000.0):
    """Power-law index d(ln QE)/d(ln E) of the high-energy QE roll-off, from a
    log-log fit to the table above ``lo`` eV. Used to continue QE past the table
    ceiling: in that clean roll-off the sensor is optically thin, QE ~ mu ~ a
    power law (~E^-2.3 for Si here), so a fit to the measured roll-off is a
    sturdier extrapolant than a fresh Henke evaluation -- which is unavailable
    above ~30 keV anyway (the same ceiling the SDD window-QE fix hit)."""
    m = Et >= lo
    return float(np.polyfit(np.log(Et[m]), np.log(np.clip(col[m], 1e-6, None)), 1)[0])


def load_qe_table(path=_QE_PATH):
    """Load the digitized datasheet QE table -> (E_eV, QE_BN, QE_BEN, slope_BN,
    slope_BEN); QE columns are fractions in [0, 1] (the file stores percent) and
    the slopes are the high-energy roll-off indices for the >29 keV tail. Cached
    after the first read."""
    global _QE_CACHE
    if _QE_CACHE is None:
        d = np.genfromtxt(path, delimiter=",", comments="#")
        Et, bn, ben = d[:, 0], d[:, 1] / 100.0, d[:, 2] / 100.0
        _QE_CACHE = (Et, bn, ben, _tail_slope(Et, bn), _tail_slope(Et, ben))
    return _QE_CACHE


def qe(E_eV, coating="BN"):
    """Quantum efficiency of the Eagle XO at photon energy E [eV], in [0, 1].

    Inside the datasheet range (12 eV - 29 keV) this is the manufacturer's
    measured curve for the chosen ``coating`` ("BN" default, or "BEN"),
    log-energy interpolated -- so it carries the EUV peak, the soft-X-ray dip,
    the rise to ~95% near 1 keV, and the Si-K notch at ~1.84 keV verbatim.

      * Below 12 eV -> 0 (no data; no simulated flux lives there anyway).
      * Above 29 keV -> a power-law continuation of the measured roll-off,
        QE(E) = QE(E_max) * (E / E_max)^slope (slope ~ -2.3, see _tail_slope).
        Anchored to the last tabulated point, so it is continuous; this keeps
        detected brem finite out to a 30-60 keV beam rather than clipping it to
        zero at the table edge. (A Henke 1/L_abs tail can't be used here -- the
        f2 tables stop at ~30 keV, exactly where we need the extrapolation.)

    Accepts scalars or arrays; returns the same shape."""
    E = np.asarray(E_eV, dtype=float)
    Et, bn, ben, sl_bn, sl_ben = load_qe_table()
    col, slope = (ben, sl_ben) if coating.upper() == "BEN" else (bn, sl_bn)
    out = np.interp(
        np.log10(np.clip(E, 1e-3, None)), np.log10(Et), col, left=0.0, right=col[-1]
    )  # clamp top; overwrite >Emax
    tail = col[-1] * (np.clip(E, Et[-1], None) / Et[-1]) ** slope
    out = np.where(E > Et[-1], tail, out)
    return np.clip(out, 0.0, 1.0)


def qe_absorption_model(E_eV, active_um=None, peak=0.93):
    """Physics cross-check / tuning curve: pure absorption in the active silicon,
        QE(E) = peak * (1 - exp(-t / L_abs(E))),
    with t = ``active_um`` (default ACTIVE_SI_UM) and L_abs from Henke f2. This
    is NOT the default QE (it misses the back-surface dead-layer behaviour that
    shapes the measured soft-X-ray response and the Si-K notch), but it
    reproduces the high-energy roll-off and lets you retune the sensor thickness
    or model a different back-thinning. ``peak`` caps it for entrance reflection
    / incomplete collection. Valid where Henke f2 is tabulated (~30 eV - 30 keV);
    outside that it returns 0 (NaN guarded), so restrict overlays to that range."""
    t = ACTIVE_SI_UM if active_um is None else active_um
    E = np.asarray(E_eV, dtype=float)
    L_ang = absorption_length_ang("Si", np.clip(E, 1e-3, None), SI_N_PER_ANG3)
    out = peak * (1.0 - np.exp(-(t * 1e4) / L_ang))  # t: um -> Angstrom
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---- photon-counting (energy-resolving) mode resolution ----------------------
def energy_fwhm_eV(E_eV, n_pix=4):
    """Single-photon energy-measurement FWHM [eV] in photon-counting mode:
    Fano statistics on the e-h pair count plus read noise over the ``n_pix``
    pixels the charge cloud covers, in quadrature on the variance:
        sigma_N^2 = F (E / W_Si) + n_pix sigma_read^2   [electrons^2]
        FWHM = 2.3548 * W_Si * sigma_N.
    ~60 eV at 1 keV for n_pix=4 -- Fano/read-limited, i.e. far better than the
    Timepix ToT and roughly sqrt(E)-shaped. Irrelevant in plain integrating
    mode; used only when EagleResponse.apply(resolve_energy=True)."""
    E = np.asarray(E_eV, dtype=float)
    var_e = FANO_SI * (E / W_EHP_EV) + n_pix * READ_NOISE_E**2  # [e-^2]
    return 2.3548 * W_EHP_EV * np.sqrt(var_e)  # [eV]


class EagleResponse:
    """Eagle XO response bound to one fixed energy grid + coating.

    The whole detector is ``incident x QE(E)`` (optionally followed by the
    photon-counting energy blur), so unlike the Timepix this is a per-bin
    multiply rather than a re-binning operator -- ``apply`` keeps the input grid.

    Parameters
    ----------
    E_grid_eV : the energy grid the spectra live on [eV] (need not be uniform
        for the bare QE multiply; uniform is required only for resolve_energy).
    coating : "BN" (default) or "BEN".
    resolve_energy : default False (the ``solid_angle x QE`` view). True applies
        the Fano + read-noise photon-counting line shape (needs a uniform grid).
    n_pix : pixels per photon cluster for the energy-resolution term.
    """

    def __init__(self, E_grid_eV, *, coating="BN", resolve_energy=False, n_pix=4):
        self.E = np.asarray(E_grid_eV, dtype=float)
        self.coating = coating
        self.resolve_energy = resolve_energy
        self.n_pix = n_pix
        self.qe = qe(self.E, coating=coating)  # QE on the grid, in [0,1]

    def apply(self, spec):
        """Detected spectral density on the SAME grid and in the SAME flux units
        as the input (e.g. Phs/eV/s/nA): ``spec * QE(E)``, optionally blurred by
        the photon-counting energy resolution. NaN/inf samples (a bad-geometry
        case) are treated as zero flux rather than poisoning the result."""
        spec = np.nan_to_num(
            np.asarray(spec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        if spec.shape != self.E.shape:
            raise ValueError(
                f"spec length {spec.shape} != response grid {self.E.shape}. "
                f"Build a matching response with "
                f"eaglexo_response.get_response(E_grid, ...)."
            )
        det = spec * self.qe
        if self.resolve_energy:
            from montecarlo import convolve_detector

            dE = self.E[1] - self.E[0]
            fwhm = float(np.median(energy_fwhm_eV(self.E, self.n_pix)))
            det = convolve_detector(self.E, det, fwhm)  # ~const, sqrt(E)-weak
        return det

    def detection_efficiency(self, E_eV=None):
        """QE interpolated onto ``E_eV`` (default: the response grid). This IS
        the detection efficiency for this windowless direct-detection CCD -- no
        threshold turn-on, just absorption x collection."""
        if E_eV is None:
            return self.qe
        return qe(E_eV, coating=self.coating)

    def charge_density(self, spec):
        """Detected CHARGE spectral density [e-/eV/...] -- what the CCD actually
        records, as opposed to the photon density :meth:`apply` returns.

        A CCD integrates charge: it has no per-photon counting, so it cannot
        report a photon spectrum. Each absorbed photon of energy E deposits
        ``E / W_Si`` electrons, so the recorded signal is the detected photon
        density weighted by photon ENERGY, not photon number::

            charge_density(E) = spec(E) * QE(E) * (E / W_Si)

        That energy weighting is the qualitative difference from a photon counter
        (the Timepix): it tilts the recorded signal toward harder photons, partly
        offsetting the thin-sensor QE roll-off. Same grid as the input; units are
        the input flux units * e- (e.g. e-/eV/s/nA for a Phs/eV/s/nA input)."""
        det = self.apply(spec)  # detected photon density [Phs/eV/...]
        return det * (self.E / W_EHP_EV)  # -> electrons (charge) per eV

    def integrated_charge(self, spec):
        """Total detected charge RATE: trapz of :meth:`charge_density` over the
        grid (input flux units * e-, e.g. e-/s/nA for a Phs/eV/s/nA input).

        This is the scalar "brightness" the CCD reports for a spectrum -- no
        energy information survives, just collected charge. Multiply by the beam
        current [nA] and the exposure [s] to get the electrons accumulated, and
        compare to :data:`FULL_WELL_E` for saturation."""
        return float(np.trapezoid(self.charge_density(spec), self.E))


# Responses are pure functions of (grid, coating, mode), so cache them -- a
# results set mixing grids (e.g. distinct line/brem grids) gets one response each.
_RESPONSE_CACHE = {}


def get_response(E_grid_eV, *, coating="BN", resolve_energy=False, n_pix=4):
    """Cached :class:`EagleResponse` for a grid + settings: built once per unique
    signature and reused. Prefer this over constructing EagleResponse directly
    when looping over many spectra so each gets a response matching ITS grid."""
    E = np.asarray(E_grid_eV, dtype=float)
    key = (
        E.size,
        round(float(E[0]), 6),
        round(float(E[-1]), 6),
        coating.upper(),
        bool(resolve_energy),
        int(n_pix),
    )
    resp = _RESPONSE_CACHE.get(key)
    if resp is None:
        resp = EagleResponse(
            E, coating=coating, resolve_energy=resolve_energy, n_pix=n_pix
        )
        _RESPONSE_CACHE[key] = resp
    return resp


def poisson_counts(
    E_grid_eV, detected_per_s, time_s, *, add_read_dark=False, n_pix_line=4, rng=None
):
    """Draw a Poisson realization of one acquisition -- what the camera records,
    statistical noise and all.

    Expected counts in a bin are rate * bin_width * live_time, where the rate is
    the detected density already scaled to absolute units:
        detected_per_s : detected density [Phs/eV/s] on E_grid_eV, i.e.
                         EagleResponse.apply(spec) * beam_current (* any per-nA).
    With ``add_read_dark`` the photon-counting read + dark noise is added as a
    Gaussian on each bin (sigma in counts = sqrt(n_pix_line) * READ_NOISE / N_eh
    per photon-equivalent, plus dark over the exposure) -- usually negligible
    next to photon Poisson for these rates, off by default.
    Returns (counts_per_bin, expected_per_bin): the integer draw and its mean."""
    rng = np.random.default_rng() if rng is None else rng
    E = np.asarray(E_grid_eV, dtype=float)
    dE = E[1] - E[0]
    expected = np.clip(np.asarray(detected_per_s, dtype=float) * dE * time_s, 0.0, None)
    counts = rng.poisson(expected).astype(float)
    if add_read_dark:
        # read noise in charge -> equivalent photon-count jitter per bin, plus
        # accumulated dark charge; both small for a deep-cooled, low-noise CCD
        sigma_cts = np.sqrt(n_pix_line) * READ_NOISE_E / np.maximum(E / W_EHP_EV, 1.0)
        counts = counts + rng.normal(0.0, sigma_cts, counts.shape)
        counts += DARK_CURRENT_E_PER_S * time_s * W_EHP_EV / np.maximum(E, 1.0)
    return counts, expected
