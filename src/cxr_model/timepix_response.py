"""
timepix_response.py
===================

Forward model of a 2x2 Timepix3 quad (silicon sensor) *recording* an incident
X-ray spectrum. Everything upstream (montecarlo / crystallography)
predicts the photons that LEAVE the sample and reach the detector solid angle;
this module predicts what the detector actually COUNTS, which is a very
different spectrum once the threshold, charge sharing, and energy noise are in.

Why it matters here: the counting threshold is ~525 e- ~= 1.9 keV, which sits
*above* much of the MoSe2 line signal. So the headline effect is not Gaussian
blurring -- it is wholesale loss of sub-2 keV flux plus a low-energy tail. The
model makes that quantitative.


The per-photon chain
--------------------
  1. Photoabsorption     The photon is absorbed with probability
                         eps_abs(E) = 1 - exp(-t / L_abs(E)), t = sensor
                         thickness, L_abs from Henke f2 (we reuse
                         crystallography.absorption_length_ang). Photons
                         that pass through deposit nothing and are never counted.
  2. Charge generation   An absorbed photon makes N = E / W electron-hole pairs
                         (W = 3.65 eV in Si), with Fano broadening
                         sigma_N = sqrt(F * N). (At a few keV this is ~15 e-,
                         negligible next to the 124 e- ToT noise, but it is free
                         to include in the MC.)
  3. Charge sharing      The cloud drifts the full sensor thickness (backside
                         illumination) and diffuses to a Gaussian of width
                         sigma_diff = d * sqrt(2kT/qV). On the 55 um pixel pitch
                         it splits across a 3x3 block of pixels; pixel (i,j) gets
                         the integral of that Gaussian over the pixel's area,
                         which factorizes into a product of 1-D erf differences.
  4. Counting threshold  Each pixel fragment fires iff its charge clears the
                         discriminator: Q + Gauss(0, sigma_thr) > Q_thr. A photon
                         is COUNTED iff >= 1 fragment fires. Fragments that fall
                         below threshold are simply lost -- this both kills the
                         efficiency near threshold and removes charge from the
                         survivors (the low-energy tail).
  5. Recorded energy     Cluster energy = W * sum over FIRED pixels of
                         (Q + Gauss(0, ToT_res)). Untriggered fragments
                         contribute nothing, so a shared event reads back LOW.
  6. Counting statistics Over an acquisition the counts in each bin are
                         Poisson(rate * bin_width * live_time)  (poisson_counts).

Stages 1-2 are scalar physics; 3-5 depend on where in the pixel the photon
landed and how the cloud split, so they are Monte-Carlo'd per photon.


How the two layers fit together
-------------------------------
build_response() runs the per-photon MC (stages 1-5) at each of a grid of input
energies and packs the result into a linear operator:

    R[j, i] = P(an incident photon of energy E_in[i] is recorded in output
               energy bin j),  already multiplied by eps_abs(E_in[i]).

So a column of R is the detector's response to a monochromatic line: a Gaussian
core near E_in[i] plus the charge-loss tail, with the missing area being the
photons that were not counted. The column sum is the detection efficiency
P_det(E_in[i]); 1 - column_sum is the probability the photon produced no count.

Because R is linear, the detected spectrum of ANY input is just R applied to the
input photon counts. TimepixResponse precomputes R once for a fixed energy grid
and exposes .apply(spec) as a cheap matrix-multiply, so all 120 simulated cases
reuse a single Monte Carlo (~3 s) instead of re-sampling photons each time.


Units
-----
Lengths in micrometres (um) unless a name says _ang/_eV; energies in eV; charge
in electrons (e-). The electron <-> energy bridge is W = 3.65 eV/pair, so the
quoted front-end figures convert as:
    threshold   500-550 e-  ->  1.83-2.01 keV   (we use 525 e- = 1.92 keV)
    ENC         61 e-       ->  223 eV          (discriminator noise)
    ToT res     124 e-      ->  453 eV RMS      (= 1.07 keV FWHM) per pixel

### The hardware constants flagged '### FILL IN' are PLACEHOLDERS. Set them to
### the real quad values before trusting absolute rates -- BIAS_VOLTAGE_V most
### of all, since sigma_diff ~ 1/sqrt(V) controls the whole charge-sharing /
### tail / turn-on behaviour.
"""

import numpy as np
from scipy.special import erf

from .crystallography import absorption_length_ang

# ---- silicon sensor physics (fixed material constants) -----------------------
W_EHP_EV = 3.65  # mean energy to make one electron-hole pair [eV]
FANO_SI = 0.115  # Fano factor: Var(N) = F * N (sub-Poisson) for Si
SI_DENSITY_G_CM3 = 2.329
SI_A = 28.085  # Si molar mass [g/mol]
# number density n = rho/A * N_A, converted cm^-3 -> Ang^-3 (the unit
# absorption_length_ang wants). 0.602214076 = N_A * 1e-24.
SI_N_PER_ANG3 = SI_DENSITY_G_CM3 / SI_A * 0.602214076
K_OVER_Q_V_PER_K = 8.617333e-5  # Boltzmann constant / elementary charge [V/K]

# ---- hardware / operating point (### FILL IN with the real quad values) -------
SENSOR_THICKNESS_UM = 300.0  ### FILL IN -- Si sensor thickness [um] (placeholder)
BIAS_VOLTAGE_V = 100.0  ### FILL IN -- applied reverse bias [V] (placeholder)
PIXEL_PITCH_UM = 55.0  # Timepix3 pixel pitch [um] (fixed by the chip)
TEMPERATURE_K = 300.0  ### FILL IN if the quad runs cooled (placeholder)

# ---- front-end electronics (translated from the quoted electron figures) -----
THRESHOLD_E = 525.0  # discriminator threshold Q_thr [e-] (quoted 500-550)
ENC_E = 61.0  # equivalent noise charge on the discriminator [e-]
THRESHOLD_DISP_E = 35.0  # residual pixel-to-pixel threshold spread after
#   equalization [e-]
TOT_RES_E = 124.0  # Time-over-Threshold energy resolution, per pixel
#   [e-]: folds in ToT quantization, gain spread,
#   and per-pixel calibration residuals

# Two DISTINCT noises, kept separate on purpose (they act at different stages):
#   * whether a pixel FIRES is governed by the discriminator noise -- ENC and
#     the post-equalization threshold dispersion, independent, so in quadrature:
SIGMA_THR_E = float(np.hypot(ENC_E, THRESHOLD_DISP_E))  # ~70 e- counting noise
#   * how well a fired pixel's charge is MEASURED is the ToT resolution, which we
#     also carry in energy units for the analytic FWHM:
SIGMA_TOT_EV = TOT_RES_E * W_EHP_EV  # ~453 eV per pixel


def sigma_diffusion_um(thickness_um=None, bias_v=None, temperature_k=None):
    """RMS transverse width of the charge cloud when it reaches the pixels.

    A cloud created at the back face drifts the full thickness d under the field
    E = V/d. Its transverse spread is set by diffusion during the drift:
        sigma^2 = 2 D t_drift,
        D = mu * kT/q           (Einstein relation),
        t_drift = d / (mu E) = d^2 / (mu V).
    The mobility mu cancels, leaving the mobility-independent result
        sigma = d * sqrt(2 kT / (q V)).
    Backside illumination means every photon drifts the whole thickness, so we
    use d, not the (energy-dependent) absorption depth. Lower bias -> longer
    drift -> wider cloud -> more sharing; sigma ~ 1/sqrt(V) is the dominant knob.
    Example: d=300 um, V=100 V, T=300 K -> sigma ~ 6.8 um (on a 55 um pitch)."""
    d = SENSOR_THICKNESS_UM if thickness_um is None else thickness_um
    V = BIAS_VOLTAGE_V if bias_v is None else bias_v
    T = TEMPERATURE_K if temperature_k is None else temperature_k
    return d * np.sqrt(2.0 * K_OVER_Q_V_PER_K * T / V)


def absorption_efficiency(E_eV, thickness_um=None):
    """Probability a photon is photoabsorbed in the sensor (Beer-Lambert):
        eps_abs(E) = 1 - exp(-t / L_abs(E)),
    with L_abs the attenuation length from Henke f2 (absorption_length_ang).
    For 300 um Si this is ~1 from ~2 keV up to ~8 keV (the diode is essentially
    black there) and only starts dropping above ~10 keV as Si turns transparent.
    The Si K-edge at 1.839 keV is present but invisible in eps_abs because the
    sensor is already fully absorbing on both sides of it -- it would only show
    in a much thinner sensor. Accepts scalars or arrays."""
    d = SENSOR_THICKNESS_UM if thickness_um is None else thickness_um
    E = np.asarray(E_eV, dtype=float)
    L_abs_ang = absorption_length_ang("Si", E, SI_N_PER_ANG3)  # [Angstrom]
    return 1.0 - np.exp(-(d * 1e4) / L_abs_ang)  # d: um -> Angstrom


def energy_fwhm_eV(E_eV, n_pix=1):
    """Analytic photopeak FWHM for a cluster summed over n_pix fired pixels.

    Two contributions add in quadrature on the variance:
      * ToT measurement noise: SIGMA_TOT_EV per pixel, so n_pix pixels summed
        give n_pix * SIGMA_TOT_EV^2 (this flat ~453 eV/pixel term dominates ->
        ~1.07 keV FWHM, roughly energy-independent, unlike a Fano-limited SDD);
      * Fano statistics on the pair count: Var = F*N pairs = F*(E/W), i.e.
        W*F*E in eV^2 (tens of eV at a few keV -- negligible here).
    This is the analytic core only; the MC effective width (build_response's
    fwhm_rec) is larger because it also includes the charge-loss tail."""
    E = np.asarray(E_eV, dtype=float)
    sigma2 = n_pix * SIGMA_TOT_EV**2 + W_EHP_EV * FANO_SI * E  # variance [eV^2]
    return 2.3548 * np.sqrt(sigma2)  # 2*sqrt(2 ln2)*sigma


def _split_fractions(x_um, y_um, sigma_um, pitch_um, half=1):
    """Fraction of a photon's charge collected by each pixel of a (2*half+1)^2
    block, for a Gaussian cloud of width sigma centred at (x, y) measured from
    the centre of the hit pixel.

    The collected fraction for a pixel is the integral of the normalized 2-D
    Gaussian over that pixel's square. A 2-D isotropic Gaussian factorizes into
    independent x and y Gaussians, so the area integral factorizes too:
        f_ij = [integral over pixel i in x] * [integral over pixel j in y].
    Each 1-D integral of a unit Gaussian (centre x, width sigma) over [a, b] is
        1/2 [ erf((b - x)/(sqrt2 sigma)) - erf((a - x)/(sqrt2 sigma)) ].
    Pixel k spans [(k-0.5)*pitch, (k+0.5)*pitch] (k = 0 is the hit pixel), giving
    the per-axis fraction arrays fx, fy; the outer product fx_i * fy_j is the
    per-pixel fraction. With sigma ~ 7-10 um and pitch 55 um, 3*sigma stays well
    inside the 3x3 block (half=1), so the fractions sum to ~1 and almost no
    charge escapes the block. Fully vectorized over the n_photon hit positions;
    returns shape (n_photon, (2*half+1)^2)."""
    k = np.arange(-half, half + 1)  # pixel indices, e.g. [-1,0,1]
    lo = (k - 0.5) * pitch_um  # each pixel's lower edges
    hi = (k + 0.5) * pitch_um  # each pixel's upper edges
    s2 = np.sqrt(2.0) * sigma_um  # the sqrt(2) sigma in erf
    # broadcast hit positions (n,1) against pixel edges (1,m) -> (n_photon, m)
    fx = 0.5 * (
        erf((hi[None, :] - x_um[:, None]) / s2)
        - erf((lo[None, :] - x_um[:, None]) / s2)
    )
    fy = 0.5 * (
        erf((hi[None, :] - y_um[:, None]) / s2)
        - erf((lo[None, :] - y_um[:, None]) / s2)
    )
    # outer product over the two pixel axes -> (n_photon, m, m), flattened to m*m
    return (fx[:, :, None] * fy[:, None, :]).reshape(x_um.size, -1)


def build_response(
    E_in_eV,
    E_out_edges_eV,
    *,
    thickness_um=None,
    bias_v=None,
    n_mc=60000,
    neighborhood=1,
    seed=0,
):
    """
    Monte-Carlo the detector response on a grid of input photon energies.

    For each input energy E_in[i] we fire n_mc photons that have ALREADY been
    absorbed (stages 2-5 below) and histogram the recorded cluster energies;
    the photoabsorption probability eps_abs(E_in[i]) (stage 1) is folded in
    afterwards as an overall weight on that column. Running the MC only on
    absorbed photons -- rather than wasting samples on ones that pass through --
    keeps the statistics where they matter.

    Parameters
    ----------
    E_in_eV : (n_in,) input photon energies to characterize [eV].
    E_out_edges_eV : (n_out+1,) bin edges for the recorded-energy histogram [eV].
    thickness_um, bias_v : override the module hardware defaults.
    n_mc : photons per input energy (statistics of each response column).
    neighborhood : half-width of the pixel block; 1 -> 3x3 (ample for sigma<<pitch).
    seed : RNG seed (responses are reproducible).

    Returns a dict:
      R       : (n_out, n_in) response matrix. R[j,i] = probability that an
                incident photon at E_in[i] is recorded in output bin j, already
                multiplied by eps_abs. Each column is the detector's line shape
                for that energy; the column sum is P_det(E_in[i]).
      P_det   : (n_in,) detection efficiency = eps_abs * (fraction that fire).
      eps_abs : (n_in,) photoabsorption efficiency alone.
      E_out   : (n_out,) recorded-energy bin centres [eV].
      mean_rec, fwhm_rec : (n_in,) MC mean recorded energy and effective FWHM
                (2.3548 * std of the detected events). The charge-loss tail pulls
                mean_rec BELOW E_in and inflates fwhm_rec above the analytic core.
      sigma_diff_um : the diffusion width used (diagnostic).
    """
    rng = np.random.default_rng(seed)
    E_in = np.asarray(E_in_eV, dtype=float)
    edges = np.asarray(E_out_edges_eV, dtype=float)
    E_out = 0.5 * (edges[:-1] + edges[1:])  # bin centres
    pitch = PIXEL_PITCH_UM
    sigma = sigma_diffusion_um(thickness_um, bias_v)  # same for every energy
    eps_abs = absorption_efficiency(E_in, thickness_um)

    R = np.zeros((E_out.size, E_in.size))
    P_det = np.zeros(E_in.size)
    mean_rec = np.full(E_in.size, np.nan)
    fwhm_rec = np.full(E_in.size, np.nan)

    for i, E in enumerate(E_in):
        # -- stage 3a: where in the hit pixel did the photon land? uniform over
        #    the 55x55 um cell (relative to the pixel centre, in [-pitch/2, +pitch/2])
        x = rng.uniform(-0.5, 0.5, n_mc) * pitch
        y = rng.uniform(-0.5, 0.5, n_mc) * pitch
        # -- stage 2: number of electron-hole pairs, Fano-broadened; clip the
        #    (rare) negative tail of the Gaussian at zero
        mean_N = E / W_EHP_EV
        N_tot = np.clip(rng.normal(mean_N, np.sqrt(FANO_SI * mean_N), n_mc), 0.0, None)
        # -- stage 3b: split that charge across the 3x3 block by the erf areas;
        #    Q is electrons per pixel, shape (n_mc, 9)
        Q = _split_fractions(x, y, sigma, pitch, neighborhood) * N_tot[:, None]
        # -- stage 4: discriminator. A pixel fires if its charge plus independent
        #    counting noise clears the threshold. any_fired -> the photon is counted.
        fired = Q + rng.normal(0.0, SIGMA_THR_E, Q.shape) > THRESHOLD_E
        any_fired = fired.any(axis=1)
        # -- stage 5: recorded energy = W * sum of the FIRED pixels' measured
        #    charge (true charge + independent ToT noise). Unfired fragments add
        #    nothing -> shared events read low (the charge-loss tail). Keep only
        #    photons that produced a count, and clip tiny negative noise excursions.
        Q_meas = Q + rng.normal(0.0, TOT_RES_E, Q.shape)
        rec_E = W_EHP_EV * np.where(fired, Q_meas, 0.0).sum(axis=1)
        rec_E = np.clip(rec_E[any_fired], 0.0, None)

        # this energy's response column: histogram of recorded energies,
        # normalized per incident photon (1/n_mc) and weighted by eps_abs so the
        # column integrates to P_det. Photons that never fired just don't appear.
        hist, _ = np.histogram(rec_E, bins=edges)
        R[:, i] = hist / n_mc * eps_abs[i]
        P_det[i] = any_fired.mean() * eps_abs[i]
        if rec_E.size:  # tail/peak diagnostics
            mean_rec[i] = rec_E.mean()
            fwhm_rec[i] = 2.3548 * rec_E.std()

    return dict(
        R=R,
        P_det=P_det,
        eps_abs=eps_abs,
        E_out=E_out,
        mean_rec=mean_rec,
        fwhm_rec=fwhm_rec,
        sigma_diff_um=sigma,
    )


class TimepixResponse:
    """
    A precomputed detector response bound to one fixed (uniform) energy grid, so
    many spectra share a single Monte Carlo.

    Speed/quality strategy -- two deliberately COARSE internal grids:

      * coarse INPUT grid (dE_mc, ~50 eV): the MC characterizes the response at
        these energies. We can bin the input this coarsely because the detector
        then smears everything by >~450 eV (the ToT resolution); a 50 eV
        placement error inside a bin is invisible after that blur. This keeps the
        MC to ~50 energies instead of ~1000 fine bins.
      * coarse OUTPUT grid (dE_out, ~25 eV): the recorded-energy histogram. Wider
        bins -> more counts per bin -> a smoother response with the same n_mc.

    .apply() then (1) bins a fine input spectrum into the coarse input grid
    conserving total photons, (2) matmuls through R, (3) interpolates the
    detected density back onto the fine grid. The detector's own blur makes that
    interpolation lossless in practice.

    Parameters
    ----------
    E_grid_eV : the fine, uniform energy grid the spectra live on [eV].
    dE_mc, dE_out : coarse input / output bin widths [eV].
    n_mc, seed : Monte-Carlo photons per input energy, and RNG seed.
    thickness_um, bias_v : override the module hardware defaults.
    """

    def __init__(
        self,
        E_grid_eV,
        *,
        dE_mc=50.0,
        dE_out=25.0,
        n_mc=60000,
        seed=0,
        thickness_um=None,
        bias_v=None,
    ):
        E = np.asarray(E_grid_eV, dtype=float)
        self.E = E  # the fine output grid
        self.dE_fine = float(E[1] - E[0])  # fine bin width [eV]
        lo, hi = float(E[0]), float(E[-1])

        # --- coarse INPUT grid ---------------------------------------------
        # edges start half a fine-bin below the first sample so each fine bin
        # falls cleanly inside one coarse bin
        in_edges = np.arange(lo - self.dE_fine / 2, hi + dE_mc, dE_mc)
        self.E_in = 0.5 * (in_edges[:-1] + in_edges[1:])  # coarse bin centres
        self.n_in = self.E_in.size
        # precompute which coarse-input bin each fine bin maps to, so .apply()
        # can re-bin any spectrum with a single bincount (flux-conserving)
        self.idx_in = np.clip(((E - in_edges[0]) / dE_mc).astype(int), 0, self.n_in - 1)

        # --- coarse OUTPUT grid --------------------------------------------
        # recorded energy can sit below E_in (charge loss) or above it (ToT
        # noise has ~0.5 keV width), so pad the top by 4*sigma_tot and start at 0
        out_edges = np.arange(0.0, hi + 4.0 * SIGMA_TOT_EV + dE_out, dE_out)
        self.dE_out = dE_out

        # run the Monte Carlo once; everything below is reused on every .apply()
        resp = build_response(
            self.E_in,
            out_edges,
            n_mc=n_mc,
            seed=seed,
            thickness_um=thickness_um,
            bias_v=bias_v,
        )
        self.R = resp["R"]  # (n_out, n_in) response operator
        self.E_out = resp["E_out"]  # coarse output bin centres
        self.P_det = resp["P_det"]  # detection efficiency on E_in
        self.eps_abs = resp["eps_abs"]
        self.mean_rec = resp["mean_rec"]
        self.fwhm_rec = resp["fwhm_rec"]
        self.sigma_diff_um = resp["sigma_diff_um"]

    def apply(self, spec):
        """Detected spectral density on the fine grid, in the SAME flux units per
        eV as the input.

        `spec` is the photon spectrum incident on the detector (e.g.
        Phs/eV/s/nA into the detector solid angle) and must be on the grid this
        response was built for -- if unsure, fetch a matching response with
        get_response(E_grid). NaN/inf samples (e.g. a bad-geometry case) are
        treated as zero flux rather than poisoning the result.

        Bookkeeping: spec*dE_fine is photons per fine bin; bincount sums those
        into photons per coarse-input bin; R @ that is detected photons per
        coarse-OUTPUT bin (R already carries absorption + counting efficiency);
        dividing by dE_out makes it a density again; interp lifts it back onto
        the fine grid. Total detected photons are conserved through the chain."""
        spec = np.nan_to_num(
            np.asarray(spec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        if spec.shape != self.E.shape:
            raise ValueError(
                f"spec length {spec.shape} != response grid {self.E.shape}. "
                f"Build a matching response with "
                f"timepix_response.get_response(E_grid, ...)."
            )
        n_in = np.bincount(
            self.idx_in, weights=spec * self.dE_fine, minlength=self.n_in
        )  # photons / coarse bin
        S_out_coarse = (self.R @ n_in) / self.dE_out  # detected density / eV
        return np.interp(self.E, self.E_out, S_out_coarse, left=0.0, right=0.0)

    def detection_efficiency(self, E_eV=None):
        """P_det interpolated onto E_eV (default: the fine grid). This is the
        s-curve x charge-sharing turn-on -- ~0 below threshold, rising through
        ~2-4 keV, plateauing near eps_abs above ~5 keV."""
        E = self.E if E_eV is None else np.asarray(E_eV, dtype=float)
        return np.interp(E, self.E_in, self.P_det)


# Responses are pure functions of (energy grid, hardware, MC settings), so cache
# them: a results set that mixes grids (e.g. a stale checkpoint with both 2500
# and 3000 eV runs) then transparently gets one response per distinct grid.
_RESPONSE_CACHE = {}


def get_response(
    E_grid_eV,
    *,
    dE_mc=50.0,
    dE_out=25.0,
    n_mc=60000,
    seed=0,
    thickness_um=None,
    bias_v=None,
):
    """
    Cached TimepixResponse for a given energy grid + settings: built once per
    unique (grid, hardware, MC) signature and reused thereafter. Prefer this over
    constructing TimepixResponse directly when looping over many spectra -- every
    spectrum gets a response matching ITS grid, instead of one shared matrix that
    only fits the first grid encountered.
    """
    E = np.asarray(E_grid_eV, dtype=float)
    thick = SENSOR_THICKNESS_UM if thickness_um is None else thickness_um
    bias = BIAS_VOLTAGE_V if bias_v is None else bias_v
    # grid identity = (size, endpoints) since the grids are uniform; plus the
    # hardware/MC settings that change the matrix
    key = (
        E.size,
        round(float(E[0]), 6),
        round(float(E[-1]), 6),
        dE_mc,
        dE_out,
        n_mc,
        seed,
        thick,
        bias,
    )
    resp = _RESPONSE_CACHE.get(key)
    if resp is None:
        resp = TimepixResponse(
            E,
            dE_mc=dE_mc,
            dE_out=dE_out,
            n_mc=n_mc,
            seed=seed,
            thickness_um=thickness_um,
            bias_v=bias_v,
        )
        _RESPONSE_CACHE[key] = resp
    return resp


def poisson_counts(E_grid_eV, detected_per_s, time_s, rng=None):
    """
    Draw a Poisson realization of one acquisition -- what the detector actually
    records, statistical noise and all.

    The expected counts in a bin are rate * bin_width * live_time, where the rate
    is the detected spectral density already scaled to absolute units:
        detected_per_s : detected density [Phs/eV/s] on E_grid_eV, i.e. the
                         output of TimepixResponse.apply() multiplied by the beam
                         current (and any per-nA factor).
    Returns (counts_per_bin, expected_per_bin): the integer Poisson draw and its
    mean, per energy bin -- plot counts as the 'measurement', expected as the
    smooth truth."""
    rng = np.random.default_rng() if rng is None else rng
    E = np.asarray(E_grid_eV, dtype=float)
    dE = E[1] - E[0]
    expected = np.asarray(detected_per_s, dtype=float) * dE * time_s
    expected = np.clip(expected, 0.0, None)  # guard tiny negative noise
    return rng.poisson(expected), expected
