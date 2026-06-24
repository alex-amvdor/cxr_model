---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python (sci)
  language: python
  name: sci
---

# Parametric X-ray Radiation (PXR)

# and Coherent Bremsstrahlung Radiation (CBR)

The coordinate system used in the following derivation is defined as in the following figure:

![{5AE41139-D2AD-4701-A1B2-D9718DD67A1F}.png](attachment:{5AE41139-D2AD-4701-A1B2-D9718DD67A1F}.png)

+++

According to Kleiner et al. (1994), the radiated electric field due to a fast free electron
incident on a periodic lattice in the far-field (assuming a small electron scattering angle,
and working with simplified units $\hbar=c=1$) can be derived from

$$
\nabla \times \nabla \times \mathbf{E}_\omega
- \omega^2\varepsilon(\omega,\,r)\,\mathbf{E}_\omega
= 2 i \omega q\int \mathrm{d}t \, \mathbf{V}_q \, \delta(\mathbf{r} - \mathbf{r}_q)e^{i \omega t}
$$

where the dielectric function is

$$
\varepsilon(\omega, \, r) = 1 + \varkappa_0(\omega) + \sum_g ' \varkappa_g (\omega)e^{i\mathbf{gr}}
$$

with $\mathbf{g}$ being the inverse crystal lattice vector and $\varkappa_0(\omega)=-\omega_0^2/\omega^2$ for the X-ray emissions
($\omega_0$ being the crystal plasma frequency).

For a mono-atom crystal, we have

$$
\varkappa_g = \varkappa_0\frac{F(g)}{Z}e^{-g^2u_T^2} \equiv -\omega_g^2/\omega^2
$$

Thus, $\omega_g^2 = \omega_0^2 \frac{F(g)}{Z}e^{-g^2u_T^2}$, where $F(g)$ is the atomic form factor, $Z$ is the kern charge and $e^{-g^2u_T^2}$ is the Debye-Waller Factor at temperature $T$.

The far-field is then approximately given by

$$
\begin{aligned}
E_{\lambda}^{(g)}
  &= \left( \frac{i\omega q}{2} \right)
     \sum_{l=1}^{2}
     \exp\!\left[
         i \xi_{\lambda g}^{(l)} \mathbf{r}
       - i \left( k_0 + \xi_{\lambda g}^{(l)} \right) \mathbf{n} \mathbf{r}_0
     \right]
     \Bigg\{
       \left[
         1 - (-1)^l \left( \Delta_g / \varkappa_{\lambda g} \right)
       \right]
  \\[4pt]
  &\quad\times
     \sum_{\mathbf{g}'}
     \left(
         \mathbf{e}_{\lambda g} \boldsymbol{\nu}_{\mathbf{g}'}
       - i k_0\, \mathbf{e}_{\lambda g} \boldsymbol{\nu}_0\, \mathbf{n} \mathbf{r}_{\mathbf{g}'}
     \right)
     \exp\!\left[ i \mathbf{g}' \mathbf{r}_0 \right]
     \,\delta\!\left[
         \omega
       - \left( k_0 + \xi_{\lambda g}^{(l)} \right) \mathbf{n} \boldsymbol{\nu}_0
       + \mathbf{g}' \boldsymbol{\nu}_0
     \right]
     + (-1)^l
  \\[4pt]
  &\quad\times
     \left( \frac{\omega_g^{2} \alpha_{\lambda}}{\omega \varkappa_{\lambda g}} \right)
     \mathbf{e}_{\lambda 0} \boldsymbol{\nu}_0
     \exp\!\left[ i \mathbf{g} \mathbf{r}_0 \right]
     \,\delta\!\left[
         \omega
       - \left( k_0 + \xi_{\lambda g}^{(l)} \right) \mathbf{n} \boldsymbol{\nu}_0
       + \mathbf{g} \boldsymbol{\nu}_0
     \right]
     \Bigg\}
     \exp\!\left[ i k_0 r \right] / r
\end{aligned}
$$

where $\lambda \in \{1,\,2\}$ signifies the across-reaction-plane and in-reaction-plane polarizations, respectively, while $e_{\lambda \{0,g\}}$ signifies the polarization states of the diffracted and non-diffracted waves, respectively.

Our various wavevectors are given by

$$
k_0 = |\mathbf{k}| = \frac{\omega \sqrt{\varepsilon_0}}{c},\qquad \mathbf{k}_g = \mathbf{k} + \mathbf{g}, \qquad |\mathbf{k}_g| = k_0 + \xi_{\lambda g}^{(l)}
$$

where $\mathbf{k}_g$ is the diffracted photon wavevector, $\mathbf{k}$ is the non-diffracted photon wavevector, and $\Delta_g$ can be interpreted as the magnitude of the detuning from the Bragg resonance.

The constants $\alpha_{\lambda}$ are given by

$$
\alpha_1 = 1, \qquad \alpha_2 = \frac{\mathbf{k}\cdot\mathbf{k}_g}{|\mathbf{k}||\mathbf{k_g}|} = \cos\theta_g
$$

Further constant relations are given by

$$

\Delta_g = -\mathbf{\hat{n} \cdot g} + g^2/2k_0, \qquad \xi_{\lambda g}^{(l)} = -\frac{\Delta_g + (-1)^l\varkappa_{\lambda g}}{2(1-\mathbf{\hat{n} \cdot g}/k_0)}, \qquad \varkappa_{\lambda g} = \sqrt{\Delta_g^2 + \frac{\omega_g^4 \alpha_\lambda^2(1 - \mathbf{\hat{n} \cdot g}/k_0)}{\omega^2}}\\


$$

Assuming small detunings from $\theta_B$ we have

$$
\mathbf{k}_g = |\mathbf{k}_g|\mathbf{\hat{n}} \approx k_0 \mathbf{\hat{n}}
$$

where $\mathbf{\hat{n}}$ is the unit vector along the diffracted field Poynting vector.

We then have

$$
\alpha_1 = 1,\qquad \alpha_2 \approx \cos\theta_g
$$

The above expressions are valid in the small detuning regime, $\Delta_g \ll k_0$, i.e., small detunings from the Bragg angle.

## PXR Photon Energy (Kinematic)

The PXR reflex in $E_\lambda^{(g)}$ is the term carrying the lattice vector $\mathbf{g}$ (the $\omega_g^2$ contribution);
the $\mathbf{g}'$ sum describes coherent bremsstrahlung (CBR), which we set aside for the moment. The emitted PXR angular
frequency for a given lattice vector $\mathbf{g}$, Bragg angle $\theta_B$, and observation angle $\Omega$ is fixed
by the argument of the delta function:

$$

\delta\!\left[
\omega
- \left( k_0 + \xi_{g}^{(l)} \right) \mathbf{n} \cdot \boldsymbol{\nu}_0
+ \mathbf{g} \cdot \boldsymbol{\nu}_0 \right]

\;\Longrightarrow\;

\omega =
\left[\left( k_0 + \xi_{g}^{(l)} \right) \mathbf{n} - \mathbf{g}\right]
\cdot \boldsymbol{\nu}_0
$$

Where $\xi_{g}^{(l)}$ is a dynamical correction term due to coupling with the lattice, and $\boldsymbol{\nu}_0$ is the electron velocity. Assuming (for the moment) that the coupling is weak, and/or that the X-ray absorption length in the crystal is much shorter than the extinction length (i.e., making the kinematic assumption), $\xi_{g}^{(l)}$ is then negligible, and the conservation law becomes

$$
\omega = (k_0 \mathbf{\hat{n}} - \mathbf{g}) \cdot \boldsymbol{\nu}_0
$$

**Geometry.** Defining the reflex plane (the plane formed by $\boldsymbol{\nu}_0$ and $\mathbf{g}$) to be the $x$–$z$ plane,
with the electron beam in the $+\hat{z}$ direction:

$$
\begin{aligned}
  \boldsymbol{\nu}_0
    &= \beta c\, \hat{z} \\

  \mathbf{\hat{n}}
    &= \sin\Omega\, \hat{x} + \cos\Omega\, \hat{z} \\

  \mathbf{g}
    &= g \left(
          \cos\theta_B\, \hat{x}
        - \sin\theta_B\, \hat{z}
      \right)
\end{aligned}
$$

**Inner products.**

$$
k_0 \mathbf{\hat{n}} \cdot \boldsymbol{\nu}_0
  = \frac{\omega \sqrt{\varepsilon_0}}{c}\, \beta c \cos\Omega
  = \omega \sqrt{\varepsilon_0}\, \beta \cos\Omega
$$

$$
\mathbf{g} \cdot \boldsymbol{\nu}_0
  = - g\, \beta c \sin\theta_B
$$

**Solve for $\omega$.** Substituting into $\omega = \mathbf{k}\cdot\boldsymbol{\nu}_0 - \mathbf{g}\cdot\boldsymbol{\nu}_0$,

$$
\omega
  = \omega \sqrt{\varepsilon_0}\, \beta \cos\Omega
  + g\, \beta c \sin\theta_B .
$$

Collecting the factors of $\omega$ and using $g = 2\pi / d_{hkl}$,

$$
\omega \left(1 - \sqrt{\varepsilon_0}\, \beta \cos\Omega \right)
  = \frac{2\pi}{d_{hkl}}\, \beta c \sin\theta_B .
$$

So, for the PXR photon energy $\hbar\omega$,

$$
\boxed{\;
\hbar\omega
  = \frac{2\pi \hbar c}{d_{hkl}}\,
    \frac{\beta \sin\theta_B}{\,1 - \sqrt{\varepsilon_0}\, \beta \cos\Omega\,}
\;}
$$

## PXR Photon Energy (Dynamical)

Under the full dynamical framework, $\xi_g^{(l)}$ cannot be neglected. This is the case when the following conditions are met:
1) X-ray absorption length within the crystal, $L_\mathrm{abs}$, is similar to or longer than the
X-ray extinction length, $L_\mathrm{ext}$
2) The actual X-ray path length in the crystal is similar to or longer than $L_\mathrm{ext}$, such that multiple sequential reflections/diffractions can take place within the crystal.
3) The emitted photon must lie within roughly the Darwin width of a Bragg condition, $|\Delta_g| \lesssim \varkappa_{\lambda g} \sim \omega|\chi_g|/c$. For nonrelativistic beam energies the detuning is of order $g$ itself, so no crystal thickness brings the line into the dynamical regime; see kinematic_validity_check.py.

For completeness, however, we calculate the expected dynamical results, to compare to that given by the Kinematic approximation.

Restating from above, we have

$$
\xi_{\lambda g}^{(l)} = -\frac{\Delta_g + (-1)^l\varkappa_{\lambda g}}{2(1-\mathbf{\hat{n} \cdot g}/k_0)},
\qquad \Delta_g = -\mathbf{\hat{n} \cdot g} + g^2/2k_0
$$

The inner product $\mathbf{\hat{n} \cdot g}$ is

$$
\begin{aligned}
\mathbf{\hat{n} \cdot g}
&= |\mathbf{g}|(\sin\Omega\cos\theta_B - \cos\Omega\sin\theta_B) \\
&= |\mathbf{g}|\sin(\Omega - \theta_B)
\end{aligned}
$$

so, we have

$$
\Delta_g = -|\mathbf{g}|\sin(\Omega - \theta_B) + g^2/2k_0
$$

and, thus, our dynamical correction factor:

$$
\xi_{\lambda g}^{(l)}
  = \frac{
    |\mathbf{g}|\sin(\Omega - \theta_B) - g^2/2k_0 - (-1)^l\varkappa_{\lambda g}
    }{
      2\left(1 - \frac{
        |\mathbf{g}|\sin(\Omega - \theta_B)}
        {k_0}
      \right)
    },
$$

For the full dynamical form, we have $\omega = [(k_0 + \xi_{\lambda g}^{(l)})\mathbf{\hat{n}} - \mathbf{g}] \cdot \boldsymbol{\nu}_0$.
So, plugging in $\xi_{\lambda g}^{(l)}$ to the first dot product, we have

$$
(k_0 + \xi_{\lambda g}^{(l)})\mathbf{\hat{n}} \cdot \boldsymbol{\nu}_0 
  = \left(
    \frac{\omega \sqrt{\varepsilon_0}}{c}
    + \xi_{\lambda g}^{(l)}
  \right) \beta c \cos\Omega
$$

while the second dot product remains

$$
\mathbf{g} \cdot \boldsymbol{\nu}_0
  = - g\, \beta c \sin\theta_B
$$

and, thus

$$
\begin{aligned}
\omega_{\lambda g}^{(l)}
  &= \left[(k_0 + \xi_{\lambda g}^{(l)})\mathbf{\hat{n}} - \mathbf{g}\right] \cdot \boldsymbol{\nu}_0 \\

  &= \beta c \left[\left(
    \frac{\omega_{\lambda g}^{(l)} \sqrt{\varepsilon_0}}{c}
    + \xi_{\lambda g}^{(l)} \right) \cos\Omega + g \sin\theta_B\right] \\
\end{aligned}
$$

Solving for $\omega_{\lambda g}^{(l)}$

$$
\begin{aligned}
\frac{\omega_{\lambda g}^{(l)}}{\beta c \cos\Omega} - \frac{\omega_{\lambda g}^{(l)} \sqrt{\varepsilon_0}}{c}
  &= \xi_{\lambda g}^{(l)} + \frac{g \sin\theta_B}{\cos\Omega} \\

\omega_{\lambda g}^{(l)} \left(\frac{1 - \beta \cos\Omega\sqrt{\varepsilon_0}}{\beta c} \right)
  &= \xi_{\lambda g}^{(l)}\cos\Omega + g \sin\theta_B \\
\end{aligned}
$$

Our final, full dynamical energy terms are then given by

$$
\omega_{\lambda g}^{(l)}
  = \frac{\xi_{\lambda g}^{(l)}\cos\Omega + g \sin\theta_B}
  {1 - \beta\cos\Omega\sqrt{\varepsilon_0}} \beta c
$$

Where the dynamical correction factor $\xi_{\lambda g}^{(l)}$ is as defined above, yielding two (slightly split)
lines per branch $l \in \{1,\, 2\}$, per polarization $\lambda \in \{1,\, 2\}$, for each individual reciprocal
lattice vector $\mathbf{g}$ in the crystal. In our regime, the linewidths should be sufficiently broad
so as to make the two lines indistinguishable.

NOTE: The R.H.S implicitly contains $\omega_{\lambda g}^{(l)}$ via the $k_0$ terms within $\xi_{\lambda g}^{(l)}$.
The correction term $\xi_{\lambda g}^{(l)}$ is generally small compared to $k_0$ and $\mathbf{g}$
(especially so in our energy regime), so it should generally be acceptable to evaluate
$\xi_{\lambda g}^{(l)}$ at the approximate kinematic result for $\omega$.

## PXR Photon Flux

Assuming we examine small displacements about the Bragg angle



| Symbol (code)                                         | Meaning                                                                                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `E` $=\hbar\omega$                                    | emitted photon energy (keV)                                                                                                     |
| `omega` $=\Omega$                                     | photon emission angle from the beam (Bragg: $\Omega = 2\theta_B$)                                                               |
| `theta` $=\theta_B$                                   | Bragg angle, between beam and diffracting planes                                                                                |
| `d` $=d_{hkl}$                                        | $(hkl)$-oriented interplanar crystalline spacing                                                                                |
| `beta` $=\beta=v/c$                                   | electron speed, normalized ($0.33$–$0.45$ for 30–60 keV)                                                                        |
| `h_c` $=hc$                                           | $1.2398$ keV·nm                                                                                                                 |
| `eps_0` $=\varepsilon_0\approx 1-(\omega_0/\omega)^2$ | mean crystal dielectric constant; $\sqrt{\varepsilon_0}\approx 1$ for X-rays (rendered as 1 in code, contributes &lt;0.1% here) |

```{code-cell} ipython3
# Core library modules live in src/; put it on the import path. Run from repo root.
import sys

sys.path.insert(0, "src")
```

```{code-cell} ipython3
# Reflections to study. Structure data (lattice, basis -> d-spacing, elements)
# comes from crystal_structures.toml via CRYSTALS; only the reflection choice
# and the plot label live here.
import numpy as np
import time
from scipy import constants as const
from cxr_model.crystallography import CRYSTALS, reciprocal_g_vector

reflections = {
    # NOTE: MoSe2 previously used a d = 6.0 A placeholder with unknown indices;
    # (002) is the closest layer reflection, d = c/2 = 6.46 A.
    "mose2": {"label": r"$\mathrm{MoSe_2}$", "hkl": (0, 0, 2)},
    "graphite": {"label": r"$\mathrm{Graphite}$", "hkl": (1, 1, 0)},
    "silicon": {"label": r"$\mathrm{Silicon}$", "hkl": (4, 4, 0)},
    "diamond": {"label": r"$\mathrm{Diamond}$", "hkl": (4, 0, 0)},
}

crystals = {}
for key, sel in reflections.items():
    info = CRYSTALS[key]
    _, g = reciprocal_g_vector(sel["hkl"], info["lattice"])
    crystals[sel["label"]] = {
        "elements": sorted({el for el, _ in info["basis"]}),
        "d": (2 * np.pi / g) * 1e-10,  # interplanar spacing [m]
        "miller_indices": "".join(map(str, sel["hkl"])),
    }
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from cxr_model.atomic_form_factors import atomic_form_factor
from cxr_model.crystallography import (
    beta_from_Ee,
    absorption_length_ang,
    reciprocal_g_vector,
    Z_TABLE,
    ALPHA_FS,
    HC_EV_ANG,
    CRYSTALS,
)
from feranchuk_spence import (
    omega_n,
    amplitudes_PXR_CBS_sweep,
    delta_g,
    cxr_to_bremsstrahlung,
)

%matplotlib inline


def plot_photon_energy(result, title):
    plt.figure(figsize=(8, 6))
    betas = result[:, 0]
    for b in np.unique(betas):
        mask = betas == b
        Omega_array = result[mask, 1]
        photon_energies = result[mask, 2]
        E_elec = result[mask, 3][0]
        i_max = np.argmax(photon_energies)
        label = (
            rf"$\beta$={b:.2f},"
            rf" $E_{{\mathrm{{elec}}}}$={E_elec:.1f} keV,"
            rf" $E_{{\gamma,\mathrm{{max}}}}$={photon_energies[i_max]:.2f} keV"
            rf" at {Omega_array[i_max] * 180 / np.pi:.0f}°"
        )
        plt.plot(Omega_array * 180 / np.pi, photon_energies, label=label)
    plt.title(title)
    plt.xlabel(r"$\Omega$ (degrees)", fontsize=12)
    plt.ylabel("Energy (keV)", fontsize=12)
    plt.xticks(np.linspace(0, 180, 9, endpoint=True))
    plt.grid(True, axis="both")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1))
    plt.show()


def calculate_photon_energy(beta, theta_B, Omega, d, h_c, eps_0):
    photon_energies = (
        (h_c / d)
        * beta
        * np.sin(theta_B)
        / (1 - eps_0 ** (1 / 2) * beta * np.cos(Omega))
    )

    return photon_energies


CAL = 1.0


def flux_and_ratio_vs_Omega(
    crystal,
    hkl,
    beta,
    Omega_array,
    L_z_ang,
    current_A,
    dOmega_sr,
    B_ang2=0.0,
    use_henke=False,
    eps0=1.0,
):
    """
    Returns (E_keV, flux_per_s, pxr_frac) arrays over Omega for one beta.
      flux_per_s : calibrated total coherent flux [photons/s]
      pxr_frac   : |A_PXR|^2 / (|A_PXR|^2 + |A_CBS|^2)  (intensity fraction)
    Symmetric reflex: Omega = 2*theta_Bragg, theta_B_normal = 90deg - theta_Bragg.
    Fully vectorized over Omega (amplitudes_PXR_CBS_sweep); energies outside
    the Henke tables come back as NaN and simply leave gaps in the plots.
    """
    info = CRYSTALS[crystal]
    _, g = reciprocal_g_vector(hkl, info["lattice"])
    d = 2 * np.pi / g

    theta_bragg = Omega_array / 2.0
    theta_B_normal = 0.5 * np.pi - theta_bragg

    # line energy at each geometry (the boxed dispersion above)
    E = (
        (HC_EV_ANG / d)
        * beta
        * np.sin(theta_bragg)
        / (1.0 - np.sqrt(eps0) * beta * np.cos(Omega_array))
    )
    E_keV = E / 1e3
    valid = np.isfinite(E) & (E > 0)

    # absorption length at each line energy (single dominant absorber;
    # crude for compounds, refine if needed)
    el0 = info["basis"][0][0]
    n_atoms = len(info["basis"]) / info["V_cell"]
    L_abs = absorption_length_ang(el0, np.where(valid, E, np.nan), n_atoms)

    # full amplitudes, both polarizations, all angles at once
    amps, omega, _ = amplitudes_PXR_CBS_sweep(
        crystal,
        hkl,
        E,
        beta,
        theta_B_normal,
        B_ang2,
        use_henke,
        geometry="symmetric",
    )
    (Aps, Acs), (App, Acp) = amps["sigma"], amps["pi"]

    A2 = np.abs(Aps + Acs) ** 2 + np.abs(App + Acp) ** 2
    L_eff = L_abs * (1.0 - np.exp(-L_z_ang / L_abs))
    e_charge = 1.602176634e-19
    n_e_per_s = current_A / e_charge
    flux = (
        CAL
        * n_e_per_s
        * ALPHA_FS
        / (2 * np.pi)
        * omega
        * (L_eff / beta)
        * A2
        * dOmega_sr
    )

    # PXR intensity fraction (sum polarizations)
    pxr_I = np.abs(Aps) ** 2 + np.abs(App) ** 2
    cbs_I = np.abs(Acs) ** 2 + np.abs(Acp) ** 2
    tot_I = pxr_I + cbs_I
    pxr_frac = np.where(tot_I > 0, pxr_I / np.where(tot_I > 0, tot_I, 1.0), np.nan)

    flux = np.where(valid, flux, np.nan)
    pxr_frac = np.where(valid, pxr_frac, np.nan)
    return E_keV, flux, pxr_frac
```

```{code-cell} ipython3
# ============================================================================
# per-crystal config: (hkl, Debye-Waller B [Ang^2], use_henke)
# use_henke=True for elements with edges in 1-4 keV (Si, Ge, Mo, Se)
# ============================================================================
h_c = const.Planck * const.c / const.elementary_charge  # Convert J to eV
eps_0 = 1
electron_energies = np.array([30e3, 40e3, 50e3, 60e3])  # eV
beta_values = beta_from_Ee(electron_energies)
gamma = 1 / np.sqrt(1 - beta_values**2)

configs = {
    "graphite": dict(hkl=[1, 1, 0], B_ang2=0.25, use_henke=True),
    "silicon": dict(hkl=[4, 4, 0], B_ang2=0.46, use_henke=True),
    "diamond": dict(hkl=[4, 0, 0], B_ang2=0.21, use_henke=True),
    # "mose2":  dict(hkl=[0, 0, 2], B_ang2=...,  use_henke=True),  # basis is a placeholder, see TOML
}
crystals_to_plot = [
    c for c in configs if c in CRYSTALS and CRYSTALS[c].get("basis") is not None
]

# shared experiment params
L_z_ang = 1000.0  # crystal thickness [Angstroms]
current_A = 5e-9  # beam current [Amps]
dOmega_sr = 0.05  # Detector solid angle [steradians]
Omega = np.linspace(0.01, np.pi - 0.01, 1000)

for crystal in crystals_to_plot:
    cfg = configs[crystal]

    # top row: vs. emission angle | bottom row: parametric in Omega, vs. energy
    fig, ax_E = plt.subplots(1, 1, figsize=(8, 5))
    ax_E.set_title(f"{crystal} — CXR Energy vs. Emission Angle", fontsize=16)

    ax_E.set_xlabel(r"$\Omega = 2\theta_B$ (deg)", fontsize=14)
    ax_E.set_ylabel("Photon energy (keV)", fontsize=14)
    ax_E.set_xticks(np.linspace(0, 180, 9))
    ax_E.grid(True, which="both", alpha=0.3)

    for beta in beta_values:
        t_line = time.perf_counter()
        E_keV, flux, pxr_frac = flux_and_ratio_vs_Omega(
            crystal,
            cfg["hkl"],
            beta,
            Omega,
            L_z_ang,
            current_A,
            dOmega_sr,
            B_ang2=cfg["B_ang2"],
            use_henke=cfg["use_henke"],
        )

        Ee_keV = (1 / np.sqrt(1 - beta**2) - 1) * 510.99891
        lbl = rf"$\beta$={beta:.2f}, $E_e$={Ee_keV:.0f} keV"

        deg = np.degrees(Omega)
        ax_E.plot(deg, E_keV, label=lbl)

    ax_E.legend(fontsize=12, title="Electron energy")
```

```{code-cell} ipython3
# ============================================================================
# per-crystal config: (hkl, Debye-Waller B [Ang^2], use_henke)
# use_henke=True for elements with edges in 1-4 keV (Si, Ge, Mo, Se)
# ============================================================================
h_c = const.Planck * const.c / const.elementary_charge  # Convert J to eV
eps_0 = 1
electron_energies = np.array([15e3, 30e3, 45e3, 60e3, 75e3])  # eV
beta_values = beta_from_Ee(electron_energies)
gamma = 1 / np.sqrt(1 - beta_values**2)

configs = {
    "graphite": dict(hkl=[1, 1, 0], B_ang2=0.25, use_henke=True),
    "silicon": dict(hkl=[4, 4, 0], B_ang2=0.46, use_henke=True),
    "diamond": dict(hkl=[4, 0, 0], B_ang2=0.21, use_henke=True),
    # "mose2":  dict(hkl=[0, 0, 2], B_ang2=...,  use_henke=True),  # basis is a placeholder, see TOML
}
crystals_to_plot = [
    c for c in configs if c in CRYSTALS and CRYSTALS[c].get("basis") is not None
]

# shared experiment params
L_z_ang = 1000.0  # crystal thickness [Angstroms]
current_A = 5e-9  # beam current [Amps]
dOmega_sr = 0.05  # Detector solid angle [steradians]
Omega = np.linspace(0.01, np.pi - 0.01, 1000)

t_all = time.perf_counter()
for crystal in crystals_to_plot:
    cfg = configs[crystal]
    t_crystal = time.perf_counter()

    # top row: vs. emission angle | bottom row: parametric in Omega, vs. energy
    fig, ((ax_E, ax_flux, ax_frac), (ax_fluxE, ax_fracE, ax_leg)) = plt.subplots(
        2, 3, figsize=(20, 10)
    )
    fig.suptitle(f"{crystal} — PXR vs. CBS", fontsize=20)

    for ax in (ax_E, ax_flux, ax_frac):
        ax.set_xlabel(r"$\Omega = 2\theta_B$ (deg)", fontsize=16)
        ax.set_xticks(np.linspace(0, 180, 9))
        ax.grid(True, which="both", alpha=0.3)
    for ax in (ax_fluxE, ax_fracE):
        ax.set_xlabel("Photon energy (keV)", fontsize=16)
        ax.grid(True, which="both", alpha=0.3)

    ax_E.set_ylabel("Photon energy (keV)", fontsize=16)
    ax_E.set_title("Line energy", fontsize=18)

    ax_flux.set_yscale("log")
    ax_flux.set_ylabel("Coherent flux (photons/s)", fontsize=16)
    ax_flux.set_title("Total CXR flux (over 4π steradians)", fontsize=18)

    ax_frac.set_ylabel("PXR fraction", fontsize=16)
    ax_frac.set_ylim(0, 1)
    ax_frac.set_title("PXR/CBS balance", fontsize=18)

    ax_fluxE.set_yscale("log")
    ax_fluxE.set_ylabel("Coherent flux (photons/s)", fontsize=16)
    ax_fluxE.set_title("Total CXR flux vs. line energy (Bragg condition)", fontsize=18)

    ax_fracE.set_ylabel("PXR fraction", fontsize=16)
    ax_fracE.set_ylim(0, 1)
    ax_fracE.set_title("PXR/CXR ratio vs. line energy (Bragg condition)", fontsize=18)

    ax_leg.axis("off")  # shared legend lives here
    fig.tight_layout()

    t_compute = 0.0
    for beta in beta_values:
        t_line = time.perf_counter()
        E_keV, flux, pxr_frac = flux_and_ratio_vs_Omega(
            crystal,
            cfg["hkl"],
            beta,
            Omega,
            L_z_ang,
            current_A,
            dOmega_sr,
            B_ang2=cfg["B_ang2"],
            use_henke=cfg["use_henke"],
        )
        t_compute += time.perf_counter() - t_line

        Ee_keV = (1 / np.sqrt(1 - beta**2) - 1) * 510.99891
        lbl = rf"$\beta$={beta:.2f}, $E_e$={Ee_keV:.0f} keV"

        deg = np.degrees(Omega)
        ax_E.plot(deg, E_keV, label=lbl)
        ax_flux.plot(deg, flux)
        ax_frac.plot(deg, pxr_frac)
        # parametric in Omega: each point is the Bragg-condition (E, value) pair
        ax_fluxE.plot(E_keV, flux)
        ax_fracE.plot(E_keV, pxr_frac)

    handles, labels = ax_E.get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center", fontsize=16, title="Electron energy")
    plt.show()  # figure appears as soon as this crystal is done
    print(
        f"{crystal}: compute {t_compute * 1e3:.0f} ms, "
        f"total {time.perf_counter() - t_crystal:.2f} s (incl. render)"
    )

print(f"all crystals: {time.perf_counter() - t_all:.2f} s")
```

[1] B. L. Henke, E. M. Gullikson, and J. C. Davis, "X-ray interactions: photoabsorption, scattering, transmission, and reflection at E = 50–30000 eV, Z = 1–92," Atomic Data and Nuclear Data Tables 54, 181–342 (1993). Data retrieved from the Center for X-Ray Optics, Lawrence Berkeley National Laboratory, https://henke.lbl.gov/optical_constants/ (accessed 2026).

[2] Cromer–Mann four-Gaussian atomic form factor coefficients from International Tables for Crystallography, Vol. C, ed. E. Prince (Wiley, 2004), Ch. 6.1. Coefficients and calculator: P. Hadley, "Atomic form factors," Graz University of Technology, https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php (accessed 2026).

```{code-cell} ipython3
# ============================================================================
# cross-crystal overlay at a fixed electron energy: which crystal reaches a
# given photon energy with the most flux, and how PXR-dominated is it there?
# (each curve is parametric in Omega, Bragg condition assumed)
# ============================================================================
for Ee_overlay_eV in [30e3, 45e3, 60e3]:
    beta_ov = beta_from_Ee(Ee_overlay_eV)

    fig, (ax_flux, ax_frac) = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(
        rf"Cross-crystal comparison at $E_e$ = {Ee_overlay_eV / 1e3:.0f} keV "
        rf"($\beta$ = {beta_ov:.2f})",
        fontsize=15,
    )

    for crystal in crystals_to_plot:
        cfg = configs[crystal]
        E_keV, flux, pxr_frac = flux_and_ratio_vs_Omega(
            crystal,
            cfg["hkl"],
            beta_ov,
            Omega,
            L_z_ang,
            current_A,
            dOmega_sr,
            B_ang2=cfg["B_ang2"],
            use_henke=cfg["use_henke"],
        )
        lbl = f"{crystal} ({''.join(map(str, cfg['hkl']))})"
        ax_flux.plot(E_keV, flux, label=lbl)
        ax_frac.plot(E_keV, pxr_frac, label=lbl)

    for ax in (ax_flux, ax_frac):
        ax.set_xlabel("Photon energy (keV)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    ax_flux.set_yscale("log")
    ax_flux.set_ylabel("Coherent flux (photons/s)")
    ax_flux.set_title("Total CXR flux vs. line energy")
    ax_frac.set_ylabel("PXR fraction")
    ax_frac.set_ylim(0, 1)
    ax_frac.set_title("PXR/CXR ratio vs. line energy")
    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
# ============================================================================
# FULL energy spectrum through the detector aperture, crystal FIXED
# ----------------------------------------------------------------------------
# Crystal fixed at theta_Bragg = 22.5 deg for the configured reflection (so
# Omega = 45 deg is its specular direction); all reciprocal vectors with
# g.v0 > 0 contribute a line (cxr_lines_fixed, full Eq. 13/14 amplitudes).
#
# READING THIS FIGURE: the dense forest of weak high-g lines is NOT
# detectable peak structure -- it is the model's coherent decomposition of
# (part of) the bremsstrahlung continuum. Its binned level is cutoff-
# dependent (grows ~logarithmically with g_max_invang: the transverse-g sum
# is the Coulomb logarithm of bremsstrahlung, physically regulated only at
# g ~ electron momentum ~130 1/A) and sits at the incoherent-BS background
# of Eq. (17) (dashed line). Trustworthy, detectable peaks are the lines
# rising ABOVE that background (Eq. 18 criterion, eta > 1) -- highlighted
# as stems. The same CAL prefactor is applied to lines and background so the
# ratio is meaningful.
# Bins ~ detector resolution; each line's intrinsic + aperture width
# (<~50 eV) is at most about one bin. The azimuthal setting about the
# configured g is the minimal-rotation convention (out-of-plane line
# positions depend on it).
# ============================================================================
from cxr_model.crystallography import Z_TABLE
from feranchuk_spence import cxr_lines_fixed, bremsstrahlung_background

Ee_spec_eV = 120e3
beta_sp = beta_from_Ee(Ee_spec_eV)
theta_bragg_fix = np.deg2rad(22.5)
theta_B_fix = np.pi / 2 - theta_bragg_fix  # angle of g from beam (to plane normal)
Omega_aps = np.deg2rad([5, 45, 105])

bin_eV = 50.0
E_min_eV, E_max_eV = 1000.0, 10000.0
bin_edges_eV = np.arange(E_min_eV, E_max_eV + bin_eV, bin_eV)
bin_ctr_eV = 0.5 * (bin_edges_eV[1:] + bin_edges_eV[:-1])
n_e_per_s = current_A / 1.602176634e-19

fig, axes = plt.subplots(len(crystals_to_plot), 1, figsize=(14, 11), sharex=True)
fig.suptitle(
    rf"CXR lines vs. bremsstrahlung background in the aperture "
    rf"($d\Omega$={dOmega_sr:g} sr, {bin_eV:.0f} eV bins), crystal fixed at "
    rf"$\theta_B$={np.degrees(theta_bragg_fix):.1f}°, "
    rf"$E_e$={Ee_spec_eV / 1e3:.0f} keV",
    fontsize=14,
)

for ax, crystal in zip(np.atleast_1d(axes), crystals_to_plot):
    cfg = configs[crystal]
    info = CRYSTALS[crystal]
    el0 = info["basis"][0][0]
    n_atoms = len(info["basis"]) / info["V_cell"]

    # incoherent BS background per bin through the aperture (Eq. 17)
    bs_bin = (
        CAL
        * n_e_per_s
        * dOmega_sr
        * bremsstrahlung_background(bin_ctr_eV, Z_TABLE[el0], n_atoms, L_z_ang, bin_eV)
    )
    ax.plot(
        bin_ctr_eV / 1e3, bs_bin, "k--", lw=1.3, label="incoherent BS per bin (Eq. 17)"
    )

    for i_ap, Om in enumerate(Omega_aps):
        lines = cxr_lines_fixed(
            crystal,
            beta_sp,
            Om,
            cfg["hkl"],
            theta_B_fix,
            E_min_eV=E_min_eV,
            E_max_eV=E_max_eV,
            B_ang2=bool(cfg["B_ang2"]),
            use_henke=bool(cfg["use_henke"]),
        )
        E, A2, omega = lines["E_eV"], lines["A2"], lines["omega"]
        L_abs = absorption_length_ang(el0, E, n_atoms)
        L_eff = L_abs * (1.0 - np.exp(-L_z_ang / L_abs))
        N_line = (
            CAL
            * n_e_per_s
            * ALPHA_FS
            / (2 * np.pi)
            * omega
            * (L_eff / beta_sp)
            * A2
            * dOmega_sr
        )
        ok = np.isfinite(N_line) & (N_line > 0)

        # the full forest, faint: effectively part of the BS background
        ax.hist(
            E[ok] / 1e3,
            bins=bin_edges_eV / 1e3,
            weights=N_line[ok],
            histtype="stepfilled",
            alpha=0.18,
            color=f"C{i_ap}",
            lw=0,
        )

        # detectable peaks: lines that rise above the BS background level
        bs_at = np.interp(E, bin_ctr_eV, bs_bin)
        strong = ok & (N_line > bs_at)
        ax.vlines(
            E[strong] / 1e3,
            bs_at[strong],
            N_line[strong],
            color=f"C{i_ap}",
            lw=1.8,
            alpha=0.95,
        )
        ax.plot(
            E[strong] / 1e3,
            N_line[strong],
            "o",
            ms=3.5,
            color=f"C{i_ap}",
            label=rf"$\Omega$={np.degrees(Om):.0f}°: lines above BS"
            rf" ({strong.sum()})",
        )

        # annotate the strongest detectable lines of the middle aperture
        if i_ap == 1 and strong.any():
            top = np.argsort(N_line * strong)[::-1][:4]
            for i_t in top:
                if not strong[i_t]:
                    continue
                hkl_s = "".join(str(int(x)) for x in lines["hkl"][i_t])
                ax.annotate(
                    f"({hkl_s})",
                    (E[i_t] / 1e3, N_line[i_t]),
                    textcoords="offset points",
                    xytext=(3, 4),
                    fontsize=8,
                    color=f"C{i_ap}",
                )

        n_det = strong.sum()
        print(
            f"{crystal:9s} Omega={np.degrees(Om):4.0f}°: {ok.sum():4d} lines, "
            f"{n_det:3d} above BS; coherent total {np.nansum(N_line[ok]):.2e}, "
            f"BS band total {bs_bin.sum():.2e} photons/s"
        )

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.2 * bs_bin.min())
    ax.set_ylabel("photons/s per bin")
    ax.set_title(
        f"{crystal} (oriented on {''.join(map(str, str(cfg['hkl'])))}); faint fill ="
        " all-g forest (cutoff-dependent, ~BS level)",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

np.atleast_1d(axes)[-1].set_xlabel("Photon energy (keV)")
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# ============================================================================
# Replication of paper Fig. 2 (a),(b): ideal CXR spectra, Si, 120 keV,
# observation angle 96 deg, beam parallel to <111> (a) and <100> (b)
# ----------------------------------------------------------------------------
# "Electron velocity parallel to <uvw>" is set by ORIENTING THE CRYSTAL, not
# the beam: cxr_lines_fixed(beam_uvw=[u,v,w]) rotates the direct-lattice
# direction [uvw] onto the beam axis (+z). Along a zone axis every line
# energy depends only on g_z, so the thousands of allowed reflections
# collapse onto a discrete series E ~ (h+k+l-projection); the azimuth about
# the axis (azimuth_rad) changes individual line intensities at the detector
# but not the series energies.
# Dark fill = PXR part (sum |A_PXR|^2), light bar = total CXR, the analog of
# the paper's filled/full bars. NOTE: the paper's bar heights use their
# angle-independent Eq. (28) estimate; ours use the full Eqs. (13)/(14) at
# this specific detector geometry (96 deg, minimal-rotation azimuth), so
# relative heights can differ -- series positions should match exactly.
# dOmega from Table I (0.05 sr); y-scale uses our beam current + CAL anchor.
# ============================================================================
from feranchuk_spence import cxr_lines_fixed

Ee_f2_eV = 120e3
beta_f2 = beta_from_Ee(Ee_f2_eV)
theta_obs_f2 = np.deg2rad(96.0)
dOmega_f2 = 0.05  # sr, Table I (Si / Reese et al.)
bin_f2_eV = 100.0
E_lo_eV, E_hi_eV = 1500.0, 10500.0
edges_keV = np.arange(E_lo_eV, E_hi_eV + bin_f2_eV, bin_f2_eV) / 1e3

crystal = "silicon"
info = CRYSTALS[crystal]
n_atoms = len(info["basis"]) / info["V_cell"]
B_si = configs[crystal]["B_ang2"]
n_e_per_s = current_A / 1.602176634e-19

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 5.0))
fig.suptitle(
    rf"Paper Fig. 2 (a),(b) analog: Si, $E_e$={Ee_f2_eV / 1e3:.0f} keV, "
    rf"$\theta_0$={np.degrees(theta_obs_f2):.0f}°, {bin_f2_eV:.0f} eV bins, "
    rf"$d\Omega$={dOmega_f2:g} sr",
    fontsize=14,
)

for ax, uvw, yscale, panel in (
    (ax_a, [1, 1, 1], "log", "a"),
    (ax_b, [1, 0, 0], "linear", "b"),
):
    lines = cxr_lines_fixed(
        crystal,
        beta_f2,
        theta_obs_f2,
        beam_uvw=uvw,
        E_min_eV=E_lo_eV,
        E_max_eV=E_hi_eV,
        B_ang2=B_si,
        use_henke=True,
    )
    E, om = lines["E_eV"], lines["omega"]
    L_abs = absorption_length_ang("Si", E, n_atoms)
    L_eff = L_abs * (1.0 - np.exp(-L_z_ang / L_abs))
    pref = CAL * n_e_per_s * ALPHA_FS / (2 * np.pi) * om * (L_eff / beta_f2) * dOmega_f2
    N_tot = pref * lines["A2"]
    N_pxr = pref * lines["A2_PXR"]
    ok = np.isfinite(N_tot) & (N_tot > 0)

    ax.hist(
        E[ok] / 1e3,
        bins=edges_keV,
        weights=N_tot[ok],
        histtype="stepfilled",
        color="0.85",
        edgecolor="0.25",
        lw=1.0,
        label="CXR total",
    )
    ax.hist(
        E[ok] / 1e3,
        bins=edges_keV,
        weights=N_pxr[ok],
        histtype="stepfilled",
        color="C0",
        alpha=0.9,
        lw=0,
        label="PXR part",
    )

    ax.set_yscale(yscale)
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel("photons/s per bin")
    uvw_str = "".join(map(str, uvw))
    ax.set_title(
        rf"({panel}) $\mathbf{{v}}_0 \parallel \langle {uvw_str} \rangle$"
        rf" ({yscale} scale)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # console summary: the discrete series and its leading reflection
    E_r = np.round(E[ok], 1)
    print(f"--- v0 || <{uvw_str}>: series peaks ---")
    for e_u in np.unique(E_r):
        m = E_r == e_u
        if N_tot[ok][m].sum() < 1e-6:
            continue
        i_lead = np.flatnonzero(ok)[m][np.argmax(N_tot[ok][m])]
        hkl_s = ",".join(str(int(x)) for x in lines["hkl"][i_lead])
        print(
            f"  E = {e_u / 1e3:6.3f} keV: {m.sum():4d} reflections, "
            f"N = {N_tot[ok][m].sum():.2e} photons/s "
            f"(PXR {100 * N_pxr[ok][m].sum() / N_tot[ok][m].sum():3.0f}%), "
            f"leading ({hkl_s})"
        )

plt.tight_layout()
plt.show()
```
