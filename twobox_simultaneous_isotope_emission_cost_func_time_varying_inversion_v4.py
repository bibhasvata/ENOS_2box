"""
COMPLETE TWO-BOX CH4 INVERSION WITH TIME-VARYING EMISSIONS AND ISOTOPES
========================================================================

Optimises simultaneously:
  - Time-varying emissions E(t) for 5 sources × 2 hemispheres
  - Time-varying source signatures δ¹³C(t) and δD(t)
  - Time-varying sink perturbations (OH, stratosphere, soil)

State vector per year t (36 elements):
  [E_wet_sh … E_waste_nh | δ¹³C_wet_sh … δ¹³C_waste_nh |
   δD_wet_sh … δD_waste_nh | Δτ_OH_sh Δτ_OH_nh Δτ_strat … Δτ_soil_nh]

Cost function:
  J(x) = ‖y − F(x)‖²_Sy  +  ‖x − xₐ‖²_Sa  +  Σ_k λ_k ‖D(x−xₐ)/σ_k‖²

Two-step approach:
  Step 1 – Emissions + sinks only, constrained by CH4 observations.
  Step 2 – All parameters, constrained by CH4 + δ¹³C + δD observations.

Bug-fixes vs. v0
----------------
* **Smoothness Hessian** – `compute_smoothness_penalty` now returns the exact
  quadratic Hessian H_smooth = 2λ DᵀD/σ² for every parameter group, and
  `two_step_inversion` adds it directly to H.  Previously both δ¹³C and δD
  used `smoothness_lambda_isotopes` (= λ_d13c) in the Hessian, and the 1/σ²
  normalisation was missing.  That caused δ¹³C to be under-smoothed (too much
  interannual scatter) and δD to be over-smoothed (only long-term trends).

Author: Bibhasvata Dasgupta
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg, stats
from matplotlib.lines import Line2D

# ============================================================================
# CONFIGURATION
# ============================================================================

class InversionConfig:
    """
    Central configuration namespace for the CH4 two-box inversion.

    Edit the class attributes below to change inversion behaviour.
    Every parameter is documented with its units and physical meaning.
    """

    # --- Time period --------------------------------------------------------
    year1 = 1980   # [yr]  First inversion year (inclusive).
    year2 = 2024   # [yr]  Last  inversion year (inclusive).

    # --- Sources & hemispheres ----------------------------------------------
    sources       = ['wetlands', 'agriculture', 'pyrogenic', 'fossil', 'waste']
    n_sources     = 5
    hemispheres   = ['sh', 'nh']
    n_hemispheres = 2

    # --- Transport model parameters ----------------------------------------
    tau_ihex = 0.75    # [yr]  Interhemispheric exchange timescale.
                       #       Typical observational estimate: 0.7–1.0 yr.
                       #       Smaller → faster NH↔SH mixing.
    tg2ppb   = 0.7178  # [ppb Tg⁻¹]  Mass-to-mixing-ratio conversion.
                       #              1 Tg CH4 produces tg2ppb ppb globally.
    nsteps   = 10      # [-]  Sub-annual integrator steps per year.
                       #      10 is sufficient for annual-mean output.

    # --- Baseline sink lifetimes (prior, hemisphere-resolved) ---------------
    #   Sink parameters in the state vector are expressed as % perturbations
    #   around these fixed baseline values; see prior_sink_error.
    lifetime_oh    = np.array([9.97, 9.97])  # [yr]  OH loss lifetime, [SH, NH].
    lifetime_strat = np.array([181,  181 ])  # [yr]  Stratospheric loss lifetime.
    lifetime_soil  = np.array([188,  188 ])  # [yr]  Soil-uptake lifetime.

    # --- Isotope standard ratios --------------------------------------------
    cfact_d13c = 0.012372    # [-]  ¹³C/¹²C of VPDB standard.
    cfact_dd   = 0.00015576  # [-]  D/H of VSMOW standard.

    # --- Kinetic Isotope Effects (KIE = k_light / k_heavy) -----------------
    #   Values > 1 indicate that the lighter isotopologue reacts faster.
    kie_oh_d13c    = 1.0068  # [-]  ¹³C KIE, OH sink.
    kie_oh_dd      = 1.313   # [-]  D   KIE, OH sink.
    kie_strat_d13c = 1.0144  # [-]  ¹³C KIE, stratospheric sink.
    kie_strat_dd   = 1.133   # [-]  D   KIE, stratospheric sink.
    kie_soil_d13c  = 1.0201  # [-]  ¹³C KIE, soil sink.
    kie_soil_dd    = 1.083   # [-]  D   KIE, soil sink.

    # --- Observation uncertainties (1-σ, diagonal of Sy) -------------------
    obs_sigma_ch4_s0 = 0.1   # [ppb]  CH4 σ in Step 1 (tight → emissions
                              #        strongly constrained before isotopes).
    obs_sigma_ch4    = 1.0   # [ppb]  CH4 σ in Step 2.
    obs_sigma_d13c   = 0.05  # [‰]   δ¹³C σ in Step 2.
                              #        Tighten (e.g. 0.01) to give isotopes
                              #        more weight relative to CH4.
    obs_sigma_dd     = 0.5   # [‰]   δD σ in Step 2.

    # --- Prior uncertainties (1-σ, diagonal of Sa) -------------------------
    prior_emission_error     = 0.325  # [-]   Fractional emission σ.
                                     #        σ_E = error × scale_factor × E_prior.
                                     #        0.35 → ~35 % per source.
    prior_isotope_d13c_error = 2.0   # [‰]  δ¹³C source-signature prior σ.
                                     #        Larger → inversion can move
                                     #        signatures further from prior.
    prior_isotope_dd_error   = 10.0   # [‰]  δD source-signature prior σ.
    prior_sink_error         = 4.0   # [%]   Sink-perturbation prior σ.

    # --- Temporal smoothness penalty  J_smooth = Σ_k λ_k ‖D(x−xₐ)/σ_k‖² --
    #
    #   λ controls how strongly interannual changes are penalised.
    #   Larger λ → smoother time series, less year-to-year variability,
    #   but poorer fit to observations.
    #
    #   The Hessian contribution is  2λ DᵀD / σ²  (exact for quadratic cost).
    #   σ is taken from the corresponding prior_*_error value.
    #   → Increasing σ weakens smoothness even at fixed λ, and vice versa.
    #
    #   *** The δD smoothness (λ_dd) is now applied independently from δ¹³C
    #       (λ_d13c).  Previously both used the same value via the deprecated
    #       `smoothness_lambda_isotopes` property, causing δD to be over-
    #       smoothed.  Tune λ_d13c and λ_dd separately to balance interannual
    #       variability in each isotope system. ***
    smoothness_lambda_emissions = 10.0  # [-]  Emission smoothness weight.
    smoothness_lambda_d13c      = 0.1   # [-]  δ¹³C signature smoothness weight.
    smoothness_lambda_dd        = 0.01  # [-]  δD  signature smoothness weight.
                                        #       Set lower than λ_d13c to allow
                                        #       more interannual δD variability.
    smoothness_lambda_sinks     = 1.0   # [-]  Sink-perturbation smoothness.
    smoothness_order            = 1     # [-]  Finite-difference order for D.
                                        #       1 = first  derivative (penalise Δx);
                                        #       2 = second derivative (penalise Δ²x,
                                        #           produces smoother curvature).

    # --- Backward-compatibility shim ----------------------------------------
    #   smoothness_lambda_isotopes existed in v0 as a single value shared by
    #   both δ¹³C and δD.  v4 uses separate λ_d13c and λ_dd.  The property
    #   below lets code that reads config.smoothness_lambda_isotopes continue
    #   to work (returns λ_d13c).  It is never used internally by v4.
    @property
    def smoothness_lambda_isotopes(self):
        """v0 compatibility: returns smoothness_lambda_d13c."""
        return self.smoothness_lambda_d13c

    # --- Gauss-Newton settings ----------------------------------------------
    n_iterations = 5   # [-]  Max Gauss-Newton iterations in Step 2.
                        #       Step 1 uses 3 iterations (CH4 only, cheaper).


# ============================================================================
# DATA LOADING
# ============================================================================

def load_prior_emissions(config):
    """
    Load prior emission estimates (CAMS inventory) from text files.

    Returns
    -------
    prior_emissions : dict[source] → ndarray (n_years, 2)
        Columns are [SH, NH] in Tg yr⁻¹.
    """
    inputdir = (
        '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)'
        '/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run'
        '/input_data/prior_emissions'
    )
    years = np.arange(config.year1, config.year2 + 1)
    prior_emissions = {}

    for source in config.sources:
        try:
            data = np.loadtxt(os.path.join(inputdir, f'v2_{source}.txt'))
            prior_emissions[source] = np.column_stack([
                np.interp(years, data[:, 0], data[:, 1]),  # SH
                np.interp(years, data[:, 0], data[:, 2]),  # NH
            ])
        except FileNotFoundError:
            print(f"  ⚠ File not found for {source}, using 100 Tg/yr placeholder")
            prior_emissions[source] = np.ones((len(years), 2)) * 100.0

    return prior_emissions


def load_observations(config):
    """
    Load atmospheric observations (CH4, δ¹³C, δD) and apply latitudinal
    gradients to convert from measurement-site mean to hemispheric mean.

    Returns
    -------
    observations : dict[tracer][hemisphere] → ndarray (n_years,)
        tracers  : 'ch4' [ppb], 'd13c' [‰], 'dd' [‰]
        hemispheres : 'sh', 'nh'
    """
    obs_dir = (
        '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)'
        '/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run'
        '/input_data/observations'
    )
    years = np.arange(config.year1, config.year2 + 1)

    lat_gradients = {
        'ch4':  {'sh':  4.811,  'nh': -39.018},
        'd13c': {'sh': -0.018,  'nh':   0.178},
        'dd':   {'sh': -2.0808, 'nh':   4.315},
    }

    file_map   = [('ch4', 'ch4'), ('d13c-ch4', 'd13c'), ('dd-ch4', 'dd')]
    observations = {}

    for file_tracer, key in file_map:
        observations[key] = {}
        for hem in config.hemispheres:
            grad = lat_gradients[key][hem]
            try:
                data     = np.loadtxt(os.path.join(obs_dir, f'{hem.upper()}_{file_tracer}.txt'))
                time_obs = data[:, 0]
                values   = data[:, -1]

                mask = (time_obs >= config.year1) & (time_obs < config.year2 + 1)
                time_obs, values = time_obs[mask], values[mask]

                annual = np.array([
                    np.mean(values[(time_obs >= y) & (time_obs < y + 1)])
                    if np.sum((time_obs >= y) & (time_obs < y + 1)) > 0
                    else np.nan
                    for y in years
                ])

                if np.any(np.isnan(annual)):
                    valid = ~np.isnan(annual)
                    annual[~valid] = np.interp(years[~valid], years[valid], annual[valid])

                observations[key][hem] = annual + grad

            except FileNotFoundError:
                print(f"  ⚠ Observation file not found: {hem.upper()}_{file_tracer}.txt")
                default = 1800.0 if key == 'ch4' else 0.0
                observations[key][hem] = np.ones(len(years)) * (default + grad)

    return observations


def get_prior_isotopes(config):
    """
    Return fixed prior source-signature values.

    Returns
    -------
    prior_isotopes : dict[source][isotope] → [sh_value, nh_value]
        isotopes : 'd13c' [‰], 'dd' [‰]
    """
    return {
        'wetlands':    {'d13c': [-58.2, -64.8], 'dd': [-299, -336]},
        'agriculture': {'d13c': [-63.4, -63.3], 'dd': [-312, -314]},
        'pyrogenic':   {'d13c': [-22.3, -22.4], 'dd': [-213, -183]},
        'fossil':      {'d13c': [-44.0, -44.0], 'dd': [-192, -191]},
        'waste':       {'d13c': [-54.5, -54.5], 'dd': [-292.2, -292.2]},
    }


# ============================================================================
# STATE VECTOR
# ============================================================================

class StateVector:
    """
    Pack/unpack the flat optimisation vector x.

    Layout per year (36 elements):
      indices  0– 9 : emissions [SH,NH] for each of 5 sources  (Tg yr⁻¹)
      indices 10–19 : δ¹³C signatures [SH,NH] for each source  (‰)
      indices 20–29 : δD signatures [SH,NH] for each source     (‰)
      indices 30–35 : sink perturbations [SH,NH] for OH, strat, soil (%)
    """

    def __init__(self, config):
        self.config           = config
        self.n_years          = config.year2 - config.year1 + 1
        self.n_emissions      = config.n_sources * config.n_hemispheres   # 10
        self.n_iso_per_type   = config.n_sources * config.n_hemispheres   # 10
        self.n_sinks          = 3 * config.n_hemispheres                  # 6
        self.n_params_per_year = self.n_emissions + 2 * self.n_iso_per_type + self.n_sinks  # 36
        self.n_total          = self.n_params_per_year * self.n_years
        print(f"  State vector: {self.n_years} yr × {self.n_params_per_year} params = {self.n_total} total")

    def pack(self, emissions, isotopes_d13c, isotopes_dd, sinks):
        x = np.zeros(self.n_total)
        for yr in range(self.n_years):
            off = yr * self.n_params_per_year
            idx = 0
            for src in self.config.sources:
                x[off + idx]     = emissions[src][yr, 0]
                x[off + idx + 1] = emissions[src][yr, 1]
                idx += 2
            for src in self.config.sources:
                x[off + idx]     = isotopes_d13c[src][yr, 0]
                x[off + idx + 1] = isotopes_d13c[src][yr, 1]
                idx += 2
            for src in self.config.sources:
                x[off + idx]     = isotopes_dd[src][yr, 0]
                x[off + idx + 1] = isotopes_dd[src][yr, 1]
                idx += 2
            for snk in ['oh', 'strat', 'soil']:
                x[off + idx]     = sinks[snk][yr, 0]
                x[off + idx + 1] = sinks[snk][yr, 1]
                idx += 2
        return x

    def unpack(self, x):
        emissions     = {s: np.zeros((self.n_years, 2)) for s in self.config.sources}
        isotopes_d13c = {s: np.zeros((self.n_years, 2)) for s in self.config.sources}
        isotopes_dd   = {s: np.zeros((self.n_years, 2)) for s in self.config.sources}
        sinks         = {k: np.zeros((self.n_years, 2)) for k in ['oh', 'strat', 'soil']}

        for yr in range(self.n_years):
            off = yr * self.n_params_per_year
            idx = 0
            for src in self.config.sources:
                emissions[src][yr, 0]     = x[off + idx]
                emissions[src][yr, 1]     = x[off + idx + 1]
                idx += 2
            for src in self.config.sources:
                isotopes_d13c[src][yr, 0] = x[off + idx]
                isotopes_d13c[src][yr, 1] = x[off + idx + 1]
                idx += 2
            for src in self.config.sources:
                isotopes_dd[src][yr, 0]   = x[off + idx]
                isotopes_dd[src][yr, 1]   = x[off + idx + 1]
                idx += 2
            for snk in ['oh', 'strat', 'soil']:
                sinks[snk][yr, 0]         = x[off + idx]
                sinks[snk][yr, 1]         = x[off + idx + 1]
                idx += 2

        return emissions, isotopes_d13c, isotopes_dd, sinks


# ============================================================================
# FORWARD MODEL
# ============================================================================

class ForwardModel:
    """Two-box atmospheric transport and isotope chemistry model."""

    def __init__(self, config):
        self.config = config
        self.dt     = 1.0 / config.nsteps

    # ---- isotope ↔ mixing-ratio conversions --------------------------------
    def d13c_to_mr(self, ch4, delta):
        r = (delta / 1000 + 1) * self.config.cfact_d13c
        return ch4 * r / (1 + r)

    def mr_to_d13c(self, c13, ch4):
        return ((c13 / (ch4 - c13)) / self.config.cfact_d13c - 1) * 1000

    def dd_to_mr(self, ch4, delta):
        r = (delta / 1000 + 1) * self.config.cfact_dd
        return ch4 * r / (1 + r)

    def mr_to_dd(self, dch3, ch4):
        return ((dch3 / (ch4 - dch3)) / self.config.cfact_dd - 1) * 1000

    # ---- main integration --------------------------------------------------
    def run(self, emissions, isotopes_d13c, isotopes_dd, sinks, init_conditions):
        """
        Integrate the two-box model forward in time.

        Parameters
        ----------
        emissions, isotopes_d13c, isotopes_dd : dict[source] → (n_years, 2)
        sinks : dict[sink_type] → (n_years, 2)   sink perturbations in %
        init_conditions : dict with 'ch4','d13c','dd' → arrays of shape (2,)

        Returns
        -------
        dict with 'ch4', 'd13c', 'dd' → (n_years, 2)
        """
        n_years  = self.config.year2 - self.config.year1 + 1
        cfg      = self.config

        ch4  = np.copy(init_conditions['ch4'])
        c13h4 = self.d13c_to_mr(ch4, init_conditions['d13c'])
        dch3  = self.dd_to_mr(ch4, init_conditions['dd'])

        ch4_out  = np.zeros((n_years, 2))
        d13c_out = np.zeros((n_years, 2))
        dd_out   = np.zeros((n_years, 2))

        for yr in range(n_years):
            total_emis    = np.zeros(2)
            weighted_d13c = np.zeros(2)
            weighted_dd   = np.zeros(2)
            for src in cfg.sources:
                e = emissions[src][yr, :]
                total_emis    += e
                weighted_d13c += e * isotopes_d13c[src][yr, :]
                weighted_dd   += e * isotopes_dd[src][yr, :]

            mean_d13c = weighted_d13c / (total_emis + 1e-10)
            mean_dd   = weighted_dd   / (total_emis + 1e-10)
            c13_emis  = self.d13c_to_mr(total_emis, mean_d13c)
            d_emis    = self.dd_to_mr(total_emis, mean_dd)

            s_oh    = (1 / cfg.lifetime_oh)    * (1 + sinks['oh'][yr, :]    / 100)
            s_strat = (1 / cfg.lifetime_strat) * (1 + sinks['strat'][yr, :] / 100)
            s_soil  = (1 / cfg.lifetime_soil)  * (1 + sinks['soil'][yr, :]  / 100)

            for _ in range(cfg.nsteps):
                ch4  += total_emis * self.dt * cfg.tg2ppb
                c13h4 += c13_emis  * self.dt * cfg.tg2ppb
                dch3  += d_emis    * self.dt * cfg.tg2ppb

                flux   = 0.5 * self.dt / cfg.tau_ihex * (ch4[1]  - ch4[0])
                flux13 = 0.5 * self.dt / cfg.tau_ihex * (c13h4[1] - c13h4[0])
                fluxd  = 0.5 * self.dt / cfg.tau_ihex * (dch3[1]  - dch3[0])
                ch4[0] += flux;  ch4[1] -= flux
                c13h4[0] += flux13; c13h4[1] -= flux13
                dch3[0]  += fluxd;  dch3[1]  -= fluxd

                ch4  -= self.dt * (s_oh + s_strat + s_soil) * ch4
                c13h4 -= self.dt * (s_oh / cfg.kie_oh_d13c    +
                                    s_strat / cfg.kie_strat_d13c +
                                    s_soil  / cfg.kie_soil_d13c) * c13h4
                dch3  -= self.dt * (s_oh / cfg.kie_oh_dd    +
                                    s_strat / cfg.kie_strat_dd +
                                    s_soil  / cfg.kie_soil_dd) * dch3

            ch4_out[yr, :]  = ch4
            d13c_out[yr, :] = self.mr_to_d13c(c13h4, ch4)
            dd_out[yr, :]   = self.mr_to_dd(dch3, ch4)

        return {'ch4': ch4_out, 'd13c': d13c_out, 'dd': dd_out}


# ============================================================================
# PRIOR COVARIANCE (diagonal Sa)
# ============================================================================

def two_step_construct_prior_covariance(config, state_vec, prior_emissions, prior_isotopes):
    """
    Build the diagonal prior error covariance matrix Sa.

    Emission σ scales with source fraction so that minor sources have
    relatively wider priors.  Isotope and sink σ are uniform (from config).

    Returns
    -------
    Sa : ndarray (n_total, n_total), diagonal.
    """
    n_total          = state_vec.n_total
    n_years          = state_vec.n_years
    n_params_per_year = state_vec.n_params_per_year

    total_by_src = {s: np.mean(prior_emissions[s][:, 0] + prior_emissions[s][:, 1])
                    for s in config.sources}
    global_total = sum(total_by_src.values())
    fractions    = {s: total_by_src[s] / global_total for s in config.sources}
    print("  Source fractions: " +
          ", ".join(f"{s}: {fractions[s]:.2f}" for s in config.sources))

    Sa_diag = np.zeros(n_total)
    for yr in range(n_years):
        off = yr * n_params_per_year
        idx = 0

        for src in config.sources:
            scale = 1.0 - fractions[src]
            for hem in range(2):
                Sa_diag[off + idx] = (config.prior_emission_error * scale *
                                      prior_emissions[src][yr, hem]) ** 2
                idx += 1

        for _ in range(config.n_sources * config.n_hemispheres):   # δ¹³C
            Sa_diag[off + idx] = config.prior_isotope_d13c_error ** 2
            idx += 1

        for _ in range(config.n_sources * config.n_hemispheres):   # δD
            Sa_diag[off + idx] = config.prior_isotope_dd_error ** 2
            idx += 1

        for _ in range(6):                                          # sinks
            Sa_diag[off + idx] = config.prior_sink_error ** 2
            idx += 1

    return np.diag(Sa_diag)


# ============================================================================
# TEMPORAL SMOOTHNESS  J_smooth = Σ_k λ_k ‖D(x−xₐ)/σ_k‖²
# ============================================================================

def build_smoothness_matrix(n_years, n_params_per_year, param_indices, order=1):
    """
    Build a finite-difference matrix D of shape (n_constraints, n_total).

    order=1: D penalises first differences  Δx[t] = x[t] − x[t−1]
    order=2: D penalises second differences Δ²x[t] = x[t−1] − 2x[t] + x[t+1]
    """
    n_total  = n_years * n_params_per_year
    n_params = len(param_indices)

    if order == 1:
        n_rows = (n_years - 1) * n_params
        D      = np.zeros((n_rows, n_total))
        row    = 0
        for yr in range(n_years - 1):
            for p in param_indices:
                D[row, yr * n_params_per_year + p]       = -1.0
                D[row, (yr + 1) * n_params_per_year + p] =  1.0
                row += 1
    else:  # order == 2
        n_rows = (n_years - 2) * n_params
        D      = np.zeros((n_rows, n_total))
        row    = 0
        for yr in range(1, n_years - 1):
            for p in param_indices:
                D[row, (yr - 1) * n_params_per_year + p] =  1.0
                D[row,  yr      * n_params_per_year + p] = -2.0
                D[row, (yr + 1) * n_params_per_year + p] =  1.0
                row += 1
    return D


def _compute_smoothness_penalty_full(x, x_prior, config, state_vec, D_matrices=None):
    """
    Internal implementation: returns (cost, grad, H_smooth, D_matrices).

    The penalty for parameter group k is:
        J_k = λ_k · ‖D·(x − xₐ) / σ_k‖²

    The exact Hessian contribution is:
        H_k = 2 λ_k / σ_k²  ·  DᵀD

    Bug fixed vs v0: δ¹³C and δD now use independent λ and σ in both the
    gradient and Hessian.  The 1/σ² normalisation is correctly applied.

    Parameters
    ----------
    D_matrices : tuple or None
        Pre-computed (D_emis, D_d13c, D_dd, D_sinks); computed on first call.

    Returns
    -------
    cost      : float
    grad      : ndarray (n_total,)
    H_smooth  : ndarray (n_total, n_total)
    D_matrices: tuple
    """
    n_years           = state_vec.n_years
    n_params_per_year = state_vec.n_params_per_year

    if D_matrices is None:
        D_emis  = build_smoothness_matrix(n_years, n_params_per_year,
                                          list(range(10)),    config.smoothness_order)
        D_d13c  = build_smoothness_matrix(n_years, n_params_per_year,
                                          list(range(10, 20)), config.smoothness_order)
        D_dd    = build_smoothness_matrix(n_years, n_params_per_year,
                                          list(range(20, 30)), config.smoothness_order)
        D_sinks = build_smoothness_matrix(n_years, n_params_per_year,
                                          list(range(30, 36)), config.smoothness_order)
        D_matrices = (D_emis, D_d13c, D_dd, D_sinks)
    else:
        D_emis, D_d13c, D_dd, D_sinks = D_matrices

    # Typical emission scale for dimensionless normalisation
    emis_dict, _, _, _ = state_vec.unpack(x_prior)
    all_emis = np.concatenate([emis_dict[s].flatten() for s in config.sources])
    typical_emis = np.mean(all_emis[all_emis > 0])
    sig_emis  = config.prior_emission_error * typical_emis  # Tg/yr
    sig_d13c  = config.prior_isotope_d13c_error              # ‰
    sig_dd    = config.prior_isotope_dd_error                 # ‰
    sig_sinks = config.prior_sink_error                       # %

    dx = x - x_prior

    # Normalised residuals
    r_emis  = (D_emis  @ dx) / sig_emis
    r_d13c  = (D_d13c  @ dx) / sig_d13c
    r_dd    = (D_dd    @ dx) / sig_dd
    r_sinks = (D_sinks @ dx) / sig_sinks

    lam_e = config.smoothness_lambda_emissions
    lam_c = config.smoothness_lambda_d13c
    lam_d = config.smoothness_lambda_dd
    lam_s = config.smoothness_lambda_sinks

    # Scalar cost
    cost = (lam_e * np.dot(r_emis,  r_emis)  +
            lam_c * np.dot(r_d13c,  r_d13c)  +
            lam_d * np.dot(r_dd,    r_dd)    +
            lam_s * np.dot(r_sinks, r_sinks))

    # Gradient  ∂J/∂x = 2 λ DᵀD dx / σ²
    grad = (2 * lam_e * D_emis.T  @ r_emis  / sig_emis  +
            2 * lam_c * D_d13c.T  @ r_d13c  / sig_d13c  +
            2 * lam_d * D_dd.T    @ r_dd    / sig_dd    +
            2 * lam_s * D_sinks.T @ r_sinks / sig_sinks)

    # Exact Hessian (quadratic penalty → Hessian is constant)
    H_smooth = (2 * lam_e / sig_emis**2  * (D_emis.T  @ D_emis)  +
                2 * lam_c / sig_d13c**2  * (D_d13c.T  @ D_d13c)  +
                2 * lam_d / sig_dd**2    * (D_dd.T    @ D_dd)    +
                2 * lam_s / sig_sinks**2 * (D_sinks.T @ D_sinks))

    return cost, grad, H_smooth, D_matrices


def compute_smoothness_penalty(x, x_prior, config, state_vec, D_matrices=None):
    """
    Public interface — returns (cost, grad, D_matrices) matching the v0
    signature so the runner's  cost_smooth, _, _ = compute_smoothness_penalty(...)
    continues to work unchanged.

    two_step_inversion calls _compute_smoothness_penalty_full directly to
    obtain H_smooth for the Gauss-Newton Hessian.
    """
    cost, grad, _, D_matrices = _compute_smoothness_penalty_full(
        x, x_prior, config, state_vec, D_matrices)
    return cost, grad, D_matrices


# ============================================================================
# OBSERVATION OPERATORS
# ============================================================================

def get_observation_covariance(config, n_years, ch4_only=False):
    """Diagonal observation error covariance Sy."""
    if ch4_only:
        errors = [config.obs_sigma_ch4_s0] * (2 * n_years)
    else:
        errors = []
        sigma_map = {'ch4': config.obs_sigma_ch4,
                     'd13c': config.obs_sigma_d13c,
                     'dd': config.obs_sigma_dd}
        for tracer in ['ch4', 'd13c', 'dd']:
            errors.extend([sigma_map[tracer]] * (2 * n_years))   # SH + NH
    return np.diag(np.array(errors) ** 2)


def get_observation_vector(observations, ch4_only=False):
    """Stack observations into a single vector y_obs."""
    y = []
    if ch4_only:
        for hem in ['sh', 'nh']:
            y.extend(observations['ch4'][hem])
    else:
        for tracer in ['ch4', 'd13c', 'dd']:
            for hem in ['sh', 'nh']:
                y.extend(observations[tracer][hem])
    return np.array(y)


def observation_operator(model_output, observations, ch4_only=False):
    """Sample model output at observation locations."""
    y = []
    if ch4_only:
        for hi, hem in enumerate(['sh', 'nh']):
            y.extend(model_output['ch4'][:, hi])
    else:
        for tracer in ['ch4', 'd13c', 'dd']:
            for hi in range(2):
                y.extend(model_output[tracer][:, hi])
    return np.array(y)


# ============================================================================
# JACOBIAN (finite differences)
# ============================================================================

def compute_jacobian(config, state_vec, forward_model, x, init_conditions,
                     y_ref, ch4_only=False):
    """
    Compute the Jacobian matrix K = ∂y / ∂x by finite differences.

    Perturbation sizes are adaptive for emissions and sinks, fixed for
    isotope signatures (0.1‰ for δ¹³C, 1.0‰ for δD).

    Returns
    -------
    K : ndarray (n_obs, n_state)
    """
    n_state           = len(x)
    n_obs             = len(y_ref)
    n_years           = state_vec.n_years
    n_params_per_year = state_vec.n_params_per_year
    n_e               = 10   # emissions
    n_c               = 10   # δ¹³C
    n_d               = 10   # δD

    K = np.zeros((n_obs, n_state))
    print("  Computing Jacobian ...")

    # Check parameter variability
    _, iso_d13c_ref, iso_dd_ref, _ = state_vec.unpack(x)
    for src in config.sources:
        print(f"    {src:12s}: δ¹³C std={np.std(iso_d13c_ref[src]):.3f}‰,"
              f"  δD std={np.std(iso_dd_ref[src]):.2f}‰")

    for yr in range(n_years):
        if yr % 5 == 0:
            print(f"    Year {config.year1 + yr} ({100 * yr / n_years:.0f}%) ...")
        off = yr * n_params_per_year

        # --- Emissions (0–9) ------------------------------------------------
        for i in range(n_e):
            pi = off + i
            pert = max(abs(x[pi]) * 0.01, 1e-6)
            xp = x.copy(); xp[pi] += pert
            e, c, d, s = state_vec.unpack(xp)
            yp = observation_operator(forward_model.run(e, c, d, s, init_conditions), None, ch4_only)
            K[:, pi] = (yp - y_ref) / pert

        if not ch4_only:
            # --- δ¹³C (10–19) -----------------------------------------------
            for i in range(n_c):
                pi = off + n_e + i
                xp = x.copy(); xp[pi] += 0.1
                e, c, d, s = state_vec.unpack(xp)
                yp = observation_operator(forward_model.run(e, c, d, s, init_conditions), None, ch4_only)
                K[:, pi] = (yp - y_ref) / 0.1

            # --- δD (20–29) -------------------------------------------------
            for i in range(n_d):
                pi = off + n_e + n_c + i
                xp = x.copy(); xp[pi] += 1.0
                e, c, d, s = state_vec.unpack(xp)
                yp = observation_operator(forward_model.run(e, c, d, s, init_conditions), None, ch4_only)
                K[:, pi] = (yp - y_ref) / 1.0

        # --- Sinks (30–35) --------------------------------------------------
        for i in range(6):
            pi = off + n_e + n_c + n_d + i
            pert = max(abs(x[pi]) * 0.01, 0.1)
            xp = x.copy(); xp[pi] += pert
            e, c, d, s = state_vec.unpack(xp)
            yp = observation_operator(forward_model.run(e, c, d, s, init_conditions), None, ch4_only)
            K[:, pi] = (yp - y_ref) / pert

    zero_cols = np.sum(np.abs(K), axis=0) < 1e-15
    if zero_cols.any():
        print(f"  ⚠ {zero_cols.sum()} zero-gradient parameters")
    K_norms = np.linalg.norm(K, axis=0)
    valid   = K_norms > 0
    print(f"  ✓ Jacobian {K.shape}  |K| range: "
          f"{K_norms[valid].min():.2e} – {K_norms.max():.2e}")
    return K


# ============================================================================
# TWO-STEP GAUSS-NEWTON INVERSION
# ============================================================================

def two_step_inversion(config, state_vec, forward_model,
                       x_prior, prior_emissions, prior_isotopes,
                       observations, init_conditions):
    """
    Two-step Gauss-Newton inversion with temporal smoothness.

    Normal equations solved at each iteration:
        H · δx = g
        H = KᵀSy⁻¹K + Sa⁻¹ + H_smooth
        g = KᵀSy⁻¹(y − F(x)) − Sa⁻¹(x − xₐ) − ∇J_smooth

    H_smooth is the exact Hessian of the quadratic smoothness penalty,
    returned by compute_smoothness_penalty.

    Returns
    -------
    x_post      : ndarray  Posterior state vector.
    diagnostics : dict     Cost histories and intermediate results.
    """
    Sa     = two_step_construct_prior_covariance(config, state_vec, prior_emissions, prior_isotopes)
    Sa_inv = linalg.inv(Sa)

    # ---- Step 1: CH4-only (emissions + sinks) ------------------------------
    print("\n" + "=" * 68)
    print("STEP 1: CH4-ONLY INVERSION (EMISSIONS + SINKS)")
    print("=" * 68)

    y_obs1   = get_observation_vector(observations, ch4_only=True)
    Sy1      = get_observation_covariance(config, state_vec.n_years, ch4_only=True)
    Sy1_inv  = linalg.inv(Sy1)

    x1          = x_prior.copy()
    costs1      = []
    D_matrices  = None

    for it in range(3):
        print(f"\n  Iteration {it + 1}/3")
        e, c, d, s = state_vec.unpack(x1)
        out  = forward_model.run(e, c, d, s, init_conditions)
        y1   = observation_operator(out, None, ch4_only=True)

        innov     = y_obs1 - y1
        dx        = x1 - x_prior
        cost_obs  = float(innov @ Sy1_inv @ innov)
        cost_pri  = float(dx @ Sa_inv @ dx)

        cost_sm, grad_sm, H_smooth, D_matrices = _compute_smoothness_penalty_full(
            x1, x_prior, config, state_vec, D_matrices)

        total = cost_obs + cost_pri + cost_sm
        costs1.append(total)
        print(f"    Cost: {total:.2f}  (obs {cost_obs:.2f}, prior {cost_pri:.2f}, smooth {cost_sm:.2f})")

        K = compute_jacobian(config, state_vec, forward_model, x1,
                             init_conditions, y1, ch4_only=True)

        H = K.T @ Sy1_inv @ K + Sa_inv + H_smooth
        g = K.T @ Sy1_inv @ innov - Sa_inv @ dx - grad_sm

        try:
            x1 = x1 + linalg.solve(H, g)
        except linalg.LinAlgError:
            print("    Matrix solve failed – stopping Step 1")
            break

    print(f"\n  Step 1 done.  Cost: {costs1[0]:.2f} → {costs1[-1]:.2f}")

    # ---- Step 2: all tracers (emissions + isotopes + sinks) ----------------
    print("\n" + "=" * 68)
    print("STEP 2: MULTI-TRACER OPTIMISATION (ALL PARAMETERS)")
    print("=" * 68)
    print(f"  σ_obs: CH4={config.obs_sigma_ch4} ppb, "
          f"δ¹³C={config.obs_sigma_d13c} ‰, δD={config.obs_sigma_dd} ‰")
    print(f"  λ_smooth: emis={config.smoothness_lambda_emissions}, "
          f"d13c={config.smoothness_lambda_d13c}, "
          f"dD={config.smoothness_lambda_dd}, "
          f"sinks={config.smoothness_lambda_sinks}")

    y_obs2  = get_observation_vector(observations, ch4_only=False)
    Sy2     = get_observation_covariance(config, state_vec.n_years, ch4_only=False)
    Sy2_inv = linalg.inv(Sy2)

    x2         = x1.copy()
    costs2     = []
    D_matrices = None

    for it in range(config.n_iterations):
        print(f"\n  Iteration {it + 1}/{config.n_iterations}")
        e, c, d, s = state_vec.unpack(x2)
        out  = forward_model.run(e, c, d, s, init_conditions)
        y2   = observation_operator(out, None, ch4_only=False)

        innov     = y_obs2 - y2
        dx        = x2 - x_prior
        cost_obs  = float(innov @ Sy2_inv @ innov)
        cost_pri  = float(dx @ Sa_inv @ dx)

        cost_sm, grad_sm, H_smooth, D_matrices = _compute_smoothness_penalty_full(
            x2, x_prior, config, state_vec, D_matrices)

        total = cost_obs + cost_pri + cost_sm
        costs2.append(total)
        print(f"    Cost: {total:.2f}  (obs {cost_obs:.2f}, prior {cost_pri:.2f}, smooth {cost_sm:.2f})")

        K = compute_jacobian(config, state_vec, forward_model, x2,
                             init_conditions, y2, ch4_only=False)

        # Hessian now includes exact smoothness term (no manual D assembly needed)
        H = K.T @ Sy2_inv @ K + Sa_inv + H_smooth
        g = K.T @ Sy2_inv @ innov - Sa_inv @ dx - grad_sm

        try:
            delta = linalg.solve(H, g)
        except linalg.LinAlgError:
            print("    Matrix solve failed – stopping Step 2")
            break

        # Backtracking line search
        alpha  = 1.0
        x_new  = x2 + delta
        for _ in range(5):
            e_t, c_t, d_t, s_t = state_vec.unpack(x_new)
            out_t = forward_model.run(e_t, c_t, d_t, s_t, init_conditions)
            inn_t = y_obs2 - observation_operator(out_t, None, ch4_only=False)
            dx_t  = x_new - x_prior
            csm_t, _, _, _ = _compute_smoothness_penalty_full(
                x_new, x_prior, config, state_vec, D_matrices)
            cost_t = float(inn_t @ Sy2_inv @ inn_t) + float(dx_t @ Sa_inv @ dx_t) + csm_t
            if cost_t < total:
                break
            alpha *= 0.5
            x_new  = x2 + alpha * delta

        x2 = x_new
        print(f"    Step ‖δx‖={np.linalg.norm(delta):.2e}, α={alpha:.3f}")

        if it > 0:
            rel_chg = abs(costs2[-1] - costs2[-2]) / (costs2[-2] + 1e-30)
            if rel_chg < 1e-5:
                print(f"    Converged (relative change {rel_chg:.2e})")
                break

    print(f"\n  Step 2 done.  Cost: {costs2[0]:.2f} → {costs2[-1]:.2f}")

    e_f, c_f, d_f, s_f = state_vec.unpack(x2)
    out_final = forward_model.run(e_f, c_f, d_f, s_f, init_conditions)

    return x2, {
        'costs_step1': costs1,
        'costs_step2': costs2,
        'costs':       costs1 + costs2,
        'x_step1':     x1,
        'final_output': out_final,
    }


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def _temporal_stats(years, emissions, source):
    """Year-to-year fractional change statistics for one source."""
    total   = emissions[source][:, 0] + emissions[source][:, 1]
    changes = np.diff(total) / (total[:-1] + 1e-10)
    return {
        'mean_abs': np.mean(np.abs(changes)) * 100,
        'max_abs':  np.max(np.abs(changes))  * 100,
        'std':      np.std(changes)           * 100,
    }


def print_smoothness_diagnostics(years, prior_emissions, post_emissions, config):
    """Print before/after year-to-year variability for each source."""
    print("\n" + "=" * 68)
    print("TEMPORAL SMOOTHNESS DIAGNOSTICS")
    print("=" * 68)
    for src in config.sources:
        pr  = _temporal_stats(years, prior_emissions, src)
        po  = _temporal_stats(years, post_emissions,  src)
        tag = "✓ smoother" if po['mean_abs'] < pr['mean_abs'] else "⚠ less smooth"
        print(f"\n  {src.upper()}")
        print(f"    Mean |Δ|: {pr['mean_abs']:.1f}% → {po['mean_abs']:.1f}%  {tag}")
        print(f"    Max  |Δ|: {pr['max_abs']:.1f}%  → {po['max_abs']:.1f}%")
        print(f"    Std   Δ : {pr['std']:.1f}%    → {po['std']:.1f}%")


def calculate_total_lifetime(sinks, config):
    """
    Convert sink perturbations (%) to total CH4 lifetime.

    Returns ndarray (n_years, 2) where columns are [SH, NH].
    """
    r_oh    = (1 / config.lifetime_oh.reshape(1, 2))    * (1 + sinks['oh']    / 100)
    r_strat = (1 / config.lifetime_strat.reshape(1, 2)) * (1 + sinks['strat'] / 100)
    r_soil  = (1 / config.lifetime_soil.reshape(1, 2))  * (1 + sinks['soil']  / 100)
    return 1.0 / (r_oh + r_strat + r_soil)


# ============================================================================
# EXCEL OUTPUT  (one file, four sheets)
# ============================================================================

def save_excel_output(years, observations, output_prior, output_post,
                      prior_emissions, emis_post,
                      prior_sinks, sinks_post,
                      prior_iso_d13c, iso_d13c_post,
                      prior_iso_dd,   iso_dd_post,
                      config, output_dir):
    """
    Write all time-series results to a single Excel workbook with four sheets.

    Sheet 1 – obs              : observations and model fit (CH4, δ¹³C, δD)
    Sheet 2 – emissions_lifetime : prior/posterior emissions + total lifetime
    Sheet 3 – d13C             : prior/posterior δ¹³C source signatures
    Sheet 4 – dD               : prior/posterior δD source signatures
    """
    path   = os.path.join(output_dir, 'inversion_results.xlsx')
    n      = len(years)

    # ---- Sheet 1: observations & model fit ---------------------------------
    d1 = {'Year': years}
    for tracer, unit in [('ch4', 'ppb'), ('d13c', 'permil'), ('dd', 'permil')]:
        for hem in ['sh', 'nh']:
            suffix = f'{tracer}_{hem.upper()}'
            d1[f'obs_{suffix}']   = observations[tracer][hem]
            d1[f'prior_{suffix}'] = output_prior[tracer][:, 0 if hem == 'sh' else 1]
            d1[f'post_{suffix}']  = output_post[tracer][:,  0 if hem == 'sh' else 1]
    df_obs = pd.DataFrame(d1)

    # ---- Sheet 2: emissions & lifetime -------------------------------------
    d2 = {'Year': years}
    for src in config.sources:
        for hem_idx, hem in enumerate(['SH', 'NH']):
            d2[f'{src}_prior_{hem}'] = prior_emissions[src][:, hem_idx]
            d2[f'{src}_post_{hem}']  = emis_post[src][:,      hem_idx]
    prior_lt = calculate_total_lifetime(prior_sinks, config)
    post_lt  = calculate_total_lifetime(sinks_post,  config)
    for hem_idx, hem in enumerate(['SH', 'NH']):
        d2[f'lifetime_prior_{hem}'] = prior_lt[:, hem_idx]
        d2[f'lifetime_post_{hem}']  = post_lt[:,  hem_idx]
    df_emis = pd.DataFrame(d2)

    # ---- Sheet 3: δ¹³C source signatures -----------------------------------
    d3 = {'Year': years}
    for src in config.sources:
        for hem_idx, hem in enumerate(['SH', 'NH']):
            d3[f'{src}_prior_{hem}'] = prior_iso_d13c[src][:, hem_idx]
            d3[f'{src}_post_{hem}']  = iso_d13c_post[src][:,  hem_idx]
    df_d13c = pd.DataFrame(d3)

    # ---- Sheet 4: δD source signatures -------------------------------------
    d4 = {'Year': years}
    for src in config.sources:
        for hem_idx, hem in enumerate(['SH', 'NH']):
            d4[f'{src}_prior_{hem}'] = prior_iso_dd[src][:, hem_idx]
            d4[f'{src}_post_{hem}']  = iso_dd_post[src][:, hem_idx]
    df_dd = pd.DataFrame(d4)

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df_obs.to_excel( writer, sheet_name='obs',                index=False)
        df_emis.to_excel(writer, sheet_name='emissions_lifetime', index=False)
        df_d13c.to_excel(writer, sheet_name='d13C',              index=False)
        df_dd.to_excel(  writer, sheet_name='dD',                index=False)

    print(f"  ✓ Saved: inversion_results.xlsx  (4 sheets: obs, emissions_lifetime, d13C, dD)")
    return path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 68)
    print("TIME-VARYING EMISSION + ISOTOPE INVERSION (TWO-STEP)")
    print("=" * 68)

    config        = InversionConfig()
    state_vec     = StateVector(config)
    forward_model = ForwardModel(config)

    print("\n=== Loading Data ===")
    prior_emissions = load_prior_emissions(config)
    observations    = load_observations(config)
    prior_isotopes  = get_prior_isotopes(config)

    print("\n=== Initialising Prior ===")
    n_years = config.year2 - config.year1 + 1
    years   = np.arange(config.year1, config.year2 + 1)

    prior_iso_d13c = {s: np.tile(prior_isotopes[s]['d13c'], (n_years, 1))
                      for s in config.sources}
    prior_iso_dd   = {s: np.tile(prior_isotopes[s]['dd'],   (n_years, 1))
                      for s in config.sources}
    prior_sinks    = {k: np.zeros((n_years, 2)) for k in ['oh', 'strat', 'soil']}

    x_prior = state_vec.pack(prior_emissions, prior_iso_d13c, prior_iso_dd, prior_sinks)
    print(f"  Prior state vector: {len(x_prior)} elements")

    init_conditions = {
        'ch4':  np.array([observations['ch4']['sh'][0],  observations['ch4']['nh'][0]]),
        'd13c': np.array([observations['d13c']['sh'][0], observations['d13c']['nh'][0]]),
        'dd':   np.array([observations['dd']['sh'][0],   observations['dd']['nh'][0]]),
    }

    print("\n=== Testing Prior ===")
    e0, c0, d0, s0 = state_vec.unpack(x_prior)
    output_prior   = forward_model.run(e0, c0, d0, s0, init_conditions)

    print("\n=== Running Inversion ===")
    x_post, diagnostics = two_step_inversion(
        config, state_vec, forward_model,
        x_prior, prior_emissions, prior_isotopes,
        observations, init_conditions,
    )

    print("\n=== Extracting Posterior ===")
    emis_post, iso_d13c_post, iso_dd_post, sinks_post = state_vec.unpack(x_post)
    print_smoothness_diagnostics(years, prior_emissions, emis_post, config)

    output_post = forward_model.run(emis_post, iso_d13c_post, iso_dd_post,
                                    sinks_post, init_conditions)

    output_dir = (
        '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)'
        '/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run'
        '/multi_inversion_results_v4/time_varying_inversion_outputs_two_step'
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Saving Results ===")
    save_excel_output(
        years, observations, output_prior, output_post,
        prior_emissions, emis_post,
        prior_sinks, sinks_post,
        prior_iso_d13c, iso_d13c_post,
        prior_iso_dd,   iso_dd_post,
        config, output_dir,
    )

    print("\n=== Creating Plots ===")
    plot_results(
        config, years,
        emis_post, iso_d13c_post, iso_dd_post, sinks_post,
        prior_emissions, prior_iso_d13c, prior_iso_dd, prior_sinks,
        output_prior, output_post, observations, output_dir,
    )

    print(f"\n✅ Inversion complete!")
    print(f"   Cost: {diagnostics['costs'][0]:.2f} → {diagnostics['costs'][-1]:.2f}")
    print(f"📁 Results saved to: {output_dir}")


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

FONTSIZE = {'title': 12, 'tick_label': 10}

COLORS = {
    'wetlands':  '#2ecc71',
    'agriculture': '#e67e22',
    'pyrogenic': '#e74c3c',
    'fossil':    '#7f8c8d',
    'waste':     '#9b59b6',
}


def load_mei_data(year1=1980, year2=2024):
    """Load and classify annual MEI values (El Niño / La Niña / Neutral)."""
    try:
        mei_file = (
            '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)'
            '/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run'
            '/input_data/meiv2.data.txt'
        )
        cols = ['Year', 'DJ', 'JF', 'FM', 'MA', 'AM', 'MJ',
                'JJ', 'JA', 'AS', 'SO', 'ON', 'ND']
        raw = pd.read_csv(mei_file, sep=r'\s+', skiprows=1, skipfooter=4,
                          header=None, engine='python')
        raw.columns = cols
        long = raw.melt(id_vars='Year', var_name='Month', value_name='Anomaly')
        month_map = {n: str(i).zfill(2) for i, n in enumerate(cols[1:], 1)}
        long['Month'] = long['Month'].map(month_map)
        long['Date']  = pd.to_datetime(long['Year'].astype(str) + '-' + long['Month'],
                                       format='%Y-%m')
        long = long.dropna(subset=['Anomaly'])
        long['Year'] = long['Date'].dt.year
        mei = (long[(long['Year'] >= year1) & (long['Year'] <= year2)]
               .groupby('Year')['Anomaly'].mean()
               .reset_index()
               .rename(columns={'Anomaly': 'MEI'}))
        mei['ENSO_type'] = 'Neutral'
        mei.loc[mei['MEI'] >  0.5, 'ENSO_type'] = 'El Niño'
        mei.loc[mei['MEI'] < -0.5, 'ENSO_type'] = 'La Niña'
        return mei
    except Exception as exc:
        print(f"  ⚠ MEI load failed: {exc}")
        years = np.arange(year1, year2 + 1)
        return pd.DataFrame({'Year': years, 'MEI': 0.0, 'ENSO_type': 'Neutral'})


mei_df        = load_mei_data(1980, 2024)
el_nino_years = mei_df[mei_df['MEI'] >  0.5]['Year'].values
la_nina_years = mei_df[mei_df['MEI'] < -0.5]['Year'].values


def shade_enso(ax):
    """Shade El Niño (red) and La Niña (blue) years on an axes."""
    for y in el_nino_years:
        ax.axvspan(y - 0.5, y + 0.5, color='red',  alpha=0.1)
    for y in la_nina_years:
        ax.axvspan(y - 0.5, y + 0.5, color='blue', alpha=0.1)


def _spinup_shade(ax, years):
    """Grey shading for spin-up (pre-1994) and spin-down (post-2022) periods."""
    ax.axvspan(years[0], 1994,       facecolor='grey', alpha=0.3)
    ax.axvspan(2022,     years[-1],  facecolor='grey', alpha=0.3)


# ============================================================================
# PLOT ORCHESTRATOR
# ============================================================================

def plot_results(config, years,
                 emis_post, iso_d13c_post, iso_dd_post, sinks_post,
                 prior_emissions, prior_iso_d13c, prior_iso_dd, prior_sinks,
                 output_prior, output_post, observations, output_dir):
    """Create all output figures."""
    plot_f1_model_data_fit(years, output_prior, output_post,
                           observations, output_dir)
    plot_f2_emissions_comparison(years, prior_emissions, emis_post,
                                 config, output_dir, prior_sinks, sinks_post)
    plot_f5_isotopes(years, iso_d13c_post, iso_dd_post,
                     prior_iso_d13c, prior_iso_dd, config, output_dir)
    plot_isotope_enso_shifts_diff(years, iso_d13c_post, iso_dd_post, output_dir,
                                  prior_emissions=prior_emissions)
    plot_f4_optimized_timeseries(years, iso_d13c_post, iso_dd_post, sinks_post,
                                 prior_iso_d13c, prior_iso_dd, prior_sinks,
                                 config, output_dir, emis_post)


# ============================================================================
# F1: MODEL-DATA FIT
# ============================================================================

def plot_f1_model_data_fit(years, output_prior, output_post,
                           observations, output_dir):
    """
    F1: Observations (black) vs prior (blue) vs posterior (red) for
    CH4, δ¹³C, and δD in SH (+) and NH (·).
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12.5), sharex=True)
    tracers = ['ch4', 'd13c', 'dd']
    labels  = ['CH$_4$ (ppb)', r'$\delta^{13}$C-CH$_4$ (‰)', r'$\delta$D-CH$_4$ (‰)']
    markers = ['+', '.']

    for i, (tracer, label) in enumerate(zip(tracers, labels)):
        ax = axes[i]
        shade_enso(ax)
        for j, hem in enumerate(['sh', 'nh']):
            hi = j
            ax.plot(years, observations[tracer][hem], color='black',
                    marker=markers[j], linestyle='--', markersize=6, alpha=0.8,
                    label=hem.upper() if i == 0 else '')
            ax.plot(years, output_prior[tracer][:, hi], color='blue',
                    linestyle='-', linewidth=1.5, marker=markers[j],
                    markersize=3, alpha=0.6)
            ax.plot(years, output_post[tracer][:, hi], color='red',
                    linestyle='-', linewidth=1.5, marker=markers[j],
                    markersize=3, alpha=0.6)

        ax.set_ylabel(label, fontsize=12)
        ax.yaxis.set_ticks_position('both')
        ax.grid(alpha=0.3)
        _spinup_shade(ax, years)
        ax.set_xlim(years[0], years[-1])

        if i < 2:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)
        if i > 0:
            ax.spines['top'].set_visible(False)

    axes[0].legend(loc='upper left', frameon=False, fontsize=10)
    axes[2].set_xlabel('Year', fontsize=12)
    axes[2].set_xticks(years[::4])

    plt.tight_layout(h_pad=0)
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(output_dir, 'f1_model_data_fit.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: f1_model_data_fit.png")


# ============================================================================
# F2: PRIOR vs POSTERIOR EMISSIONS + LIFETIME (SH and NH separate)
# ============================================================================

def plot_f2_emissions_comparison(years, prior_emissions, emis_post, config,
                                 output_dir, prior_sinks, sinks_post):
    """
    F2: 3×2 panel showing prior vs posterior for 5 emission sources (a–e)
    and hemispheric CH4 lifetime SH/NH separately in panel (f).

    Panel (f) now shows four curves:
        Prior SH (grey dotted), Prior NH (grey dash-dot),
        Posterior SH (red solid), Posterior NH (red dashed)
    matching the style convention used for the emission panels.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    sources_plot  = config.sources + ['total_sink']
    panel_labels  = list('abcdef')

    prior_lt = calculate_total_lifetime(prior_sinks, config)   # (n_years, 2)
    post_lt  = calculate_total_lifetime(sinks_post,  config)

    for i, source in enumerate(sources_plot):
        row = i // 2
        col = i %  2
        ax  = axes[row, col]
        shade_enso(ax)
        ax.text(0.01, 0.98, f'({panel_labels[i]})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left', zorder=200)

        if source == 'total_sink':
            # Four separate lines: prior SH/NH, posterior SH/NH
            ax.plot(years, prior_lt[:, 0], color='grey', linestyle=':',
                    linewidth=2.5, alpha=0.7,  label='Prior SH',      zorder=50)
            ax.plot(years, prior_lt[:, 1], color='grey', linestyle='-.',
                    linewidth=2.5, alpha=0.7,  label='Prior NH',      zorder=50)
            ax.plot(years, post_lt[:, 0],  color='red',  linestyle='-',
                    linewidth=2.0, alpha=0.85, label='Posterior SH',  zorder=100)
            ax.plot(years, post_lt[:, 1],  color='red',  linestyle='--',
                    linewidth=1.5, alpha=0.7,  label='Posterior NH',  zorder=100)
            ax.set_ylabel('Lifetime (years)', fontsize=11)
            ax.set_title('Total Sink', fontsize=12, fontweight='bold')

            proxy = [
                Line2D([0], [0], color='grey', linestyle=':',  linewidth=2, label='Prior SH'),
                Line2D([0], [0], color='grey', linestyle='-.', linewidth=2, label='Prior NH'),
                Line2D([0], [0], color='red',  linestyle='-',  linewidth=2, label='Posterior SH'),
                Line2D([0], [0], color='red',  linestyle='--', linewidth=2, label='Posterior NH'),
            ]
            ax.legend(handles=proxy, loc='lower right', frameon=True,
                      fontsize=FONTSIZE['title'], ncol=2)

        else:
            ax.plot(years, prior_emissions[source][:, 0], color='grey',
                    linestyle=':', linewidth=2.5, marker='.', markersize=3, alpha=0.7,
                    zorder=50,  label='Prior SH' if i == 0 else '')
            ax.plot(years, prior_emissions[source][:, 1], color='grey',
                    linestyle='-.', linewidth=2.5, alpha=0.7,
                    zorder=50,  label='Prior NH' if i == 0 else '')
            ax.plot(years, emis_post[source][:, 0], color='red',
                    linestyle='-', linewidth=2.0, marker='.', markersize=3, alpha=0.85,
                    zorder=100, label='Posterior SH' if i == 0 else '')
            ax.plot(years, emis_post[source][:, 1], color='red',
                    linestyle='--', linewidth=1.5, alpha=0.7,
                    zorder=100, label='Posterior NH' if i == 0 else '')
            ax.set_ylabel('Emissions (Tg/yr)', fontsize=11)
            ax.set_title(source.capitalize(), fontsize=12, fontweight='bold')

        ax.grid(alpha=0.3)
        ax.set_xlim(years[0], years[-1])
        _spinup_shade(ax, years)

        if row < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Year', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f2_emissions_prior_vs_posterior.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: f2_emissions_prior_vs_posterior.png")


# ============================================================================
# F5: POSTERIOR ISOTOPE SIGNATURES
# ============================================================================

def plot_f5_isotopes(years, iso_d13c_post, iso_dd_post,
                     prior_iso_d13c, prior_iso_dd, config, output_dir):
    """F5: Prior vs posterior δ¹³C (left) and δD (right) for all sources."""
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    labels = list('abcdefghij')

    for i, src in enumerate(config.sources):
        ax_c = axes[i, 0]
        ax_d = axes[i, 1]
        shade_enso(ax_c); shade_enso(ax_d)

        ax_c.text(0.01, 0.98, f'({labels[2*i]})',   transform=ax_c.transAxes,
                  fontsize=12, fontweight='bold', va='top', ha='left', zorder=200)
        ax_d.text(0.01, 0.98, f'({labels[2*i+1]})', transform=ax_d.transAxes,
                  fontsize=12, fontweight='bold', va='top', ha='left', zorder=200)

        kws_post_sh = dict(color='red',      linewidth=2.0, alpha=0.85, linestyle='-',
                           marker='.', markersize=2)
        kws_post_nh = dict(color='red',      linewidth=1.5, alpha=0.7,  linestyle='--')
        kws_prior_sh = dict(color='darkgrey', linewidth=2.5, alpha=0.7,  linestyle='-',  zorder=50)
        kws_prior_nh = dict(color='darkgrey', linewidth=2.5, alpha=0.7,  linestyle='--', zorder=50)

        ax_c.plot(years, iso_d13c_post[src][:, 0], label='Posterior SH' if i == 0 else '', **kws_post_sh)
        ax_c.plot(years, iso_d13c_post[src][:, 1], label='Posterior NH' if i == 0 else '', **kws_post_nh)
        ax_c.plot(years, prior_iso_d13c[src][:, 0], label='Prior SH' if i == 0 else '', **kws_prior_sh)
        ax_c.plot(years, prior_iso_d13c[src][:, 1], label='Prior NH' if i == 0 else '', **kws_prior_nh)

        ax_d.plot(years, iso_dd_post[src][:, 0], **kws_post_sh)
        ax_d.plot(years, iso_dd_post[src][:, 1], **kws_post_nh)
        ax_d.plot(years, prior_iso_dd[src][:, 0],  **kws_prior_sh)
        ax_d.plot(years, prior_iso_dd[src][:, 1],  **kws_prior_nh)

        ax_c.set_ylabel('δ¹³C (‰)', fontsize=11)
        ax_d.set_ylabel('δD (‰)',   fontsize=11)
        ax_c.set_title(src.capitalize(), fontsize=11, fontweight='bold')
        ax_d.set_title(src.capitalize(), fontsize=11, fontweight='bold')

        for ax in (ax_c, ax_d):
            ax.grid(alpha=0.3)
            _spinup_shade(ax, years)
            ax.set_xlim(years[0], years[-1])
            if i < len(config.sources) - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Year', fontsize=11)
                ax.set_xticks(years[::4])

        if i == 0:
            ax_c.legend(loc='center right', frameon=True,
                        fontsize=FONTSIZE['title'], ncol=2)

    plt.suptitle('Posterior Isotopic Signatures (SH solid, NH dashed)\n'
                 'Red = El Niño | Blue = La Niña',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_dir, 'f5_isotopes.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: f5_isotopes.png")


# ============================================================================
# ENSO ISOTOPE SHIFT BARS
# ============================================================================

def plot_isotope_enso_shifts_diff(years, iso_d13c_post, iso_dd_post, output_dir,
                                  prior_emissions=None,
                                  analysis_year1=1994, analysis_year2=2022):
    """ENSO isotope shifts (El Nino - La Nina) for wetlands and pyrogenic.

    Restricted to the analysis period (1994-2022 by default) to exclude
    spin-up / spin-down years where isotopes are unconstrained.  Global mean
    is emission-weighted when prior_emissions is provided, otherwise a simple
    arithmetic mean is used as fallback.
    """
    # Restrict to analysis period only (consistent with runner / enso_causality)
    year_mask     = (years >= analysis_year1) & (years <= analysis_year2)
    years_sub     = years[year_mask]

    mei_local     = load_mei_data(int(analysis_year1), int(analysis_year2))
    phase         = mei_local.set_index('Year')['ENSO_type'].to_dict()
    el_mask       = np.array([phase.get(int(y)) == 'El Niño' for y in years_sub])
    la_mask       = np.array([phase.get(int(y)) == 'La Niña' for y in years_sub])

    fig, axes    = plt.subplots(2, 2, figsize=(14, 10))
    sources      = ['wetlands', 'pyrogenic']
    src_labels   = ['Wetlands', 'Pyrogenic']
    isotope_sets = [('d13c', iso_d13c_post, 'delta13C (permil)'),
                    ('dd',   iso_dd_post,   'deltaD (permil)')]

    for row, (iso_key, iso_dict, iso_label) in enumerate(isotope_sets):
        for col, (src, src_lbl) in enumerate(zip(sources, src_labels)):
            ax     = axes[row, col]
            shifts = []
            pvals  = []

            for hi in range(2):
                vals    = iso_dict[src][year_mask, hi]
                el_vals = vals[el_mask]
                la_vals = vals[la_mask]
                if len(el_vals) > 0 and len(la_vals) > 0:
                    shifts.append(float(np.mean(el_vals) - np.mean(la_vals)))
                    pvals.append(float(stats.mannwhitneyu(
                        el_vals, la_vals, alternative='two-sided').pvalue))
                else:
                    shifts.append(np.nan); pvals.append(np.nan)

            # Emission-weighted global mean (matches runner compute_enso_causality)
            if prior_emissions is not None and src in prior_emissions:
                E_sh  = prior_emissions[src][year_mask, 0]
                E_nh  = prior_emissions[src][year_mask, 1]
                E_tot = E_sh + E_nh + 1e-30
                iso_sh = iso_dict[src][year_mask, 0]
                iso_nh = iso_dict[src][year_mask, 1]
                with np.errstate(invalid='ignore', divide='ignore'):
                    g_vals = (iso_sh * E_sh + iso_nh * E_nh) / E_tot
            else:
                g_vals = np.mean(
                    np.stack([iso_dict[src][year_mask, 0],
                              iso_dict[src][year_mask, 1]], axis=1),
                    axis=1)

            if el_mask.any() and la_mask.any():
                shifts.append(float(np.mean(g_vals[el_mask]) - np.mean(g_vals[la_mask])))
                pvals.append(float(stats.mannwhitneyu(
                    g_vals[el_mask], g_vals[la_mask], alternative='two-sided').pvalue))
            else:
                shifts.append(np.nan); pvals.append(np.nan)

            colors_bar = ['lightblue', 'lightgreen', 'lightyellow']
            ax.bar(range(3), shifts, color=colors_bar, edgecolor='black', linewidth=1)
            y_range = max(abs(s) for s in shifts if np.isfinite(s)) * 0.08 + 1e-10
            for pos, (sh, pv) in enumerate(zip(shifts, pvals)):
                if np.isfinite(sh):
                    sig = '*' if np.isfinite(pv) and pv < 0.05 else ''
                    va  = 'top' if sh >= 0 else 'bottom'
                    ax.text(pos, sh + (y_range if sh >= 0 else -y_range),
                            f'{sh:.2f}{sig}\np={pv:.3f}' if np.isfinite(pv) else f'{sh:.2f}',
                            ha='center', va=va, fontsize=8)

            ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
            ax.set_xticks(range(3))
            ax.set_xticklabels(['SH', 'NH', 'Global'], fontsize=11)
            ax.set_ylabel(f'{iso_label} Shift\n(El Niño − La Niña)', fontsize=11)
            ax.set_title(f'{src_lbl} {iso_label.split()[0]}',
                         fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Isotopic Shifts During ENSO (El Niño − La Niña)\n'
                 'Bars = posterior mean phase difference, * = Mann-Whitney p<0.05',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'enso_isotope_diff.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: enso_isotope_diff.png")


# ============================================================================
# F4: DETAILED ISOTOPE + SINK DIAGNOSTICS
# ============================================================================

def plot_f4_optimized_timeseries(years, iso_d13c_post, iso_dd_post, sinks_post,
                                 prior_iso_d13c, prior_iso_dd, prior_sinks,
                                 config, output_dir, emis_post):
    """
    F4: 5-row × 2-column diagnostic panel.
      Row 0: δ¹³C posteriors for all sources (SH / NH)
      Row 1: δ¹³C emission-weighted mean (prior vs posterior)
      Row 2: δD posteriors for all sources
      Row 3: δD emission-weighted mean
      Row 4: Sink loss rates (OH, strat, soil)
    """
    fig, axes  = plt.subplots(5, 2, figsize=(16, 16))
    hems       = ['SH', 'NH']
    src_colors = ['darkgreen', 'gold', 'orangered', 'dimgray', 'brown']
    prior_iso  = get_prior_isotopes(config)

    def emission_weighted_mean(post_dict, iso_key, hem_idx):
        """Compute emission-weighted isotope mean (posterior and no-opt prior)."""
        w_post  = np.zeros(len(years))
        w_prior = np.zeros(len(years))
        for k in range(len(years)):
            tot = sum(emis_post[s][k, hem_idx] for s in config.sources) + 1e-10
            w_post[k]  = sum(emis_post[s][k, hem_idx] * post_dict[s][k, hem_idx]
                             for s in config.sources) / tot
            w_prior[k] = sum(emis_post[s][k, hem_idx] * prior_iso[s][iso_key][hem_idx]
                             for s in config.sources) / tot
        return w_post, w_prior

    # Rows 0 & 1: δ¹³C
    for hi in range(2):
        ax = axes[0, hi]; shade_enso(ax)
        for ci, src in enumerate(config.sources):
            vals = iso_d13c_post[src][:, hi]
            ax.plot(years, vals, color=src_colors[ci], linewidth=2.0,
                    linestyle='-' if np.std(vals) > 0.1 else '--',
                    label=src.capitalize())
        ax.set_title(f'δ¹³C Posterior – {hems[hi]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('δ¹³C (‰)'); ax.grid(alpha=0.3); ax.set_xticks([])
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1))

        ax = axes[1, hi]; shade_enso(ax)
        w_post, w_prior = emission_weighted_mean(iso_d13c_post, 'd13c', hi)
        ax.plot(years, w_prior, color='black', linestyle='--', linewidth=2.5, alpha=0.7, label='Avg (no opt)')
        ax.plot(years, w_post,  color='black', linestyle='-',  linewidth=2.5, alpha=0.9, label='Avg (optimised)')
        ax.set_title(f'δ¹³C Weighted Mean – {hems[hi]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('δ¹³C (‰)'); ax.grid(alpha=0.3); ax.set_xticks([])
        ax.legend(fontsize=8)

    # Rows 2 & 3: δD
    for hi in range(2):
        ax = axes[2, hi]; shade_enso(ax)
        for ci, src in enumerate(config.sources):
            vals = iso_dd_post[src][:, hi]
            ax.plot(years, vals, color=src_colors[ci], linewidth=2.0,
                    linestyle='-' if np.std(vals) > 1.0 else '--',
                    label=src.capitalize())
        ax.set_title(f'δD Posterior – {hems[hi]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('δD (‰)'); ax.grid(alpha=0.3); ax.set_xticks([])
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1))

        ax = axes[3, hi]; shade_enso(ax)
        w_post, w_prior = emission_weighted_mean(iso_dd_post, 'dd', hi)
        ax.plot(years, w_prior, color='black', linestyle='--', linewidth=2.5, alpha=0.7, label='Avg (no opt)')
        ax.plot(years, w_post,  color='black', linestyle='-',  linewidth=2.5, alpha=0.9, label='Avg (optimised)')
        ax.set_title(f'δD Weighted Mean – {hems[hi]}', fontsize=12, fontweight='bold')
        ax.set_ylabel('δD (‰)'); ax.grid(alpha=0.3); ax.set_xticks([])
        ax.legend(fontsize=8)

    # Row 4: sink loss rates
    for hi in range(2):
        ax = axes[4, hi]; shade_enso(ax)
        s_oh    = (1/config.lifetime_oh.reshape(1,2))    * (1 + sinks_post['oh']    / 100) * 1e6
        s_strat = (1/config.lifetime_strat.reshape(1,2)) * (1 + sinks_post['strat'] / 100) * 1e6
        s_soil  = (1/config.lifetime_soil.reshape(1,2))  * (1 + sinks_post['soil']  / 100) * 1e6
        ax.plot(years, s_oh[:,  hi], color='red',   linewidth=2, label='OH')
        ax.plot(years, s_strat[:,hi], color='blue',  linewidth=2, label='Stratosphere')
        ax.plot(years, s_soil[:, hi], color='green', linewidth=2, label='Soil')
        ax.set_yscale('log')
        ax.set_ylabel('Sink rate (yr⁻¹ × 10⁻⁶)')
        ax.set_xlabel('Year')
        ax.set_title(f'Loss Rates – {hems[hi]}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f4_optimized_timeseries.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: f4_optimized_timeseries.png")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()