"""
COMPLETE TWO-BOX CH4 INVERSION WITH TIME-VARYING EMISSIONS AND ISOTOPES

This script implements a full atmospheric CH4 inversion that optimizes:
- Time-varying emissions E(t) for 5 sources × 2 hemispheres (45 years)
- Time-varying isotopic signatures δ¹³C(t), δD(t) for each source × hemisphere

State vector per year t:
  x(t) = [E_wet_sh(t), E_wet_nh(t), E_agri_sh(t), E_agri_nh(t), ...,
          δ¹³C_wet_sh(t), δ¹³C_wet_nh(t), ..., δD_wet_sh(t), δD_wet_nh(t), ...]

Total state vector: 45 years × (10 emissions + 10 δ¹³C + 10 δD + 6 sinks) = 45 × 36

Cost function: J(x) = (y - F(x))ᵀ S_y⁻¹ (y - F(x)) + (x - x_a)ᵀ S_a⁻¹ (x - x_a)

TWO-STEP APPROACH:
  Step 1: Optimize emissions only with CH4 observations (tight isotope constraints)
  Step 2: Optimize isotopes with full multi-tracer data (tight CH4 constraints)

Author: Bibhasvata Dasgupta
"""
 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg, stats
from copy import deepcopy
from datetime import datetime
import numpy as np
from scipy import linalg
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
# ============================================================================
# CONFIGURATION
# ============================================================================

class InversionConfig:
    """Configuration for the inversion."""
    
    # Time period
    year1 = 1980
    year2 = 2024
    
    # Sources
    sources = ['wetlands', 'agriculture', 'pyrogenic', 'fossil', 'waste']
    n_sources = 5
    
    # Hemispheres
    hemispheres = ['sh', 'nh']
    n_hemispheres = 2
    
    # Isotopes
    isotopes = ['d13c', 'dd']
    n_isotopes = 2
    
    # Model parameters
    tau_ihex = 0.75
    tg2ppb = 0.7178
    nsteps = 10
    
    # Lifetime (years) for [OH, Stratosphere, Soil]
    lifetime_oh = np.array([9.97, 9.97])
    lifetime_strat = np.array([181, 181])
    lifetime_soil = np.array([188, 188])
    
    # Isotope constants
    cfact_d13c = 0.012372
    cfact_dd = 0.00015576
    
    # KIE values
    kie_oh_d13c = 1.0068
    kie_oh_dd = 1.313
    kie_strat_d13c = 1.0144
    kie_strat_dd = 1.133
    kie_soil_d13c = 1.0201
    kie_soil_dd = 1.083
    
    # Observation uncertainties #[1, 0.01, 0.001 ]   # error of the observations. same as in definePriorError
    obs_sigma_ch4_s0 = 0.1
    obs_sigma_ch4 = 1.0
    obs_sigma_d13c = 0.05              # ← Tighten from 0.05 to 0.01 (give isotopes more weight)
    obs_sigma_dd = 0.5                 # ← Tighten from 0.5 to 0.1

    # ==================== PRIOR UNCERTAINTIES ====================
    prior_emission_error = 0.35  # Base prior error for emissions (35%), scaled by source fraction
    prior_isotope_d13c_error = 2.0
    prior_isotope_dd_error = 10.0
    prior_sink_error = 4.0
    
    # ==================== SMOOTHNESS PARAMETERS ====================
    smoothness_lambda_emissions = 10.0
    smoothness_lambda_d13c = 0.1
    smoothness_lambda_dd = 0.01
    smoothness_lambda_sinks = 1.0
    smoothness_order = 1
    
    # Smoothness order: 1 = first derivative, 2 = second derivative
    smoothness_order = 1  # 2 is usually better (smoother curves)
    
    # Optimization
    n_iterations = 5

# ============================================================================
# DATA LOADING
# ============================================================================

def load_prior_emissions(config): 
    """
    Load prior emission estimates from CAMS inventory.
    
    Returns: dict[source] = array(n_years, 2) for [SH, NH]
    """
    inputdir = (
        '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run/input_data/prior_emissions'
    )
    years = np.arange(config.year1, config.year2 + 1)
    prior_emissions = {}
    
    for source in config.sources:
        try:
            filepath = os.path.join(inputdir, f'v2_{source}.txt')
            data = np.loadtxt(filepath)
            
            time_file = data[:, 0]
            emis_sh = data[:, 1]
            emis_nh = data[:, 2]
            
            # Interpolate to our years
            emis_sh_interp = np.interp(years, time_file, emis_sh)
            emis_nh_interp = np.interp(years, time_file, emis_nh)
            
            prior_emissions[source] = np.column_stack([emis_sh_interp, emis_nh_interp])
            
        except FileNotFoundError:
            print(f"⚠ File not found for {source}, using default values")
            # Default: 100 Tg/yr per hemisphere
            prior_emissions[source] = np.ones((len(years), 2)) * 100.0
    
    return prior_emissions

def load_observations(config):
    """
    Load atmospheric observations (CH4, δ¹³C, δD) and apply latitudinal gradients.
    
    Returns: dict with 'ch4', 'd13c', 'dd', each containing 'sh' and 'nh' time series
    """
    obs_dir = (
        '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run/input_data/observations'
    )
    years = np.arange(config.year1, config.year2 + 1)
    observations = {}
    
    tracers = [('ch4', 'ch4'), ('d13c-ch4', 'd13c'), ('dd-ch4', 'dd')]
    
    # Latitudinal gradient coefficients
    lat_gradients = {
        'ch4': {'sh': 4.811, 'nh': -39.018},
        'd13c': {'sh': -0.018, 'nh': 0.178},
        'dd': {'sh': -2.0808, 'nh': 4.315}
    }
    
    for file_tracer, key_tracer in tracers:
        observations[key_tracer] = {}
        
        for hem in config.hemispheres:
            try:
                filepath = os.path.join(obs_dir, f'{hem.upper()}_{file_tracer}.txt')
                data = np.loadtxt(filepath)
                
                time_obs = data[:, 0]
                values = data[:, -1]
                
                # Get data for our period
                mask = (time_obs >= config.year1) & (time_obs < config.year2 + 1)
                time_obs = time_obs[mask]
                values = values[mask]
                
                # Annual average
                annual_values = []
                for year in years:
                    year_mask = (time_obs >= year) & (time_obs < year + 1)
                    if np.sum(year_mask) > 0:
                        annual_values.append(np.mean(values[year_mask]))
                    else:
                        annual_values.append(np.nan)
                
                # Fill NaN with interpolation
                annual_values = np.array(annual_values)
                if np.any(np.isnan(annual_values)):
                    valid_idx = ~np.isnan(annual_values)
                    annual_values[~valid_idx] = np.interp(
                        years[~valid_idx], years[valid_idx], annual_values[valid_idx]
                    )
                
                # Apply latitudinal gradient
                gradient = lat_gradients[key_tracer][hem]# * getattr(config, 'lat_gradient_multiplier', 1.0)
                annual_values += gradient
                
                observations[key_tracer][hem] = annual_values
                
            except FileNotFoundError:
                print(f"⚠ Observation file not found: {hem.upper()}_{file_tracer}.txt")
                # Use dummy values with gradient
                default = 1800.0 if key_tracer == 'ch4' else 0.0
                gradient = lat_gradients[key_tracer][hem]# * getattr(config, 'lat_gradient_multiplier', 1.0)
                observations[key_tracer][hem] = np.ones(len(years)) * (default + gradient)
    
    return observations

def get_prior_isotopes(config):
    """
    Get prior isotopic signatures for each source.
    
    Returns: dict[source][isotope] = [sh_value, nh_value]
    """
    prior_isotopes = {
        'wetlands': {
            'd13c': [-58.2, -64.8],
            'dd': [-299, -336]
        },
        'agriculture': {
            'd13c': [-63.4, -63.3],
            'dd': [-312, -314]
        },
        'pyrogenic': {
            'd13c': [-22.3, -22.4],
            'dd': [-213, -183]
        },
        'fossil': {
            'd13c': [-44, -44],
            'dd': [-192, -191]
        },
        'waste': {
            'd13c': [-54.5, -54.5],
            'dd': [-292.2, -292.2]
        }
    }
    
    return prior_isotopes

# ============================================================================
# STATE VECTOR MANAGEMENT
# ============================================================================

class StateVector:
    """Handles packing/unpacking of the state vector."""
    
    def __init__(self, config):
        self.config = config
        self.n_years = config.year2 - config.year1 + 1
        
        # State vector structure per year:
        # [E_wet_sh, E_wet_nh, E_agri_sh, E_agri_nh, E_pyro_sh, E_pyro_nh, 
        #  E_fossil_sh, E_fossil_nh, E_waste_sh, E_waste_nh,
        #  d13C_wet_sh, d13C_wet_nh, ..., d13C_waste_sh, d13C_waste_nh,
        #  dD_wet_sh, dD_wet_nh, ..., dD_waste_sh, dD_waste_nh,
        #  sink_OH_sh, sink_OH_nh, sink_strat_sh, sink_strat_nh, sink_soil_sh, sink_soil_nh]
        
        self.n_emissions = config.n_sources * config.n_hemispheres  # 10
        self.n_isotopes_per_type = config.n_sources * config.n_hemispheres  # 10
        self.n_sinks = 3 * config.n_hemispheres  # 6
        
        self.n_params_per_year = self.n_emissions + 2 * self.n_isotopes_per_type + self.n_sinks  # 36
        self.n_total = self.n_params_per_year * self.n_years
        
        print(f"State vector: {self.n_years} years × {self.n_params_per_year} params/year = {self.n_total} total")
    
    def pack(self, emissions, isotopes_d13c, isotopes_dd, sinks):
        """
        Pack dictionaries into state vector.
        
        Args:
            emissions: dict[source] = array(n_years, 2)
            isotopes_d13c: dict[source] = array(n_years, 2)
            isotopes_dd: dict[source] = array(n_years, 2)
            sinks: dict[sink_type] = array(n_years, 2)
        
        Returns:
            x: state vector array(n_total,)
        """
        x = np.zeros(self.n_total)
        
        for year_idx in range(self.n_years):
            offset = year_idx * self.n_params_per_year
            
            # Emissions
            idx = 0
            for source in self.config.sources:
                x[offset + idx] = emissions[source][year_idx, 0]  # SH
                x[offset + idx + 1] = emissions[source][year_idx, 1]  # NH
                idx += 2
            
            # δ¹³C
            for source in self.config.sources:
                x[offset + idx] = isotopes_d13c[source][year_idx, 0]
                x[offset + idx + 1] = isotopes_d13c[source][year_idx, 1]
                idx += 2
            
            # δD
            for source in self.config.sources:
                x[offset + idx] = isotopes_dd[source][year_idx, 0]
                x[offset + idx + 1] = isotopes_dd[source][year_idx, 1]
                idx += 2
            
            # Sinks (as % perturbations)
            for sink in ['oh', 'strat', 'soil']:
                x[offset + idx] = sinks[sink][year_idx, 0]
                x[offset + idx + 1] = sinks[sink][year_idx, 1]
                idx += 2
        
        return x
    
    def unpack(self, x):
        """
        Unpack state vector into dictionaries.
        
        Returns: emissions, isotopes_d13c, isotopes_dd, sinks
        """
        emissions = {source: np.zeros((self.n_years, 2)) for source in self.config.sources}
        isotopes_d13c = {source: np.zeros((self.n_years, 2)) for source in self.config.sources}
        isotopes_dd = {source: np.zeros((self.n_years, 2)) for source in self.config.sources}
        sinks = {sink: np.zeros((self.n_years, 2)) for sink in ['oh', 'strat', 'soil']}
        
        for year_idx in range(self.n_years):
            offset = year_idx * self.n_params_per_year
            
            # Emissions
            idx = 0
            for source in self.config.sources:
                emissions[source][year_idx, 0] = x[offset + idx]
                emissions[source][year_idx, 1] = x[offset + idx + 1]
                idx += 2
            
            # δ¹³C
            for source in self.config.sources:
                isotopes_d13c[source][year_idx, 0] = x[offset + idx]
                isotopes_d13c[source][year_idx, 1] = x[offset + idx + 1]
                idx += 2
            
            # δD
            for source in self.config.sources:
                isotopes_dd[source][year_idx, 0] = x[offset + idx]
                isotopes_dd[source][year_idx, 1] = x[offset + idx + 1]
                idx += 2
            
            # Sinks
            for sink in ['oh', 'strat', 'soil']:
                sinks[sink][year_idx, 0] = x[offset + idx]
                sinks[sink][year_idx, 1] = x[offset + idx + 1]
                idx += 2
        
        return emissions, isotopes_d13c, isotopes_dd, sinks

# ============================================================================
# FORWARD MODEL
# ============================================================================

class ForwardModel:
    """Two-box atmospheric transport and chemistry model."""
    
    def __init__(self, config):
        self.config = config
        self.dt = 1.0 / config.nsteps
    
    def d13c_to_mixing_ratio(self, ch4, delta):
        """Convert δ¹³C (‰) to mixing ratio."""
        return ch4 * (delta/1000 + 1) * self.config.cfact_d13c / (1 + (delta/1000 + 1) * self.config.cfact_d13c)
    
    def mixing_ratio_to_d13c(self, c13h4, ch4):
        """Convert mixing ratio to δ¹³C (‰)."""
        return ((c13h4 / (ch4 - c13h4)) / self.config.cfact_d13c - 1) * 1000
    
    def dd_to_mixing_ratio(self, ch4, delta):
        """Convert δD (‰) to mixing ratio."""
        return ch4 * (delta/1000 + 1) * self.config.cfact_dd / (1 + (delta/1000 + 1) * self.config.cfact_dd)
    
    def mixing_ratio_to_dd(self, dch3, ch4):
        """Convert mixing ratio to δD (‰)."""
        return ((dch3 / (ch4 - dch3)) / self.config.cfact_dd - 1) * 1000
    
    def run(self, emissions, isotopes_d13c, isotopes_dd, sinks, init_conditions):
        """
        Run forward model.
        
        Args:
            emissions: dict[source] = array(n_years, 2)
            isotopes_d13c: dict[source] = array(n_years, 2)
            isotopes_dd: dict[source] = array(n_years, 2)
            sinks: dict[sink_type] = array(n_years, 2)
            init_conditions: dict with 'ch4', 'd13c', 'dd' as arrays[2] for SH, NH
        
        Returns:
            results: dict with annual 'ch4', 'd13c', 'dd' arrays(n_years, 2)
        """
        n_years = self.config.year2 - self.config.year1 + 1
        
        # Initialize
        ch4 = np.copy(init_conditions['ch4'])  # [SH, NH]
        c13h4 = self.d13c_to_mixing_ratio(ch4, init_conditions['d13c'])
        dch3 = self.dd_to_mixing_ratio(ch4, init_conditions['dd'])
        
        # Output storage
        ch4_out = np.zeros((n_years, 2))
        d13c_out = np.zeros((n_years, 2))
        dd_out = np.zeros((n_years, 2))
        
        for year_idx in range(n_years):
            # Get this year's fluxes
            total_emis = np.zeros(2)  # [SH, NH]
            weighted_d13c = np.zeros(2)
            weighted_dd = np.zeros(2)
            
            for source in self.config.sources:
                emis = emissions[source][year_idx, :]
                total_emis += emis
                weighted_d13c += emis * isotopes_d13c[source][year_idx, :]
                weighted_dd += emis * isotopes_dd[source][year_idx, :]
            
            # Weighted mean isotopes
            mean_d13c = weighted_d13c / (total_emis + 1e-10)
            mean_dd = weighted_dd / (total_emis + 1e-10)
            
            # Convert to absolute mixing ratios
            c13_emission = self.d13c_to_mixing_ratio(total_emis, mean_d13c)
            d_emission = self.dd_to_mixing_ratio(total_emis, mean_dd)
            
            # Sinks
            sink_oh = 1 / self.config.lifetime_oh * (1 + sinks['oh'][year_idx, :] / 100)
            sink_strat = 1 / self.config.lifetime_strat * (1 + sinks['strat'][year_idx, :] / 100)
            sink_soil = 1 / self.config.lifetime_soil * (1 + sinks['soil'][year_idx, :] / 100)
            
            # Sub-annual time stepping
            for step in range(self.config.nsteps):
                # Emissions
                ch4 += total_emis * self.dt * self.config.tg2ppb
                c13h4 += c13_emission * self.dt * self.config.tg2ppb
                dch3 += d_emission * self.dt * self.config.tg2ppb
                
                # Transport (interhemispheric exchange)
                flux = 0.5 * self.dt / self.config.tau_ihex * (ch4[1] - ch4[0])
                ch4[0] += flux
                ch4[1] -= flux
                
                flux_c13 = 0.5 * self.dt / self.config.tau_ihex * (c13h4[1] - c13h4[0])
                c13h4[0] += flux_c13
                c13h4[1] -= flux_c13
                
                flux_d = 0.5 * self.dt / self.config.tau_ihex * (dch3[1] - dch3[0])
                dch3[0] += flux_d
                dch3[1] -= flux_d
                
                # Chemistry (sinks with KIE)
                ch4 -= self.dt * (sink_oh + sink_strat + sink_soil) * ch4
                c13h4 -= self.dt * (sink_oh / self.config.kie_oh_d13c + 
                                   sink_strat / self.config.kie_strat_d13c + 
                                   sink_soil / self.config.kie_soil_d13c) * c13h4
                dch3 -= self.dt * (sink_oh / self.config.kie_oh_dd + 
                                  sink_strat / self.config.kie_strat_dd + 
                                  sink_soil / self.config.kie_soil_dd) * dch3
            
            # Store annual mean
            ch4_out[year_idx, :] = ch4
            d13c_out[year_idx, :] = self.mixing_ratio_to_d13c(c13h4, ch4)
            dd_out[year_idx, :] = self.mixing_ratio_to_dd(dch3, ch4)
        
        return {'ch4': ch4_out, 'd13c': d13c_out, 'dd': dd_out}

# ============================================================================
# PRIOR COVARIANCE WITH SOURCE SCALING
# ============================================================================

def two_step_construct_prior_covariance_wo_smoothness(config, state_vec, prior_emissions, prior_isotopes):
    """
    Construct prior error covariance matrix for two-step inversion.
    
    Observation uncertainties (sigma) directly control parameter uncertainty weighting:
    - CH4: 1.0 ppb (tightest constraint on emissions)
    - δ13C: 0.01‰ (looser constraint on isotopes)
    - δD: 0.001‰ (loosest constraint on isotopes)
    
    This ensures CH4 data dominates the inversion (as it should due to better quality),
    while isotope data provides secondary constraints.
    
    Args:
        config: InversionConfig object
        state_vec: StateVector object
        prior_emissions: dict[source] = array(n_years, 2) for [SH, NH]
        prior_isotopes: dict[source][isotope] = [sh_value, nh_value]
    
    Returns:
        Sa: Diagonal prior covariance matrix (n_total × n_total)
    """
    n_total = state_vec.n_total
    n_years = state_vec.n_years
    n_params_per_year = state_vec.n_params_per_year
    
    # Calculate source contributions for emission scaling
    total_emissions_by_source = {}
    for source in config.sources:
        total_emissions_by_source[source] = np.mean(
            prior_emissions[source][:, 0] + prior_emissions[source][:, 1]
        )
    
    global_total = sum(total_emissions_by_source.values())
    fractions = {source: total_emissions_by_source[source] / global_total 
                 for source in config.sources}
    
    print(f"Source fractions: {', '.join([f'{s}: {fractions[s]:.2f}' for s in config.sources])}")
    
    # Build diagonal covariance
    Sa_diag = np.zeros(n_total)
    
    for year_idx in range(n_years):
        offset = year_idx * n_params_per_year
        
        # === EMISSIONS ===
        # Emissions prior error is based on config and source fraction
        idx = 0
        for source in config.sources:
            scale_factor = 1.0 - fractions[source]
            base_error = config.prior_emission_error  # 0.3 (30%)
            
            for hem_idx in range(2):
                prior_val = prior_emissions[source][year_idx, hem_idx]
                Sa_diag[offset + idx] = (base_error * scale_factor * prior_val) ** 2
                #print(f"Year {config.year1 + year_idx}, Source {source}, Hem {hem_idx}: Emission prior error = {np.sqrt(Sa_diag[offset + idx]):.2f} Tg/yr")
                idx += 1
        
        # === δ13C ISOTOPIC SIGNATURES ===
        # Prior isotope errors scale inversely with observation confidence
        for source in config.sources:
            for hem_idx in range(2):
                # obs_sigma_d13c = 0.01‰, so isotope prior is looser
                Sa_diag[offset + idx] = config.prior_isotope_d13c_error ** 2  # (1.0)^2
                #print(f"Year {config.year1 + year_idx}, Source {source}, Hem {hem_idx}: δ13C prior error = {np.sqrt(Sa_diag[offset + idx]):.2f} ‰")
                idx += 1
        
        # === δD ISOTOPIC SIGNATURES ===
        for source in config.sources:
            for hem_idx in range(2):
                # obs_sigma_dd = 0.001‰, so isotope prior is much looser
                Sa_diag[offset + idx] = config.prior_isotope_dd_error ** 2  # (10.0)^2
                #print(f"Year {config.year1 + year_idx}, Source {source}, Hem {hem_idx}: δD prior error = {np.sqrt(Sa_diag[offset + idx]):.2f} ‰")
                idx += 1
        
        # === SINKS (OH, Strat, Soil) ===
        for _ in range(6):  # 3 sinks × 2 hemispheres
            Sa_diag[offset + idx] = config.prior_sink_error ** 2  # (10.0)^2
            #print(f"Year {config.year1 + year_idx}: Sink prior error = {np.sqrt(Sa_diag[offset + idx]):.2f} %")
            idx += 1
    
    return np.diag(Sa_diag)

def two_step_construct_prior_covariance(config, state_vec, prior_emissions, prior_isotopes):
    """
    Construct diagonal prior error covariance matrix.
    
    Temporal smoothness is now handled separately via penalty method,
    so this returns a simple diagonal matrix.
    """
    n_total = state_vec.n_total
    n_years = state_vec.n_years
    n_params_per_year = state_vec.n_params_per_year
    
    # Calculate source contributions for scaling
    total_emissions_by_source = {}
    for source in config.sources:
        total_emissions_by_source[source] = np.mean(
            prior_emissions[source][:, 0] + prior_emissions[source][:, 1]
        )
    
    global_total = sum(total_emissions_by_source.values())
    fractions = {source: total_emissions_by_source[source] / global_total 
                 for source in config.sources}
    
    print(f"Source fractions: {', '.join([f'{s}: {fractions[s]:.2f}' for s in config.sources])}")
    
    # Build diagonal covariance
    Sa_diag = np.zeros(n_total)
    
    for year_idx in range(n_years):
        offset = year_idx * n_params_per_year
        
        # === EMISSIONS ===
        idx = 0
        for source in config.sources:
            scale_factor = 1.0 - fractions[source]
            base_error = config.prior_emission_error
            
            for hem_idx in range(2):
                prior_val = prior_emissions[source][year_idx, hem_idx]
                Sa_diag[offset + idx] = (base_error * scale_factor * prior_val) ** 2
                idx += 1
        
        # === δ13C ISOTOPIC SIGNATURES ===
        for source in config.sources:
            for hem_idx in range(2):
                Sa_diag[offset + idx] = config.prior_isotope_d13c_error ** 2
                idx += 1
        
        # === δD ISOTOPIC SIGNATURES ===
        for source in config.sources:
            for hem_idx in range(2):
                Sa_diag[offset + idx] = config.prior_isotope_dd_error ** 2
                idx += 1
        
        # === SINKS ===
        for _ in range(6):
            Sa_diag[offset + idx] = config.prior_sink_error ** 2
            idx += 1
    
    return np.diag(Sa_diag)

def construct_temporal_prior_covariance(config, state_vec, prior_emissions, prior_isotopes):
    """
    Construct prior error covariance with temporal smoothness constraints.
    This modification adds off-diagonal elements to the prior covariance matrix
    to penalize year-to-year changes in emissions, creating smoother time series.
    
    Key changes from original:
    1. Off-diagonal correlations between consecutive years
    2. Stronger penalty on second-order differences (acceleration)
    3. Different smoothness for emissions vs isotopes
    
    Args:
        config: InversionConfig object (must have temporal smoothness params)
        state_vec: StateVector object
        prior_emissions: dict[source] = array(n_years, 2)
        prior_isotopes: dict[source][isotope] = [sh_value, nh_value]
    
    Returns:
        Sa: Prior covariance matrix with temporal correlations (n_total × n_total)
    """
    n_total = state_vec.n_total
    n_years = state_vec.n_years
    n_params_per_year = state_vec.n_params_per_year
    
    # Source fractions for emission scaling
    total_emissions_by_source = {}
    for source in config.sources:
        total_emissions_by_source[source] = np.mean(
            prior_emissions[source][:, 0] + prior_emissions[source][:, 1]
        )
    
    global_total = sum(total_emissions_by_source.values())
    fractions = {source: total_emissions_by_source[source] / global_total 
                 for source in config.sources}
    
    print(f"\n=== Temporal Smoothness Configuration ===")
    print(f"  Emission correlation: {config.temporal_correlation_emissions}")
    print(f"  Isotope correlation: {config.temporal_correlation_isotopes}")
    print(f"  Smoothness weight: {config.smoothness_weight}")
    
    # Initialize full covariance matrix (not just diagonal)
    Sa = np.zeros((n_total, n_total))
    
    # Build diagonal variances first
    for year_idx in range(n_years):
        offset = year_idx * n_params_per_year
        
        # === EMISSIONS ===
        idx = 0
        for source in config.sources:
            scale_factor = 1.0 - fractions[source]
            base_error = config.prior_emission_error
            
            for hem_idx in range(2):
                prior_val = prior_emissions[source][year_idx, hem_idx]
                variance = (base_error * scale_factor * prior_val) ** 2
                Sa[offset + idx, offset + idx] = variance
                idx += 1
        
        # === δ13C ISOTOPES ===
        for source in config.sources:
            for hem_idx in range(2):
                Sa[offset + idx, offset + idx] = config.prior_isotope_d13c_error ** 2
                idx += 1
        
        # === δD ISOTOPES ===
        for source in config.sources:
            for hem_idx in range(2):
                Sa[offset + idx, offset + idx] = config.prior_isotope_dd_error ** 2
                idx += 1
        
        # === SINKS ===
        for _ in range(6):
            Sa[offset + idx, offset + idx] = config.prior_sink_error ** 2
            idx += 1
    
    # === ADD TEMPORAL CORRELATIONS (OFF-DIAGONAL) ===
    # This creates smooth time series by correlating consecutive years
    
    for year_idx in range(n_years - 1):
        offset_current = year_idx * n_params_per_year
        offset_next = (year_idx + 1) * n_params_per_year
        
        # EMISSIONS: Apply temporal correlation
        for param_idx in range(10):  # 5 sources × 2 hemispheres
            var_current = Sa[offset_current + param_idx, offset_current + param_idx]
            var_next = Sa[offset_next + param_idx, offset_next + param_idx]
            
            # Covariance = correlation × sqrt(var1 × var2)
            covariance = (config.temporal_correlation_emissions * 
                         np.sqrt(var_current * var_next))
            
            # Set symmetric off-diagonal elements
            Sa[offset_current + param_idx, offset_next + param_idx] = covariance
            Sa[offset_next + param_idx, offset_current + param_idx] = covariance
        
        # ISOTOPES: Stronger temporal correlation (slower changes expected)
        for param_idx in range(10, 30):  # δ13C and δD (10 + 10 parameters)
            var_current = Sa[offset_current + param_idx, offset_current + param_idx]
            var_next = Sa[offset_next + param_idx, offset_next + param_idx]
            
            covariance = (config.temporal_correlation_isotopes * 
                         np.sqrt(var_current * var_next))
            
            Sa[offset_current + param_idx, offset_next + param_idx] = covariance
            Sa[offset_next + param_idx, offset_current + param_idx] = covariance
        
        # SINKS: Moderate correlation
        for param_idx in range(30, 36):  # 6 sink parameters
            var_current = Sa[offset_current + param_idx, offset_current + param_idx]
            var_next = Sa[offset_next + param_idx, offset_next + param_idx]
            
            covariance = (0.85 * np.sqrt(var_current * var_next))  # Fixed 0.85 correlation
            
            Sa[offset_current + param_idx, offset_next + param_idx] = covariance
            Sa[offset_next + param_idx, offset_current + param_idx] = covariance
    
    # === OPTIONAL: SECOND-ORDER SMOOTHNESS (PENALIZE ACCELERATION) ===
    # This prevents sharp turns in time series
    if hasattr(config, 'smoothness_weight') and config.smoothness_weight > 0:
        print(f"  Applying second-order smoothness penalty (weight={config.smoothness_weight})")
        
        for year_idx in range(1, n_years - 1):
            offset_prev = (year_idx - 1) * n_params_per_year
            offset_curr = year_idx * n_params_per_year
            offset_next = (year_idx + 1) * n_params_per_year
            
            # Only apply to emissions (first 10 parameters)
            for param_idx in range(10):
                # Second derivative penalty: (x[t-1] - 2*x[t] + x[t+1])²
                # This is implemented by adjusting the diagonal
                var_current = Sa[offset_curr + param_idx, offset_curr + param_idx]
                
                # Increase variance slightly to allow flexibility
                # but penalize through the cost function structure
                penalty = config.smoothness_weight * var_current
                Sa[offset_curr + param_idx, offset_curr + param_idx] += penalty
    
    # Ensure positive definiteness
    eigenvalues = np.linalg.eigvalsh(Sa)
    min_eigenvalue = np.min(eigenvalues)
    
    if min_eigenvalue < 1e-10:
        print(f"  ⚠ Warning: Near-singular covariance (min eigenvalue: {min_eigenvalue:.2e})")
        print(f"    Adding regularization...")
        Sa += np.eye(n_total) * 1e-8
    
    print(f"  ✓ Covariance matrix constructed: {n_total}×{n_total}")
    print(f"    Condition number: {np.linalg.cond(Sa):.2e}")
    
    return Sa

def compute_temporal_statistics(years, emissions, source):
    """
    Compute statistics to quantify temporal variability.
    
    Returns:
        - mean_annual_change: Average |E[t] - E[t-1]| / E[t-1]
        - max_annual_change: Maximum year-to-year change
        - std_of_changes: Std dev of year-to-year changes

    TARGET METRICS:
        - Mean annual emission change: <10% (currently probably 15-25%)
        - Max annual change: <25% (currently probably 40-60%)
        - RMSE increase: <10% from baseline
    """
    total_emis = emissions[source][:, 0] + emissions[source][:, 1]
    
    # Year-to-year fractional changes
    changes = np.diff(total_emis) / total_emis[:-1]
    
    stats = {
        'mean_abs_change': np.mean(np.abs(changes)) * 100,  # %
        'max_abs_change': np.max(np.abs(changes)) * 100,
        'std_of_changes': np.std(changes) * 100
    }
    
    return stats

def print_smoothness_diagnostics(years, prior_emissions, posterior_emissions, config):
    """
    Print before/after smoothness comparison.
    """
    print("\n" + "="*70)
    print("TEMPORAL SMOOTHNESS DIAGNOSTICS")
    print("="*70)
    
    for source in config.sources:
        prior_stats = compute_temporal_statistics(years, prior_emissions, source)
        post_stats = compute_temporal_statistics(years, posterior_emissions, source)
        
        print(f"\n{source.upper()}")
        print(f"  Mean annual change: {prior_stats['mean_abs_change']:.1f}% → "
              f"{post_stats['mean_abs_change']:.1f}% "
              f"({'✓ smoother' if post_stats['mean_abs_change'] < prior_stats['mean_abs_change'] else '⚠ less smooth'})")
        print(f"  Max annual change:  {prior_stats['max_abs_change']:.1f}% → "
              f"{post_stats['max_abs_change']:.1f}%")
        print(f"  Std of changes:     {prior_stats['std_of_changes']:.1f}% → "
              f"{post_stats['std_of_changes']:.1f}%")

# ============================================================================
# OBSERVATION COVARIANCE
# ============================================================================

def get_observation_covariance(config, n_years, ch4_only=False):
    """
    Construct observation error covariance.
    
    Args:
        ch4_only: If True, only use CH4 observations (step 1)
    """
    if ch4_only:
        # Step 1: Only CH4 observations
        n_obs_per_year = 2  # CH4_SH, CH4_NH
        n_total = n_obs_per_year * n_years
        errors = [config.obs_sigma_ch4_s0] * n_total
    else:
        # Step 2: All tracers
        n_obs_per_year = 6  # 3 tracers × 2 hemispheres
        n_total = n_obs_per_year * n_years
        
        errors = []
        for tracer in ['ch4', 'd13c', 'dd']:
            sigma = {'ch4': config.obs_sigma_ch4, 'd13c': config.obs_sigma_d13c, 'dd': config.obs_sigma_dd}[tracer]
            errors.extend([sigma, sigma] * n_years)  # SH, NH
    
    return np.diag(np.array(errors) ** 2)

def get_observation_vector(observations, ch4_only=False):
    """Stack observations into vector."""
    y_obs = []
    
    if ch4_only:
        # Step 1: Only CH4
        for hem in ['sh', 'nh']:
            y_obs.extend(observations['ch4'][hem])
    else:
        # Step 2: All tracers
        for tracer in ['ch4', 'd13c', 'dd']:
            for hem in ['sh', 'nh']:
                y_obs.extend(observations[tracer][hem])
    
    return np.array(y_obs)

# ============================================================================
# JACOBIAN (Finite Differences)
# ============================================================================
def compute_jacobian(config, state_vec, forward_model, x, init_conditions, y_ref, ch4_only=False):
    """
    Compute Jacobian matrix using finite differences.
    
    Now includes gradients for:
    - Emissions (always computed)
    - Isotopic signatures δ13C and δD (NEWLY ENABLED)
    - Sinks (always computed)
    
    Args:
        config: InversionConfig object
        state_vec: StateVector object
        forward_model: ForwardModel object
        x: Current state vector
        init_conditions: Initial atmospheric state
        y_ref: Reference model output (observations operator applied)
        ch4_only: If True, only use CH4 obs (Step 1); else all tracers (Step 2)
    
    Returns:
        K: Jacobian matrix (n_obs × n_state)
    """
    n_state = len(x)
    n_obs = len(y_ref)
    n_years = state_vec.n_years
    n_params_per_year = state_vec.n_params_per_year
    
    K = np.zeros((n_obs, n_state))
    
    # Parameter counts
    n_emissions = 10  # 5 sources × 2 hemispheres
    n_isotopes_d13c = 10  # 5 sources × 2 hemispheres
    n_isotopes_dd = 10  # 5 sources × 2 hemispheres
    n_sinks = 6  # 3 sinks × 2 hemispheres
    
    print("  Computing Jacobian...")
    print(f"    State dimension: {n_state} ({n_years} years × {n_params_per_year} params/year)")
    print(f"    Observation dimension: {n_obs}")
    print(f"    Computing gradients for: Emissions + δ13C + δD + Sinks")
    
    # === DIAGNOSTIC: Check which parameters actually vary ===
    print("\n    Checking parameter variability...")
    emis_ref, iso_d13c_ref, iso_dd_ref, sinks_ref = state_vec.unpack(x)
    
    for source in config.sources:
        d13c_std = np.std(iso_d13c_ref[source])
        dd_std = np.std(iso_dd_ref[source])
        print(f"      {source:12s}: δ13C std={d13c_std:.3f}‰, δD std={dd_std:.2f}‰")
    
    # Perturbation sizes
    emis_pert_frac = 0.01  # 1% relative perturbation for emissions
    d13c_pert_abs = 0.1    # 0.1‰ absolute perturbation for δ13C
    dd_pert_abs = 1.0      # 1.0‰ absolute perturbation for δD
    sink_pert_abs = 0.1    # 0.1% perturbation for sinks
    
    # Loop over years
    for year_idx in range(n_years):
        if year_idx % 5 == 0:
            progress = 100 * year_idx / n_years
            print(f"    Year {config.year1 + year_idx} ({progress:.0f}% complete)...")
        
        offset = year_idx * n_params_per_year
        
        # ========================================================================
        # 1. EMISSIONS (indices 0-9)
        # ========================================================================
        for i in range(n_emissions):
            param_idx = offset + i
            
            x_pert = x.copy()
            # Adaptive perturbation (at least 1e-6 to avoid division by zero)
            pert_size = max(abs(x[param_idx]) * emis_pert_frac, 1e-6)
            x_pert[param_idx] += pert_size
            
            emis, iso_d13c, iso_dd, sinks = state_vec.unpack(x_pert)
            output_pert = forward_model.run(emis, iso_d13c, iso_dd, sinks, init_conditions)
            y_pert = observation_operator(output_pert, None, ch4_only)
            
            K[:, param_idx] = (y_pert - y_ref) / pert_size
        
        # ========================================================================
        # 2. δ13C ISOTOPIC SIGNATURES (indices 10-19) ← NEWLY ENABLED
        # ========================================================================
        if not ch4_only:  # Only in Step 2 (multi-tracer optimization)
            for i in range(n_isotopes_d13c):
                param_idx = offset + n_emissions + i
                
                x_pert = x.copy()
                x_pert[param_idx] += d13c_pert_abs  # Fixed 0.1‰ perturbation
                
                emis, iso_d13c, iso_dd, sinks = state_vec.unpack(x_pert)
                output_pert = forward_model.run(emis, iso_d13c, iso_dd, sinks, init_conditions)
                y_pert = observation_operator(output_pert, None, ch4_only)
                
                K[:, param_idx] = (y_pert - y_ref) / d13c_pert_abs
        
        # ========================================================================
        # 3. δD ISOTOPIC SIGNATURES (indices 20-29) ← NEWLY ENABLED
        # ========================================================================
        if not ch4_only:  # Only in Step 2
            for i in range(n_isotopes_dd):
                param_idx = offset + n_emissions + n_isotopes_d13c + i
                
                x_pert = x.copy()
                x_pert[param_idx] += dd_pert_abs  # Fixed 1.0‰ perturbation
                
                emis, iso_d13c, iso_dd, sinks = state_vec.unpack(x_pert)
                output_pert = forward_model.run(emis, iso_d13c, iso_dd, sinks, init_conditions)
                y_pert = observation_operator(output_pert, None, ch4_only)
                
                K[:, param_idx] = (y_pert - y_ref) / dd_pert_abs
        
        # ========================================================================
        # 4. SINKS (indices 30-35)
        # ========================================================================
        for i in range(n_sinks):
            param_idx = offset + n_emissions + n_isotopes_d13c + n_isotopes_dd + i
            
            x_pert = x.copy()
            # At least 0.1% perturbation for sinks
            pert_size = max(abs(x[param_idx]) * emis_pert_frac, sink_pert_abs)
            x_pert[param_idx] += pert_size
            
            emis, iso_d13c, iso_dd, sinks = state_vec.unpack(x_pert)
            output_pert = forward_model.run(emis, iso_d13c, iso_dd, sinks, init_conditions)
            y_pert = observation_operator(output_pert, None, ch4_only)
            
            K[:, param_idx] = (y_pert - y_ref) / pert_size
    
    # === DIAGNOSTIC OUTPUT ===
    print(f"\n    ✓ Jacobian computed: {K.shape}")
    
    # Check for zero columns (parameters with no impact)
    zero_cols = np.sum(np.abs(K), axis=0) < 1e-15
    n_zero = np.sum(zero_cols)
    if n_zero > 0:
        print(f"    ⚠ Warning: {n_zero} parameters have zero gradient (no impact on observations)")
        
        # Identify which parameters
        zero_indices = np.where(zero_cols)[0]
        for idx in zero_indices[:5]:  # Show first 5
            year_idx = idx // n_params_per_year
            param_in_year = idx % n_params_per_year
            
            if param_in_year < n_emissions:
                param_type = f"Emission {param_in_year}"
            elif param_in_year < n_emissions + n_isotopes_d13c:
                param_type = f"δ13C {param_in_year - n_emissions}"
            elif param_in_year < n_emissions + n_isotopes_d13c + n_isotopes_dd:
                param_type = f"δD {param_in_year - n_emissions - n_isotopes_d13c}"
            else:
                param_type = f"Sink {param_in_year - n_emissions - n_isotopes_d13c - n_isotopes_dd}"
            
            print(f"      - Year {config.year1 + year_idx}, {param_type}")
    
    # Check conditioning
    K_norm = np.linalg.norm(K, axis=0)
    print(f"    Gradient norms: min={K_norm[K_norm>0].min():.2e}, "
          f"max={K_norm.max():.2e}, ratio={K_norm.max()/K_norm[K_norm>0].min():.2e}")
    
    return K

def observation_operator(model_output, observations, ch4_only=False):
    """
    Sample model at observation locations.
    
    Args:
        model_output: dict with 'ch4', 'd13c', 'dd' arrays (n_years, 2)
        observations: Not used (kept for API compatibility)
        ch4_only: If True, only return CH4; else return all tracers
    
    Returns:
        y_model: Stacked observation vector
    """
    y_model = []
    
    if ch4_only:
        # Step 1: Only CH4 (SH, NH for each year)
        for hem in ['sh', 'nh']:
            hem_idx = 0 if hem == 'sh' else 1
            y_model.extend(model_output['ch4'][:, hem_idx])
    else:
        # Step 2: All tracers (CH4, δ13C, δD) × (SH, NH)
        for tracer in ['ch4', 'd13c', 'dd']:
            for hem in ['sh', 'nh']:
                hem_idx = 0 if hem == 'sh' else 1
                y_model.extend(model_output[tracer][:, hem_idx])
    
    return np.array(y_model)

# ============================================================================
# OPTIONAL: PARALLEL JACOBIAN COMPUTATION (FOR SPEED)
# ============================================================================

def compute_jacobian_parallel(config, state_vec, forward_model, x, init_conditions, y_ref, ch4_only=False, n_workers=4):
    """
    Parallel version of compute_jacobian using multiprocessing.
    
    Can reduce computation time by ~4x on a 4-core machine.
    Use this if Jacobian computation takes >30 minutes.
    
    Args:
        n_workers: Number of parallel workers (default: 4)
    
    Returns:
        K: Jacobian matrix
    """
    import multiprocessing as mp
    from functools import partial
    
    n_state = len(x)
    n_obs = len(y_ref)
    K = np.zeros((n_obs, n_state))
    
    n_years = state_vec.n_years
    n_params_per_year = state_vec.n_params_per_year
    
    print(f"  Computing Jacobian (parallel, {n_workers} workers)...")
    
    # Helper function for parallel computation
    def compute_column(param_idx, x, state_vec, forward_model, init_conditions, y_ref, ch4_only):
        """Compute single Jacobian column."""
        x_pert = x.copy()
        
        year_idx = param_idx // n_params_per_year
        param_in_year = param_idx % n_params_per_year
        
        # Determine perturbation size
        n_emissions = 10
        n_isotopes_d13c = 10
        n_isotopes_dd = 10
        
        if param_in_year < n_emissions:
            # Emission
            pert_size = max(abs(x[param_idx]) * 0.01, 1e-6)
        elif param_in_year < n_emissions + n_isotopes_d13c:
            # δ13C
            pert_size = 0.1
        elif param_in_year < n_emissions + n_isotopes_d13c + n_isotopes_dd:
            # δD
            pert_size = 1.0
        else:
            # Sink
            pert_size = max(abs(x[param_idx]) * 0.01, 0.1)
        
        x_pert[param_idx] += pert_size
        
        emis, iso_d13c, iso_dd, sinks = state_vec.unpack(x_pert)
        output_pert = forward_model.run(emis, iso_d13c, iso_dd, sinks, init_conditions)
        y_pert = observation_operator(output_pert, None, ch4_only)
        
        return (y_pert - y_ref) / pert_size
    
    # Compute in parallel
    compute_func = partial(compute_column, x=x, state_vec=state_vec, 
                          forward_model=forward_model, init_conditions=init_conditions,
                          y_ref=y_ref, ch4_only=ch4_only)
    
    with mp.Pool(n_workers) as pool:
        columns = pool.map(compute_func, range(n_state))
    
    K = np.column_stack(columns)
    
    print(f"  ✓ Parallel Jacobian computed: {K.shape}")
    
    return K

# ============================================================================
# TWO-STEP GAUSS-NEWTON INVERSION
# ============================================================================

def build_smoothness_matrix(n_years, n_params_per_year, param_indices, order=2):
    """
    Build finite difference operator for temporal smoothness.
    
    Args:
        n_years: Number of time steps
        n_params_per_year: Parameters per year
        param_indices: List of parameter indices to constrain (e.g., [0,1,2,...,9])
        order: 1 for first difference, 2 for second difference
    
    Returns:
        D: Constraint matrix (n_constraints × n_total)
    """
    n_total = n_years * n_params_per_year
    n_params = len(param_indices)
    
    if order == 1:
        # First-order: Δx = x[t] - x[t-1]
        n_constraints = (n_years - 1) * n_params
        D = np.zeros((n_constraints, n_total))
        
        row = 0
        for year_idx in range(n_years - 1):
            for param_local_idx in param_indices:
                idx_curr = year_idx * n_params_per_year + param_local_idx
                idx_next = (year_idx + 1) * n_params_per_year + param_local_idx
                
                D[row, idx_curr] = -1.0
                D[row, idx_next] = 1.0
                row += 1
    
    elif order == 2:
        # Second-order: Δ²x = x[t-1] - 2*x[t] + x[t+1]
        n_constraints = (n_years - 2) * n_params
        D = np.zeros((n_constraints, n_total))
        
        row = 0
        for year_idx in range(1, n_years - 1):
            for param_local_idx in param_indices:
                idx_prev = (year_idx - 1) * n_params_per_year + param_local_idx
                idx_curr = year_idx * n_params_per_year + param_local_idx
                idx_next = (year_idx + 1) * n_params_per_year + param_local_idx
                
                D[row, idx_prev] = 1.0
                D[row, idx_curr] = -2.0
                D[row, idx_next] = 1.0
                row += 1
    
    return D

def compute_smoothness_penalty(x, x_prior, config, state_vec, D_matrices=None):
    """
    Compute smoothness penalty cost and gradient.
    
    Cost: J_smooth = λ * ||D(x - x_prior) / σ||²
    Now properly scaled by prior uncertainties!
    
    Args:
        x: Current state vector
        x_prior: Prior state vector
        config: InversionConfig
        state_vec: StateVector object
        D_matrices: Pre-computed (D_emis, D_iso_d13c, D_iso_dd, D_sinks) or None
    
    Returns:
        cost: Scalar smoothness cost
        gradient: Gradient vector (same size as x)
        D_matrices: Tuple of constraint matrices (for reuse)
    """
    n_years = state_vec.n_years
    n_params_per_year = state_vec.n_params_per_year
    
    # Build or reuse constraint matrices
    if D_matrices is None:
        D_emis = build_smoothness_matrix(
            n_years, n_params_per_year, 
            list(range(10)),  # Emissions: indices 0-9
            order=config.smoothness_order
        )
        D_iso_d13c = build_smoothness_matrix(
            n_years, n_params_per_year,
            list(range(10, 20)),  # δ13C: indices 10-19
            order=config.smoothness_order
        )
        D_iso_dd = build_smoothness_matrix(
            n_years, n_params_per_year,
            list(range(20, 30)),  # δD: indices 20-29
            order=config.smoothness_order
        )
        D_sinks = build_smoothness_matrix(
            n_years, n_params_per_year,
            list(range(30, 36)),  # Sinks: indices 30-35
            order=config.smoothness_order
        )
        D_matrices = (D_emis, D_iso_d13c, D_iso_dd, D_sinks)
    else:
        D_emis, D_iso_d13c, D_iso_dd, D_sinks = D_matrices
    
    dx = x - x_prior
    
    # Compute smoothness terms
    smooth_emis = D_emis @ dx
    smooth_iso_d13c = D_iso_d13c @ dx
    smooth_iso_dd = D_iso_dd @ dx
    smooth_sinks = D_sinks @ dx
    
    # ========================================================================
    # NEW: SCALE BY PRIOR ERRORS (makes penalty dimensionless)
    # ========================================================================
    
    # Get typical emission scale for normalization
    # unpack() returns dictionaries, so we need to extract arrays
    emis_dict, _, _, _ = state_vec.unpack(x_prior)
    
    # Concatenate all emission values into a single array
    all_emissions = []
    for source in config.sources:
        all_emissions.append(emis_dict[source].flatten())  # Flatten (n_years, 2) → 1D array
    all_emissions = np.concatenate(all_emissions)
    
    # Get typical emission magnitude (mean of positive values)
    typical_emission = np.mean(all_emissions[all_emissions > 0])
    emission_scale = config.prior_emission_error * typical_emission
    
    # Normalize smoothness terms by their scales
    smooth_emis_normalized = smooth_emis / emission_scale
    smooth_iso_d13c_normalized = smooth_iso_d13c / config.prior_isotope_d13c_error
    smooth_iso_dd_normalized = smooth_iso_dd / config.prior_isotope_dd_error
    smooth_sinks_normalized = smooth_sinks / config.prior_sink_error
    
    # Cost (now dimensionless and comparable across variables)
    cost = (
        config.smoothness_lambda_emissions * np.sum(smooth_emis_normalized ** 2) +
        config.smoothness_lambda_d13c * np.sum(smooth_iso_d13c_normalized ** 2) +    # δ13C
        config.smoothness_lambda_dd * np.sum(smooth_iso_dd_normalized ** 2) +        # δD (separate!)
        config.smoothness_lambda_sinks * np.sum(smooth_sinks_normalized ** 2)
    )
    
    # Gradient: Must account for the scaling
    # ∂J/∂x = 2λ * D^T * (D*dx / σ²)
    grad = (
        2 * config.smoothness_lambda_emissions * (D_emis.T @ smooth_emis_normalized) / emission_scale +
        2 * config.smoothness_lambda_d13c * (D_iso_d13c.T @ smooth_iso_d13c_normalized) / config.prior_isotope_d13c_error +
        2 * config.smoothness_lambda_dd * (D_iso_dd.T @ smooth_iso_dd_normalized) / config.prior_isotope_dd_error +
        2 * config.smoothness_lambda_sinks * (D_sinks.T @ smooth_sinks_normalized) / config.prior_sink_error
    )
    
    return cost, grad, D_matrices

def two_step_inversion(config, state_vec, forward_model, 
                       x_prior, prior_emissions, prior_isotopes,
                       observations, init_conditions):
    """
    Two-step inversion with temporal smoothness constraint.
    
    Cost function:
    J(x) = ||y - F(x)||²_Sy + ||x - x_a||²_Sa + λ * ||D(x - x_a)||²
    """
    
    # ========== STEP 1: CH4-only inversion ==========
    print("\n" + "="*70)
    print("STEP 1: CH4-ONLY INVERSION (EMISSIONS + SINKS)")
    print("="*70)
    print(f"Smoothness: λ_emis={config.smoothness_lambda_emissions}, "
          f"λ_sinks={config.smoothness_lambda_sinks}, order={config.smoothness_order}")
    
    Sa_step1 = two_step_construct_prior_covariance(config, state_vec, prior_emissions, prior_isotopes)
    y_obs_step1 = get_observation_vector(observations, ch4_only=True)
    Sy_step1 = get_observation_covariance(config, state_vec.n_years, ch4_only=True)
    
    x_step1 = x_prior.copy()
    Sa_inv_step1 = linalg.inv(Sa_step1)
    Sy_inv_step1 = linalg.inv(Sy_step1)
    
    costs_step1 = []
    D_matrices = None  # Will be computed once
    
    for iteration in range(3):  # Fewer iterations for step 1
        print(f"\n  Iteration {iteration + 1}/3")
        
        # Forward model
        emis, iso_d13c, iso_dd, sinks = state_vec.unpack(x_step1)
        output = forward_model.run(emis, iso_d13c, iso_dd, sinks, init_conditions)
        y_model = observation_operator(output, None, ch4_only=True)
        
        # Cost components
        innovation = y_obs_step1 - y_model
        cost_obs = innovation.T @ Sy_inv_step1 @ innovation
        
        dx = x_step1 - x_prior
        cost_prior = dx.T @ Sa_inv_step1 @ dx
        
        cost_smooth, grad_smooth, D_matrices = compute_smoothness_penalty(
            x_step1, x_prior, config, state_vec, D_matrices
        )
        
        total_cost = cost_obs + cost_prior + cost_smooth
        costs_step1.append(total_cost)
        
        print(f"    Cost: {total_cost:.2f} (obs: {cost_obs:.2f}, prior: {cost_prior:.2f}, smooth: {cost_smooth:.2f})")
        
        # Jacobian
        K = compute_jacobian(config, state_vec, forward_model, x_step1, 
                            init_conditions, y_model, ch4_only=True)
        
        # Modified normal equations with smoothness Hessian
        print("    Solving normal equations with smoothness...")
        
        D_emis, D_iso_d13c, D_iso_dd, D_sinks = D_matrices
        
        H = K.T @ Sy_inv_step1 @ K + Sa_inv_step1
        H += (2 * config.smoothness_lambda_emissions * (D_emis.T @ D_emis) +
              2 * config.smoothness_lambda_sinks * (D_sinks.T @ D_sinks))
        
        g = K.T @ Sy_inv_step1 @ innovation - Sa_inv_step1 @ dx - grad_smooth
        
        try:
            dx_step = linalg.solve(H, g)
            x_step1 = x_step1 + dx_step
            print(f"    Step size: {np.linalg.norm(dx_step):.2e}")
        except linalg.LinAlgError:
            print("    Matrix solve failed")
            break
    
    print(f"\n  Step 1 complete. Cost: {costs_step1[0]:.2f} → {costs_step1[-1]:.2f}")
    
    # ========== STEP 2: Multi-tracer optimization ==========
    print("\n" + "="*70)
    print("STEP 2: MULTI-TRACER OPTIMIZATION (EMISSIONS + ISOTOPES + SINKS)")
    print("="*70)
    print(f"Observation Sigma: σ_d13C={config.obs_sigma_d13c}, σ_dD={config.obs_sigma_dd}, "
          f"Prior Sigma: σ_d13C={config.prior_isotope_d13c_error}, σ_dD={config.prior_isotope_dd_error}")
    
    print(f"Smoothness: λ_emis={config.smoothness_lambda_emissions}, "
          f"λ_iso_d13C={config.smoothness_lambda_d13c}, λ_iso_dD={config.smoothness_lambda_dd}, "
          f"λ_sinks={config.smoothness_lambda_sinks}, order={config.smoothness_order}")
    
    Sa_step2 = two_step_construct_prior_covariance(config, state_vec, prior_emissions, prior_isotopes)
    y_obs_step2 = get_observation_vector(observations, ch4_only=False)
    Sy_step2 = get_observation_covariance(config, state_vec.n_years, ch4_only=False)
    
    x_step2 = x_step1.copy()
    Sa_inv_step2 = linalg.inv(Sa_step2)
    Sy_inv_step2 = linalg.inv(Sy_step2)
    
    costs_step2 = []
    D_matrices = None  # Recompute for consistency
    
    for iteration in range(config.n_iterations):
        print(f"\n  Iteration {iteration + 1}/{config.n_iterations}")
        
        # Forward model
        emis, iso_d13c, iso_dd, sinks = state_vec.unpack(x_step2)
        output = forward_model.run(emis, iso_d13c, iso_dd, sinks, init_conditions)
        y_model = observation_operator(output, None, ch4_only=False)
        
        # Cost components
        innovation = y_obs_step2 - y_model
        cost_obs = innovation.T @ Sy_inv_step2 @ innovation
        
        dx = x_step2 - x_prior
        cost_prior = dx.T @ Sa_inv_step2 @ dx
        
        cost_smooth, grad_smooth, D_matrices = compute_smoothness_penalty(
            x_step2, x_prior, config, state_vec, D_matrices
        )
        
        total_cost = cost_obs + cost_prior + cost_smooth
        costs_step2.append(total_cost)
        
        print(f"    Cost: {total_cost:.2f} (obs: {cost_obs:.2f}, prior: {cost_prior:.2f}, smooth: {cost_smooth:.2f})")
        
        # Jacobian
        K = compute_jacobian(config, state_vec, forward_model, x_step2, 
                            init_conditions, y_model, ch4_only=False)
        
        # Modified normal equations
        print("    Solving normal equations with smoothness...")
        
        D_emis, D_iso_d13c, D_iso_dd, D_sinks = D_matrices
        
        H = K.T @ Sy_inv_step2 @ K + Sa_inv_step2
        H += (2 * config.smoothness_lambda_emissions * (D_emis.T @ D_emis) +
              2 * config.smoothness_lambda_d13c * (D_iso_d13c.T @ D_iso_d13c) +
              2 * config.smoothness_lambda_dd * (D_iso_dd.T @ D_iso_dd) +
              2 * config.smoothness_lambda_sinks * (D_sinks.T @ D_sinks))
        
        g = K.T @ Sy_inv_step2 @ innovation - Sa_inv_step2 @ dx - grad_smooth
        
        try:
            dx_step = linalg.solve(H, g)
            
            # Line search
            alpha = 1.0
            x_new = x_step2 + alpha * dx_step
            
            for ls_iter in range(5):
                emis_test, iso_test, dd_test, sink_test = state_vec.unpack(x_new)
                output_test = forward_model.run(emis_test, iso_test, dd_test, 
                                               sink_test, init_conditions)
                y_test = observation_operator(output_test, None, ch4_only=False)
                
                innov_test = y_obs_step2 - y_test
                dx_test = x_new - x_prior
                cost_smooth_test, _, _ = compute_smoothness_penalty(
                    x_new, x_prior, config, state_vec, D_matrices
                )
                
                cost_test = (innov_test.T @ Sy_inv_step2 @ innov_test + 
                            dx_test.T @ Sa_inv_step2 @ dx_test + 
                            cost_smooth_test)
                
                if cost_test < total_cost:
                    x_step2 = x_new
                    print(f"    Step size: {np.linalg.norm(dx_step):.2e} (alpha={alpha:.2f})")
                    break
                
                alpha *= 0.5
                x_new = x_step2 + alpha * dx_step
            else:
                print(f"    Line search completed (alpha={alpha:.2f})")
                x_step2 = x_new
                
        except linalg.LinAlgError:
            print("    Matrix solve failed")
            break
        
        # Early stopping
        if iteration > 0:
            cost_change = abs(costs_step2[-1] - costs_step2[-2]) / costs_step2[-2]
            if cost_change < 1e-5:
                print(f"    Convergence criterion met (relative change: {cost_change:.2e})")
                break
    
    print(f"\n  Step 2 complete. Cost: {costs_step2[0]:.2f} → {costs_step2[-1]:.2f}")
    
    # Final output
    emis_final, iso_d13c_final, iso_dd_final, sinks_final = state_vec.unpack(x_step2)
    output_final = forward_model.run(emis_final, iso_d13c_final, iso_dd_final, 
                                     sinks_final, init_conditions)
    
    diagnostics = {
        'costs_step1': costs_step1,
        'costs_step2': costs_step2,
        'costs': costs_step1 + costs_step2,
        'x_step1': x_step1,
        'final_output': output_final
    }
    
    return x_step2, diagnostics

def main():
    """Main inversion script."""
    
    print("\n" + "="*70)
    print("TIME-VARYING EMISSION + ISOTOPE INVERSION (TWO-STEP)")
    print("="*70)
    
    # Initialize
    config = InversionConfig()
    state_vec = StateVector(config)
    forward_model = ForwardModel(config)
    
    # Load data
    print("\n=== Loading Data ===")
    prior_emissions = load_prior_emissions(config)
    observations = load_observations(config)
    prior_isotopes = get_prior_isotopes(config)
    
    # Initialize prior state
    print("\n=== Initializing Prior ===")
    n_years = config.year2 - config.year1 + 1
    years = np.arange(config.year1, config.year2 + 1)  # ← DEFINE YEARS HERE
    
    # Replicate prior isotopes for each year (constant prior)
    prior_iso_d13c = {}
    prior_iso_dd = {}
    for source in config.sources:
        prior_iso_d13c[source] = np.tile(prior_isotopes[source]['d13c'], (n_years, 1))
        prior_iso_dd[source] = np.tile(prior_isotopes[source]['dd'], (n_years, 1))
    
    # Prior sinks (zero perturbation)
    prior_sinks = {sink: np.zeros((n_years, 2)) for sink in ['oh', 'strat', 'soil']}
    
    # Pack prior
    x_prior = state_vec.pack(prior_emissions, prior_iso_d13c, prior_iso_dd, prior_sinks)
    
    print(f"  Prior state vector length: {len(x_prior)}")
    
    # Initial conditions (from observations)
    init_conditions = {
        'ch4': np.array([observations['ch4']['sh'][0], observations['ch4']['nh'][0]]),
        'd13c': np.array([observations['d13c']['sh'][0], observations['d13c']['nh'][0]]),
        'dd': np.array([observations['dd']['sh'][0], observations['dd']['nh'][0]])
    }
    
    # Test prior
    print("\n=== Testing Prior ===")
    emis_prior, iso_d13c_prior, iso_dd_prior, sinks_prior = state_vec.unpack(x_prior)
    output_prior = forward_model.run(emis_prior, iso_d13c_prior, iso_dd_prior, sinks_prior, init_conditions)
    y_prior = observation_operator(output_prior, None, ch4_only=False)
    
    y_obs_all = get_observation_vector(observations, ch4_only=False)
    Sy_all = get_observation_covariance(config, n_years, ch4_only=False)
    innovation_prior = y_obs_all - y_prior
    cost_prior = innovation_prior.T @ linalg.inv(Sy_all) @ innovation_prior
    print(f"  Prior cost (obs only): {cost_prior:.2f}")
    
    # Run inversion
    print("\n=== Running Two-Step Simultaneous Inversion ===")
    x_post, diagnostics = two_step_inversion(
        config, state_vec, forward_model, 
        x_prior, prior_emissions, prior_isotopes,
        observations, init_conditions
    )
    
    # Extract posterior
    print("\n=== Extracting Posterior ===")
    emis_post, iso_d13c_post, iso_dd_post, sinks_post = state_vec.unpack(x_post)
    
    # === ADD TEMPORAL SMOOTHNESS DIAGNOSTICS ===
    print_smoothness_diagnostics(years, prior_emissions, emis_post, config)
    
    # Recalculate final posterior output
    print("\n=== Recalculating Final Model Output ===")
    output_post = forward_model.run(emis_post, iso_d13c_post, iso_dd_post, sinks_post, init_conditions)
    
    # Save results
    print("\n=== Saving Results ===")
    output_dir = '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run/multi_inversion_results_v0/time_varying_inversion_outputs_two_step'
    os.makedirs(output_dir, exist_ok=True)
    
    years = np.arange(config.year1, config.year2 + 1)
    
    # Save emissions
    for source in config.sources:
        df = pd.DataFrame({
            'Year': years,
            'Prior_SH': prior_emissions[source][:, 0],
            'Prior_NH': prior_emissions[source][:, 1],
            'Posterior_SH': emis_post[source][:, 0],
            'Posterior_NH': emis_post[source][:, 1]
        })
        df.to_csv(os.path.join(output_dir, f'emissions_{source}.csv'), index=False)
    
    # Save isotopes
    for source in config.sources:
        df = pd.DataFrame({
            'Year': years,
            'Prior_d13C_SH': prior_iso_d13c[source][:, 0],
            'Prior_d13C_NH': prior_iso_d13c[source][:, 1],
            'Post_d13C_SH': iso_d13c_post[source][:, 0],
            'Post_d13C_NH': iso_d13c_post[source][:, 1],
            'Prior_dD_SH': prior_iso_dd[source][:, 0],
            'Prior_dD_NH': prior_iso_dd[source][:, 1],
            'Post_dD_SH': iso_dd_post[source][:, 0],
            'Post_dD_NH': iso_dd_post[source][:, 1]
        })
        df.to_csv(os.path.join(output_dir, f'isotopes_{source}.csv'), index=False)

    # Save total methane lifetime in the same tidy format as emissions.
    save_lifetime_csv(years, prior_sinks, sinks_post, config, output_dir)
    
    # Save model-data comparison
    comparison = pd.DataFrame({
        'Year': years,
        'CH4_SH_obs': observations['ch4']['sh'],
        'CH4_SH_post': output_post['ch4'][:, 0],
        'CH4_NH_obs': observations['ch4']['nh'],
        'CH4_NH_post': output_post['ch4'][:, 1],
        'd13C_SH_obs': observations['d13c']['sh'],
        'd13C_SH_post': output_post['d13c'][:, 0],
        'd13C_NH_obs': observations['d13c']['nh'],
        'd13C_NH_post': output_post['d13c'][:, 1],
        'dD_SH_obs': observations['dd']['sh'],
        'dD_SH_post': output_post['dd'][:, 0],
        'dD_NH_obs': observations['dd']['nh'],
        'dD_NH_post': output_post['dd'][:, 1]
    })
    comparison.to_csv(os.path.join(output_dir, 'model_data_comparison.csv'), index=False)
    
    # Save model-data comparison CSV files
    prior_isotopes_dict = get_prior_isotopes(config)
    config.prior_isotopes = prior_isotopes_dict  # Add to config for CSV functions
    
    #save_f1_csv(years, output_prior, output_post, observations, output_dir, config)
    save_f3_csv(years, emis_post, sinks_post, config, output_dir)

    # Plot results
    print("\n=== Creating Plots ===")
    plot_results(config, years, emis_post, iso_d13c_post, iso_dd_post, sinks_post,
                prior_emissions, prior_iso_d13c, prior_iso_dd, prior_sinks,
                output_prior, output_post, observations, output_dir)
    
    print(f"\n✅ Inversion complete!")
    print(f"📁 Results saved to: {output_dir}/")
    print("The cost function is dimensionless (it's the sum of weighted squared residuals)")
    print(f"📊 Cost reduction: {diagnostics['costs'][0]:.2f} → {diagnostics['costs'][-1]:.2f}")
# ============================================================================
# PLOTTING
# ============================================================================
FONTSIZE = {
    'title': 12,
    'tick_label': 10,
}

COLORS = {
    'nh': '#f8c8c8',
    'sh': '#c8d8f8',
    'wetlands': '#2ecc71',
    'agriculture': '#e67e22',
    'pyrogenic': '#e74c3c',
    'fossil': '#7f8c8d',
    'waste': '#9b59b6',
    'trop_sink': '#1abc9c',
    'strat_sink': '#3498db',
    'soil_sink': '#795548',
}

def load_mei_data(year1=1980, year2=2024):
    """Load MEI data."""
    try:
        mei_file = '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run/input_data/meiv2.data.txt'
        mei_cols = ['Year', 'DJ', 'JF', 'FM', 'MA', 'AM', 'MJ', 'JJ', 'JA', 'AS', 'SO', 'ON', 'ND']
        mei_data = pd.read_csv(mei_file, delim_whitespace=True, skiprows=1, skipfooter=4, 
                               header=None, engine='python')
        mei_data.columns = mei_cols
        mei_long = mei_data.melt(id_vars='Year', var_name='Month', value_name='Anomaly')
        month_map = {name: str(i).zfill(2) for i, name in enumerate(mei_cols[1:], 1)}
        mei_long['Month'] = mei_long['Month'].map(month_map)
        mei_long['Date'] = pd.to_datetime(mei_long['Year'].astype(str) + '-' + mei_long['Month'], format='%Y-%m')
        mei_long = mei_long.dropna(subset=['Anomaly'])
        mei_long['Year'] = mei_long['Date'].dt.year
        mei_yearly = mei_long[(mei_long['Year'] >= year1) & (mei_long['Year'] <= year2)].groupby('Year')['Anomaly'].mean().reset_index()
        mei_yearly.rename(columns={'Anomaly': 'MEI'}, inplace=True)
        mei_yearly['abs_MEI'] = np.abs(mei_yearly['MEI'])
        mei_yearly['is_ENSO'] = mei_yearly['abs_MEI'] >= 0.5
        mei_yearly['ENSO_type'] = 'Neutral'
        mei_yearly.loc[mei_yearly['MEI'] > 0.5, 'ENSO_type'] = 'El Niño'
        mei_yearly.loc[mei_yearly['MEI'] < -0.5, 'ENSO_type'] = 'La Niña'
        return mei_yearly
    except Exception as e:
        print(f"⚠ MEI load failed: {e}")
        years = np.arange(year1, year2 + 1)
        df = pd.DataFrame({'Year': years, 'MEI': np.zeros(len(years)), 
                          'abs_MEI': np.zeros(len(years)), 'is_ENSO': False})
        df['ENSO_type'] = 'Neutral'
        return df

mei_df = load_mei_data(1980, 2024)
# Identify ENSO years for shading
el_nino_years = mei_df[mei_df['MEI'] > 0.5]['Year'].values
la_nina_years = mei_df[mei_df['MEI'] < -0.5]['Year'].values

def shade_enso(ax):
    for y in el_nino_years:
        ax.axvspan(y - 0.5, y + 0.5, color='red', alpha=0.1)
    for y in la_nina_years:
        ax.axvspan(y - 0.5, y + 0.5, color='blue', alpha=0.1)

def plot_results(config, years, emis_post, iso_d13c_post, iso_dd_post, sinks_post,
                prior_emissions, prior_iso_d13c, prior_iso_dd, prior_sinks,
                output_prior, output_post, observations, output_dir):
    """Create summary plots matching the F1, F2, F5 plotter layout."""
    
    # F1: Model-data fit (observations vs model) - NOW ALSO SAVES CSV
    plot_f1_model_data_fit(years, output_prior, output_post, observations, output_dir, config)
    
    # F2: Prior vs Posterior emissions (all sources) - 6 sources × 2 hemispheres
    plot_f2_emissions_comparison(years, prior_emissions, emis_post, config, output_dir, 
                                  prior_sinks, sinks_post)
    
    # F3: Hemispheric emissions (legacy single-run diagnostic)
    plot_f3_hemispheric_emissions(years, emis_post, config, output_dir, sinks_post)
    
    # F5: posterior isotope timeseries in the standalone plotter style
    plot_f5_isotopes(
        years=years,
        iso_d13c_post=iso_d13c_post,
        iso_dd_post=iso_dd_post,
        prior_iso_d13c=prior_iso_d13c,
        prior_iso_dd=prior_iso_dd,
        config=config,
        output_dir=output_dir,
    )

    # ENSO isotope shifts, analogous to plot_isotope_enso_shifts_diff
    plot_isotope_enso_shifts_diff(
        years=years,
        iso_d13c_post=iso_d13c_post,
        iso_dd_post=iso_dd_post,
        output_dir=output_dir,
    )

    # F4: Optimized timeseries (legacy diagnostic with weighted means and sinks)
    plot_f4_optimized_timeseries(
        years=years,
        iso_d13c_post=iso_d13c_post,
        iso_dd_post=iso_dd_post,
        sinks_post=sinks_post,
        prior_iso_d13c=prior_iso_d13c,
        prior_iso_dd=prior_iso_dd,
        prior_sinks=prior_sinks,
        config=config,
        output_dir=output_dir,
        emis_post=emis_post
    )

def plot_f1_model_data_fit(years, output_prior, output_post, observations, output_dir, config):
    """
    F1: Model-data fit plot (3 tracers × 2 hemispheres).
    Shows observations (black dots) vs prior (blue) vs posterior (red).
    Also saves the data to CSV.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12.5), sharex=True)
    
    tracers = ['ch4', 'd13c', 'dd']
    tracer_labels = ['CH$_4$ (ppb)', '$\delta^{13}$C-CH$_4$ (‰)', '$\delta$D-CH$_4$ (‰)']
    markers = ['+', '.']
    
    # ========== PREPARE CSV DATA WHILE PLOTTING ==========
    param_str = (f"{config.kie_oh_d13c}|{config.kie_oh_dd}|"
                f"{config.prior_isotopes['wetlands']['d13c'][0]}|{config.prior_isotopes['wetlands']['d13c'][1]}|"
                f"{config.prior_isotopes['wetlands']['dd'][0]}|{config.prior_isotopes['wetlands']['dd'][1]}|"
                f"{config.prior_isotopes['agriculture']['d13c'][0]}|{config.prior_isotopes['agriculture']['d13c'][1]}|"
                f"{config.prior_isotopes['agriculture']['dd'][0]}|{config.prior_isotopes['agriculture']['dd'][1]}|"
                f"{config.prior_isotopes['pyrogenic']['d13c'][0]}|{config.prior_isotopes['pyrogenic']['d13c'][1]}|"
                f"{config.prior_isotopes['pyrogenic']['dd'][0]}|{config.prior_isotopes['pyrogenic']['dd'][1]}|"
                f"{config.prior_isotopes['fossil']['d13c'][0]}|{config.prior_isotopes['fossil']['d13c'][1]}|"
                f"{config.prior_isotopes['fossil']['dd'][0]}|{config.prior_isotopes['fossil']['dd'][1]}|"
                f"{config.prior_isotopes['waste']['d13c'][0]}|{config.prior_isotopes['waste']['d13c'][1]}|"
                f"{config.prior_isotopes['waste']['dd'][0]}|{config.prior_isotopes['waste']['dd'][1]}|"
                f"101|{config.lifetime_oh[0]}|{config.prior_emission_error}|{config.prior_emission_error}|"
                f"9|{config.prior_sink_error}|{config.obs_sigma_d13c}|{config.obs_sigma_dd}|"
                f"{config.tau_ihex}|{config.tg2ppb}|True|"
                "{'wetlands': {'d13c': True, 'dd': True}, 'pyrogenic': {'d13c': True, 'dd': True}, "
                "'agriculture': {'d13c': True, 'dd': True}, 'fossil': {'d13c': True, 'dd': True}, "
                "'waste': {'d13c': True, 'dd': True}}")
    
    csv_rows = []
    csv_rows.append([param_str] * 6)
    # Header rows for posterior
    csv_rows.append(['ch4', 'ch4', 'd13c-ch4', 'd13c-ch4', 'dd-ch4', 'dd-ch4'])
    csv_rows.append(['sh', 'nh', 'sh', 'nh', 'sh', 'nh'])
    
    # ========== PLOT AND SAVE POSTERIOR DATA SIMULTANEOUSLY ==========
    for i, (tracer, label) in enumerate(zip(tracers, tracer_labels)):
        ax = axes[i]
        
        for j, hem in enumerate(['sh', 'nh']):
            hem_idx = j
            shade_enso(ax)
            # Observations (black)
            ax.plot(years, observations[tracer][hem], color='black', 
                   marker=markers[j], linestyle='--', markersize=6, 
                   label=hem.upper() if i == 0 else '', alpha=0.8)
            
            # Prior (blue)
            ax.plot(years, output_prior[tracer][:, hem_idx], color='blue',
                   marker=markers[j], linestyle='-', linewidth=1.5, 
                   markersize=3, alpha=0.6)
            
            # Posterior (red) - THIS IS WHAT WE PLOT
            posterior_data = output_post[tracer][:, hem_idx]
            ax.plot(years, posterior_data, color='red',
                   marker=markers[j], linestyle='-', linewidth=1.5,
                   markersize=3, alpha=0.6)
        
        # Formatting
        ax.set_ylabel(label, fontsize=12)
        ax.yaxis.set_ticks_position('both')
        ax.grid(alpha=0.3)
        
        # Shaded regions for spin-up and spin-down
        ax.axvspan(years[0], 1994, facecolor='grey', alpha=0.3)
        ax.axvspan(2022, years[-1], facecolor='grey', alpha=0.3)
        
        if i < 2:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)
        if i > 0:
            ax.spines['top'].set_visible(False)
    
    # Posterior data 
    for year_idx in range(len(years)):
        csv_rows.append([
            output_post['ch4'][:, 0][year_idx],   # SH - SAME as plotted
            output_post['ch4'][:, 1][year_idx],   # NH - SAME as plotted
            output_post['d13c'][:, 0][year_idx],  # SH - SAME as plotted
            output_post['d13c'][:, 1][year_idx],  # NH - SAME as plotted
            output_post['dd'][:, 0][year_idx],    # SH - SAME as plotted
            output_post['dd'][:, 1][year_idx]     # NH - SAME as plotted
        ])
    # Header rows for observations
    csv_rows.append(['ch4', 'ch4', 'd13c-ch4', 'd13c-ch4', 'dd-ch4', 'dd-ch4'])
    csv_rows.append(['sh', 'nh', 'sh', 'nh', 'sh', 'nh'])
    
    # Observations data
    for year_idx in range(len(years)):
        csv_rows.append([
            observations['ch4']['sh'][year_idx],
            observations['ch4']['nh'][year_idx],
            observations['d13c']['sh'][year_idx],
            observations['d13c']['nh'][year_idx],
            observations['dd']['sh'][year_idx],
            observations['dd']['nh'][year_idx]
        ])
    # Calculate RMSE
    rmse_row = []
    for tracer, hem in [('ch4', 'sh'), ('ch4', 'nh'), 
                        ('d13c', 'sh'), ('d13c', 'nh'),
                        ('dd', 'sh'), ('dd', 'nh')]:
        hem_idx = 0 if hem == 'sh' else 1
        obs_vals = observations[tracer][hem]
        model_vals = output_post[tracer][:, hem_idx]
        rmse = np.sqrt(np.mean((obs_vals - model_vals) ** 2))
        rmse_row.append(rmse)
    
    csv_rows.append(rmse_row)
    
    # Save CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(os.path.join(output_dir, 'model_and_obs_values_em&iso_varying.csv'), 
              index=False, header=[str(i) for i in range(6)])
    
    # Legend and save plot
    axes[0].legend(loc='upper left', frameon=False, fontsize=10)
    axes[2].set_xlabel('Year', fontsize=12)
    axes[2].set_xticks(years[::4])
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(output_dir, 'f1_model_data_fit.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  ✓ Saved: f1_model_data_fit.png")
    print(f"  ✓ Saved: model_and_obs_values_em&iso_varying.csv (from same data as plot)")
    print(f"    Diagnostic: First year CH4_SH_post = {output_post['ch4'][0, 0]:.2f}, Last year = {output_post['ch4'][-1, 0]:.2f}")

def plot_f2_emissions_comparison(years, prior_emissions, emis_post, config, output_dir, 
                                  prior_sinks, sinks_post):
    """
    F2: Prior vs posterior emissions in the standalone plotter style.
    Shows SH (solid) and NH (dashed) separately for each source plus global lifetime.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    sources_plot = config.sources + ['total_sink']
    panel_labels = list('abcdef')
    
    for i, source in enumerate(sources_plot):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        shade_enso(ax)
        ax.text(0.01, 0.98, f'({panel_labels[i]})', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left', zorder=200)
        
        if source == 'total_sink':
            prior_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + prior_sinks['oh'] / 100.0)
            prior_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + prior_sinks['strat'] / 100.0)
            prior_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + prior_sinks['soil'] / 100.0)
            prior_total_sink_rate = prior_sink_oh + prior_sink_strat + prior_sink_soil
            prior_lifetime = 1.0 / prior_total_sink_rate
            prior_lifetime_global = 2.0 / (1.0/prior_lifetime[:, 0] + 1.0/prior_lifetime[:, 1])
            
            post_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + sinks_post['oh'] / 100.0)
            post_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + sinks_post['strat'] / 100.0)
            post_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + sinks_post['soil'] / 100.0)
            post_total_sink_rate = post_sink_oh + post_sink_strat + post_sink_soil
            post_lifetime = 1.0 / post_total_sink_rate
            post_lifetime_global = 2.0 / (1.0/post_lifetime[:, 0] + 1.0/post_lifetime[:, 1])
            
            ax.plot(years, prior_lifetime_global, color='grey', linestyle=':',
                    linewidth=2.5, marker='.', alpha=0.7, label='Prior', zorder=50)
            ax.plot(years, post_lifetime_global, color='red', linestyle='-',
                    linewidth=2.0, marker='.', alpha=0.85, label='Posterior', zorder=100)
            ax.set_ylabel('Lifetime (years)', fontsize=11)
        else:
            ax.plot(years, prior_emissions[source][:, 0], color='grey',
                    linestyle=':', linewidth=2.5, marker='.', markersize=3, alpha=0.7,
                    zorder=50, label='Prior SH' if i == 0 else '')
            ax.plot(years, prior_emissions[source][:, 1], color='grey',
                    linestyle='-.', linewidth=2.5, alpha=0.7, zorder=50,
                    label='Prior NH' if i == 0 else '')
            ax.plot(years, emis_post[source][:, 0], color='red', linewidth=2.0,
                    linestyle='-', marker='.', markersize=3, alpha=0.85, zorder=100,
                    label='Posterior SH' if i == 0 else '')
            ax.plot(years, emis_post[source][:, 1], color='red', linewidth=1.5,
                    linestyle='--', alpha=0.7, zorder=100,
                    label='Posterior NH' if i == 0 else '')
            ax.set_ylabel('Emissions (Tg/yr)', fontsize=11)
        
        ax.set_title(source.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(years[0], years[-1])
        ax.axvspan(years[0], 1994, facecolor='grey', alpha=0.3)
        ax.axvspan(2022, years[-1], facecolor='grey', alpha=0.3)
        
        if row < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Year', fontsize=11)
        
        if i == 5:
            proxy = [
                Line2D([0], [0], color='grey', linestyle=':', linewidth=2, label='Prior SH'),
                Line2D([0], [0], color='grey', linestyle='-.', linewidth=2, label='Prior NH'),
                Line2D([0], [0], color='red', linestyle='-', linewidth=1.5, label='Posterior SH'),
                Line2D([0], [0], color='red', linestyle='--', linewidth=1.2, label='Posterior NH'),
            ]
            ax.legend(handles=proxy, loc='lower right', frameon=True,
                      fontsize=FONTSIZE['title'], ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f2_emissions_prior_vs_posterior.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: f2_emissions_prior_vs_posterior.png")

def plot_f3_hemispheric_emissions(years, emis_post, config, output_dir, sinks_post):
    """
    F3: Hemispheric breakdown of posterior emissions (all 5 sources + sink).
    Shows SH (green) and NH (purple) separately.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    sources_plot = config.sources + ['total_sink']
    colors = ['green', 'purple']
    
    for i, source in enumerate(sources_plot):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if source == 'total_sink':
            # Compute hemispheric lifetime (SH and NH separately)
            post_sink_oh = 1 / config.lifetime_oh * (1 + sinks_post['oh'] / 100.0)
            post_sink_strat = 1 / config.lifetime_strat * (1 + sinks_post['strat'] / 100.0)
            post_sink_soil = 1 / config.lifetime_soil * (1 + sinks_post['soil'] / 100.0)
            post_total_sink_rate = post_sink_oh + post_sink_strat + post_sink_soil
            post_lifetime = 1.0 / post_total_sink_rate  # shape: (n_years, 2)
            
            # Plot SH (green) and NH (purple)
            for j, hem in enumerate(['sh', 'nh']):
                ax.plot(years, post_lifetime[:, j], 
                       color=colors[j], marker='.', linewidth=2,
                       label=hem.upper())
            
            ax.set_ylabel('Lifetime (years)', fontsize=11)
        else:
            for j, hem in enumerate(['sh', 'nh']):
                ax.plot(years, emis_post[source][:, j], 
                       color=colors[j], marker='.', linewidth=2,
                       label=hem.upper())
            
            ax.set_ylabel('Emissions (Tg/yr)', fontsize=11)
        
        ax.set_title(source.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        shade_enso(ax)
        # Shaded regions
        ax.axvspan(years[0], 1994, facecolor='grey', alpha=0.3)
        ax.axvspan(2022, years[-1], facecolor='grey', alpha=0.3)
        
        if row < 2:
            ax.set_xticks([])
        
        if i == 0:
            ax.legend(loc='upper right', frameon=False, fontsize=10)
    
    axes[2, 0].set_xlabel('Year', fontsize=11)
    axes[2, 1].set_xlabel('Year', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f3_hemispheric_emissions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: f3_hemispheric_emissions.png")

def plot_f5_isotopes(years, iso_d13c_post, iso_dd_post, prior_iso_d13c, prior_iso_dd,
                     config, output_dir):
    """F5: isotope signatures matching the standalone plotter layout."""
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    panel_labels = list('abcdefghij')

    for i, source in enumerate(config.sources):
        ax_d13c = axes[i, 0]
        ax_dd = axes[i, 1]
        shade_enso(ax_d13c)
        shade_enso(ax_dd)

        ax_d13c.text(0.01, 0.98, f'({panel_labels[i * 2]})',
                     transform=ax_d13c.transAxes, fontsize=12,
                     fontweight='bold', va='top', ha='left', zorder=200)
        ax_dd.text(0.01, 0.98, f'({panel_labels[i * 2 + 1]})',
                   transform=ax_dd.transAxes, fontsize=12,
                   fontweight='bold', va='top', ha='left', zorder=200)

        ax_d13c.plot(years, iso_d13c_post[source][:, 0], color='red',
                     linewidth=2.0, alpha=0.85, linestyle='-', marker='.',
                     markersize=2, label='Posterior SH' if i == 0 else '')
        ax_d13c.plot(years, iso_d13c_post[source][:, 1], color='red',
                     linewidth=1.5, alpha=0.7, linestyle='--',
                     label='Posterior NH' if i == 0 else '')
        ax_dd.plot(years, iso_dd_post[source][:, 0], color='red',
                   linewidth=2.0, alpha=0.85, linestyle='-', marker='.',
                   markersize=2, label='Posterior SH' if i == 0 else '')
        ax_dd.plot(years, iso_dd_post[source][:, 1], color='red',
                   linewidth=1.5, alpha=0.7, linestyle='--',
                   label='Posterior NH' if i == 0 else '')

        ax_d13c.plot(years, prior_iso_d13c[source][:, 0], color='darkgrey',
                     linestyle='-', linewidth=2.5, alpha=0.7,
                     label='Prior SH' if i == 0 else '', zorder=50)
        ax_d13c.plot(years, prior_iso_d13c[source][:, 1], color='darkgrey',
                     linestyle='--', linewidth=2.5, alpha=0.7,
                     label='Prior NH' if i == 0 else '', zorder=50)
        ax_dd.plot(years, prior_iso_dd[source][:, 0], color='darkgrey',
                   linestyle='-', linewidth=2.5, alpha=0.7,
                   label='Prior SH' if i == 0 else '', zorder=50)
        ax_dd.plot(years, prior_iso_dd[source][:, 1], color='darkgrey',
                   linestyle='--', linewidth=2.5, alpha=0.7,
                   label='Prior NH' if i == 0 else '', zorder=50)

        ax_d13c.set_ylabel('δ¹³C (‰)', fontsize=11)
        ax_dd.set_ylabel('δD (‰)', fontsize=11)
        ax_d13c.set_title(source.capitalize(), fontsize=11, fontweight='bold')
        ax_dd.set_title(source.capitalize(), fontsize=11, fontweight='bold')
        ax_d13c.grid(alpha=0.3)
        ax_dd.grid(alpha=0.3)
        for ax in (ax_d13c, ax_dd):
            ax.axvspan(years[0], 1994, facecolor='grey', alpha=0.3, zorder=1)
            ax.axvspan(2022, years[-1], facecolor='grey', alpha=0.3, zorder=1)
            ax.set_xlim(years[0], years[-1])

        if i < len(config.sources) - 1:
            ax_d13c.set_xticks([])
            ax_dd.set_xticks([])
        else:
            ax_d13c.set_xlabel('Year', fontsize=11)
            ax_dd.set_xlabel('Year', fontsize=11)
            ax_d13c.set_xticks(years[::4])
            ax_dd.set_xticks(years[::4])

        if i == 0:
            ax_d13c.legend(loc='center right', frameon=True,
                           fontsize=FONTSIZE['title'], ncol=2)

    plt.suptitle('Posterior Isotopic Signatures (SH solid, NH dashed)\n'
                 'Red = El Niño | Blue = La Niña',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_dir, 'f5_isotopes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: f5_isotopes.png")

def plot_isotope_enso_shifts_diff(years, iso_d13c_post, iso_dd_post, output_dir):
    """ENSO isotope shifts (El Niño minus La Niña) from the single posterior run."""
    mei_local = load_mei_data(int(years[0]), int(years[-1]))
    phase_by_year = mei_local.set_index('Year')['ENSO_type'].to_dict()
    el_mask = np.array([phase_by_year.get(int(y)) == 'El Niño' for y in years])
    la_mask = np.array([phase_by_year.get(int(y)) == 'La Niña' for y in years])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sources = ['wetlands', 'pyrogenic']
    source_labels = ['Wetlands', 'Pyrogenic']
    isotope_sets = [
        ('d13c', iso_d13c_post, 'δ¹³C (‰)'),
        ('dd', iso_dd_post, 'δD (‰)'),
    ]

    for row, (isotope, iso_dict, iso_label) in enumerate(isotope_sets):
        for col, (source, src_label) in enumerate(zip(sources, source_labels)):
            ax = axes[row, col]
            shifts = []
            p_values = []

            for hem_idx, hem_name in enumerate(['SH', 'NH']):
                vals = iso_dict[source][:, hem_idx]
                el_vals = vals[el_mask]
                la_vals = vals[la_mask]
                if len(el_vals) and len(la_vals):
                    shifts.append(float(np.mean(el_vals) - np.mean(la_vals)))
                    p_values.append(float(stats.mannwhitneyu(el_vals, la_vals, alternative='two-sided').pvalue))
                else:
                    shifts.append(np.nan)
                    p_values.append(np.nan)

            global_vals = np.mean(iso_dict[source], axis=1)
            if np.any(el_mask) and np.any(la_mask):
                global_shift = float(np.mean(global_vals[el_mask]) - np.mean(global_vals[la_mask]))
                global_p = float(stats.mannwhitneyu(global_vals[el_mask], global_vals[la_mask],
                                                    alternative='two-sided').pvalue)
            else:
                global_shift = np.nan
                global_p = np.nan
            shifts.append(global_shift)
            p_values.append(global_p)

            positions = np.arange(3)
            colors = ['lightblue', 'lightgreen', 'lightyellow']
            ax.bar(positions, shifts, color=colors, edgecolor='black', linewidth=1.0)
            for pos, shift, p_val in zip(positions, shifts, p_values):
                if np.isfinite(shift):
                    y_offset = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    va = 'top' if shift >= 0 else 'bottom'
                    sig = '*' if np.isfinite(p_val) and p_val < 0.05 else ''
                    ax.text(pos, shift + (y_offset if shift >= 0 else -y_offset),
                            f'{shift:.2f}{sig}\np={p_val:.3f}' if np.isfinite(p_val) else f'{shift:.2f}',
                            ha='center', va=va, fontsize=8)

            ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
            ax.set_xticks(positions)
            ax.set_xticklabels(['SH', 'NH', 'Global'], fontsize=11)
            ax.set_ylabel(f'{iso_label} Shift\n(El Niño - La Niña)', fontsize=11)
            ax.set_title(f'{src_label} {iso_label.split()[0]}', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Isotopic Shifts During ENSO (El Niño - La Niña)\n'
                 'Bars = posterior mean phase difference, * = Mann-Whitney p<0.05',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'enso_isotope_diff.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: enso_isotope_diff.png")

def plot_f4_optimized_timeseries(
    years,
    iso_d13c_post,
    iso_dd_post,
    sinks_post,
    prior_iso_d13c,
    prior_iso_dd,
    prior_sinks,
    config,
    output_dir,
    emis_post,
):
    """
    New F4 layout (5 × 2 grid):
      Row 0: δ13C posteriors for all sources
      Row 1: δ13C weighted averages (prior vs optimized)
      Row 2: δD   posteriors for all sources
      Row 3: δD   weighted averages
      Row 4: Sink loss rates (OH, Strat, Soil)
    """

    # 5 rows: d13C posts, d13C weighted, dD posts, dD weighted, sinks
    fig, axes = plt.subplots(5, 2, figsize=(16, 16))
    hemispheres = ["SH", "NH"]
    colors = ["darkgreen", "gold", "orangered", "dimgray", "brown"]

    prior_isotopes = get_prior_isotopes(config)

    # --------------------------------------
    # Helper: compute weighted averages
    # --------------------------------------
    def compute_weighted(post_dict, prior_isotope_key, hem_idx):
        weighted_post = np.zeros(len(years))
        weighted_prior = np.zeros(len(years))

        for k in range(len(years)):
            total_emis = sum(emis_post[s][k, hem_idx] for s in config.sources)
            denom = total_emis + 1e-10

            weighted_post[k] = sum(
                emis_post[s][k, hem_idx] * post_dict[s][k, hem_idx]
                for s in config.sources
            ) / denom

            weighted_prior[k] = sum(
                emis_post[s][k, hem_idx] * prior_isotopes[s][prior_isotope_key][hem_idx]
                for s in config.sources
            ) / denom

        return weighted_post, weighted_prior

    # --------------------------------------------------------------------
    # ROW 0 — δ13C posteriors for all sources
    # --------------------------------------------------------------------
    for hem_idx in range(2):
        ax = axes[0, hem_idx]
        shade_enso(ax)

        for i, source in enumerate(config.sources):
            post_vals = iso_d13c_post[source][:, hem_idx]
            is_optimized = np.std(post_vals) > 0.1

            ax.plot(
                years,
                post_vals,
                color=colors[i],
                linewidth=2.5 if is_optimized else 1.5,
                linestyle="-" if is_optimized else "--",
                label=f"{source.capitalize()}{' *' if is_optimized else ''}",
            )

        ax.set_title(f"δ13C Posterior – {hemispheres[hem_idx]}", fontsize=12, fontweight="bold")
        ax.set_ylabel("δ13C (‰)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1))
        ax.set_xticks([])

    # --------------------------------------------------------------------
    # ROW 1 — δ13C weighted averages
    # --------------------------------------------------------------------
    for hem_idx in range(2):
        ax = axes[1, hem_idx]
        shade_enso(ax)

        w_post, w_prior = compute_weighted(iso_d13c_post, "d13c", hem_idx)

        ax.plot(years, w_prior, color="black", linestyle="--", linewidth=2.5, alpha=0.7,
                label="Avg (no opt)")
        ax.plot(years, w_post, color="black", linestyle="-", linewidth=2.5, alpha=0.9,
                label="Avg (optimized)")

        ax.set_title(f"δ13C Weighted Averages – {hemispheres[hem_idx]}", fontsize=12, fontweight="bold")
        ax.set_ylabel("δ13C (‰)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xticks([])

    # --------------------------------------------------------------------
    # ROW 2 — δD posteriors for all sources
    # --------------------------------------------------------------------
    for hem_idx in range(2):
        ax = axes[2, hem_idx]
        shade_enso(ax)

        for i, source in enumerate(config.sources):
            post_vals = iso_dd_post[source][:, hem_idx]
            is_optimized = np.std(post_vals) > 1.0

            ax.plot(
                years,
                post_vals,
                color=colors[i],
                linewidth=2.5 if is_optimized else 1.5,
                linestyle="-" if is_optimized else "--",
                label=f"{source.capitalize()}{' *' if is_optimized else ''}",
            )

        ax.set_title(f"δD Posterior – {hemispheres[hem_idx]}", fontsize=12, fontweight="bold")
        ax.set_ylabel("δD (‰)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1))
        ax.set_xticks([])

    # --------------------------------------------------------------------
    # ROW 3 — δD weighted averages
    # --------------------------------------------------------------------
    for hem_idx in range(2):
        ax = axes[3, hem_idx]
        shade_enso(ax)

        w_post, w_prior = compute_weighted(iso_dd_post, "dd", hem_idx)

        ax.plot(years, w_prior, color="black", linestyle="--", linewidth=2.5, alpha=0.7,
                label="Avg (no opt)")
        ax.plot(years, w_post, color="black", linestyle="-", linewidth=2.5, alpha=0.9,
                label="Avg (optimized)")

        ax.set_title(f"δD Weighted Averages – {hemispheres[hem_idx]}", fontsize=12, fontweight="bold")
        ax.set_ylabel("δD (‰)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xticks([])

    # --------------------------------------------------------------------
    # ROW 4 — Sink loss rates (OH, Strat, Soil)
    # --------------------------------------------------------------------
    lifetime_oh = config.lifetime_oh.reshape(1, 2)
    lifetime_strat = config.lifetime_strat.reshape(1, 2)
    lifetime_soil = config.lifetime_soil.reshape(1, 2)

    sink_oh = (1 / lifetime_oh) * (1 + sinks_post["oh"] / 100.0)
    sink_strat = (1 / lifetime_strat) * (1 + sinks_post["strat"] / 100.0)
    sink_soil = (1 / lifetime_soil) * (1 + sinks_post["soil"] / 100.0)

    for hem_idx in range(2):
        ax = axes[4, hem_idx]
        shade_enso(ax)

        ax.plot(years, sink_oh[:, hem_idx] * 1e6, label="OH sink", linewidth=2, color="red")
        ax.plot(years, sink_strat[:, hem_idx] * 1e6, label="Stratospheric", linewidth=2, color="blue")
        ax.plot(years, sink_soil[:, hem_idx] * 1e6, label="Soil", linewidth=2, color="green")

        ax.set_yscale("log")
        ax.set_ylabel("Sink (yr⁻¹ × 10⁻⁶)")
        ax.set_xlabel("Year")
        ax.set_title(f"Loss Rates – {hemispheres[hem_idx]}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # --------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f4_optimized_timeseries.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print("  ✓ Saved: f4_optimized_timeseries.png")

def save_f3_csv(years, emis_post, sinks_post, config, output_dir):
    """
    Save F3 hemispheric emissions to CSV in the specified format.
    Columns: Param, Process, Hemisphere, then yearly values
    """
    # Create parameter string
    param_str = (f"{config.kie_oh_d13c}|{config.kie_oh_dd}|"
                f"{config.prior_isotopes['wetlands']['d13c'][0]}|{config.prior_isotopes['wetlands']['d13c'][1]}|"
                f"{config.prior_isotopes['wetlands']['dd'][0]}|{config.prior_isotopes['wetlands']['dd'][1]}|"
                f"{config.prior_isotopes['agriculture']['d13c'][0]}|{config.prior_isotopes['agriculture']['d13c'][1]}|"
                f"{config.prior_isotopes['agriculture']['dd'][0]}|{config.prior_isotopes['agriculture']['dd'][1]}|"
                f"{config.prior_isotopes['pyrogenic']['d13c'][0]}|{config.prior_isotopes['pyrogenic']['d13c'][1]}|"
                f"{config.prior_isotopes['pyrogenic']['dd'][0]}|{config.prior_isotopes['pyrogenic']['dd'][1]}|"
                f"{config.prior_isotopes['fossil']['d13c'][0]}|{config.prior_isotopes['fossil']['d13c'][1]}|"
                f"{config.prior_isotopes['fossil']['dd'][0]}|{config.prior_isotopes['fossil']['dd'][1]}|"
                f"{config.prior_isotopes['waste']['d13c'][0]}|{config.prior_isotopes['waste']['d13c'][1]}|"
                f"{config.prior_isotopes['waste']['dd'][0]}|{config.prior_isotopes['waste']['dd'][1]}|"
                f"101|{config.lifetime_oh[0]}|{config.prior_emission_error}|{config.prior_emission_error}|"
                f"9|{config.prior_sink_error}|{config.obs_sigma_d13c}|{config.obs_sigma_dd}|"
                f"{config.tau_ihex}|{config.tg2ppb}|True|"
                "{'wetlands': {'d13c': True, 'dd': True}, 'pyrogenic': {'d13c': True, 'dd': True}, "
                "'agriculture': {'d13c': True, 'dd': True}, 'fossil': {'d13c': True, 'dd': True}, "
                "'waste': {'d13c': True, 'dd': True}}")
    
    # Build data
    data = {'Param': [], 'Process': [], 'Hemisphere': []}
    for year in years:
        data[str(year)] = []
    
    # Add emission rows
    for source in config.sources:
        for hem_idx, hem in enumerate(['sh', 'nh']):
            data['Param'].append(param_str)
            data['Process'].append(source)
            data['Hemisphere'].append(hem)
            for year_idx in range(len(years)):
                data[str(years[year_idx])].append(emis_post[source][year_idx, hem_idx])
    
    # Add sink rows (compute lifetime)
    post_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + sinks_post['oh'] / 100.0)
    post_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + sinks_post['strat'] / 100.0)
    post_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + sinks_post['soil'] / 100.0)
    post_total_sink_rate = post_sink_oh + post_sink_strat + post_sink_soil
    post_lifetime = 1.0 / post_total_sink_rate
    
    for hem_idx, hem in enumerate(['sh', 'nh']):
        data['Param'].append(param_str)
        data['Process'].append('sink')
        data['Hemisphere'].append(hem)
        for year_idx in range(len(years)):
            data[str(years[year_idx])].append(post_lifetime[year_idx, hem_idx])
    
    # Save to CSV
    df = pd.DataFrame(data)
    df = df.transpose()
    df.to_csv(os.path.join(output_dir, 'hemposte_values_em&iso_varying.csv'), index=True)
    
    print(f"  ✓ Saved: f3_hemposte_values.csv")


def calculate_total_lifetime(sinks, config):
    """
    Convert OH/stratospheric/soil sink perturbations (%) to total CH4 lifetime.

    Returns an array with shape (n_years, 2), where columns are SH and NH.
    """
    sink_oh = (1 / config.lifetime_oh.reshape(1, 2)) * (1 + sinks['oh'] / 100.0)
    sink_strat = (1 / config.lifetime_strat.reshape(1, 2)) * (1 + sinks['strat'] / 100.0)
    sink_soil = (1 / config.lifetime_soil.reshape(1, 2)) * (1 + sinks['soil'] / 100.0)
    total_sink_rate = sink_oh + sink_strat + sink_soil
    return 1.0 / total_sink_rate


def save_lifetime_csv(years, prior_sinks, sinks_post, config, output_dir):
    """
    Save Script 2 total lifetime for comparison_emissions_both_scripts.py.

    Output format mirrors emissions_{source}.csv:
        Year, Prior_SH, Prior_NH, Posterior_SH, Posterior_NH
    """
    prior_lifetime = calculate_total_lifetime(prior_sinks, config)
    post_lifetime = calculate_total_lifetime(sinks_post, config)

    df = pd.DataFrame({
        'Year': years,
        'Prior_SH': prior_lifetime[:, 0],
        'Prior_NH': prior_lifetime[:, 1],
        'Posterior_SH': post_lifetime[:, 0],
        'Posterior_NH': post_lifetime[:, 1],
    })
    df.to_csv(os.path.join(output_dir, 'lifetime.csv'), index=False)

    print(f"  ✓ Saved: lifetime.csv")
# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()
