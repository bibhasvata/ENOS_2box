"""
1. Correlation Tests
Spearman Rank Correlation (variable: spearman_r, spearman_p)
What it measures: Monotonic relationship between MEI (ENSO index) and emission
Why use it: Non-parametric, robust to outliers and non-linear relationships
Interpretation:

Pearson Correlation (variable: pearson_r, pearson_p)
What it measures: Linear relationship between MEI and emissions
Why use it: Captures strength of linear trends
When it differs from Spearman: Suggests non-linear relationships or outlier effects

r = +1: perfect positive correlation (higher MEI → higher emissions)
r = -1: perfect negative correlation
p < 0.05: statistically significant relationship
==========================================================================================================================
2. Group Comparison Tests
Mann-Whitney U Test (variable: mann_whitney_stat, mann_whitney_p)
What it measures: Whether emissions differ between El Niño and La Niña years
Why use it: Non-parametric alternative to t-test, doesn't assume normal distribution
Interpretation:
p < 0.05: emissions during El Niño years are significantly different from La Niña years
Lower p-value = stronger evidence of difference
==========================================================================================================================
3. Effect Size Metrics
Cohen's d (variable: cohens_d)
What it measures: Magnitude of difference between El Niño and La Niña emissions
Formula: (Mean_ElNiño - Mean_LaNiña) / Pooled_StdDev
Interpretation:
|d| < 0.2: small effect 
|d| = 0.5: medium effect
|d| > 0.8: large effect
Why important: Statistical significance (p-value) doesn't tell you if the effect is meaningful; Cohen's d does
==========================================================================================================================
4. Phase-Specific Means
(variable: mean_el_nino, mean_la_nina, mean_neutral, emission_diff_elnino_lanina / isotope_shift_elnino_lanina)
What they measure: Average emissions/isotope values during each ENSO phase
Why useful: Shows directional changes (e.g., wetlands higher during El Niño)
Threshold: Years classified by MEI > 0.5 (El Niño), < -0.5 (La Niña), or in between (Neutral)
What it measures: Raw difference between El Niño and La Niña means
Units: Same as the variable (Tg/yr for emissions, ‰ for isotopes)
Sign interpretation:
Positive: higher during El Niño
Negative: higher during La Niña
==========================================================================================================================
5. Temporal Dynamics
Lagged Correlation (variable: max_lagged_corr, best_lag)
What it measures: Whether emissions respond to ENSO with a time delay
Method: Tests correlations with 0-2 year lags
Interpretation:
best_lag = 0: immediate response (<1 years)
best_lag = 1: emissions respond 1 year after ENSO event
Why important: Physical processes (e.g., soil moisture changes) may lag climate anomalies
==========================================================================================================================
6. Sample Sizes
(variable: n_el_nino_years, n_la_nina_years, n_neutral_years)
What they measure: Number of years in each ENSO category (1994-2022)
Why important: Statistical power depends on sample size
Typical values: ~5-8 strong events of each type in the analysis period

Author: Bibhasvata Dasgupta
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
import pickle
from datetime import datetime
from itertools import product
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import the main inversion script
from twobox_simultaneous_isotope_emission_cost_func_time_varying_inversion_v4 import (
    InversionConfig, StateVector, ForwardModel, 
    load_prior_emissions, load_observations, get_prior_isotopes,
    two_step_inversion, get_observation_vector, get_observation_covariance,
    observation_operator
)

# Import chi-square diagnostics (optional)
try:
    from inversion_diagnostics_chisquare import ChiSquareMetrics
    from enhanced_chi2_diagnostics import EnhancedChiSquareMetrics
    HAS_CHI2 = True
except ImportError:
    HAS_CHI2 = False
    print("⚠ Chi-square modules not found - will skip chi2 diagnostics")

# Import MEI sensitivity analyzer (optional)
try:
    from enhanced_enso_analyzer import EnhancedENSOAnalyzer, format_sensitivity_report
    HAS_MEI_SENSITIVITY = True
except ImportError:
    HAS_MEI_SENSITIVITY = False
    print("⚠ Enhanced ENSO analyzer not found - will skip MEI sensitivity")

# ============================================================================
# ENSO CAUSALITY ANALYSIS
# ============================================================================

class ENSOCausalityAnalyzer:
    """Analyze relationship between ENSO (MEI) and emissions."""
    
    def __init__(self, mei_df):
        self.mei_df = mei_df
        
    def classify_enso_years(self, threshold=0.5):
        """Classify years as El Niño, La Niña, or Neutral."""
        el_nino = self.mei_df[self.mei_df['MEI'] > threshold]['Year'].values
        la_nina = self.mei_df[self.mei_df['MEI'] < -threshold]['Year'].values
        neutral = self.mei_df[
            (self.mei_df['MEI'] >= -threshold) & 
            (self.mei_df['MEI'] <= threshold)
        ]['Year'].values
        
        return {
            'el_nino': el_nino,
            'la_nina': la_nina,
            'neutral': neutral
        }
    
    def compute_causality_metrics(self, years, emissions, source_name):
        """Compute multiple causality metrics between MEI and emissions."""
        df = pd.DataFrame({'Year': years, 'Emissions': emissions})
        df = df.merge(self.mei_df, on='Year', how='inner')
        
        if len(df) < 10:
            return self._empty_metrics()
        
        # 1. Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(df['MEI'], df['Emissions'])
        
        # 2. Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(df['MEI'], df['Emissions'])
        
        # 3. Mann-Whitney U test
        enso_years = self.classify_enso_years(threshold=0.5)
        
        el_nino_mask = df['Year'].isin(enso_years['el_nino'])
        la_nina_mask = df['Year'].isin(enso_years['la_nina'])
        
        el_nino_emissions = df.loc[el_nino_mask, 'Emissions'].values
        la_nina_emissions = df.loc[la_nina_mask, 'Emissions'].values
        
        if len(el_nino_emissions) > 0 and len(la_nina_emissions) > 0:
            mw_stat, mw_p = stats.mannwhitneyu(
                el_nino_emissions, la_nina_emissions, alternative='two-sided'
            )
            
            mean_diff = np.mean(el_nino_emissions) - np.mean(la_nina_emissions)
            pooled_std = np.sqrt(
                (np.var(el_nino_emissions) + np.var(la_nina_emissions)) / 2
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            mean_el_nino = np.mean(el_nino_emissions)
            mean_la_nina = np.mean(la_nina_emissions)
            mean_neutral = np.mean(
                df.loc[df['Year'].isin(enso_years['neutral']), 'Emissions'].values
            ) if len(enso_years['neutral']) > 0 else np.nan
            
        else:
            mw_stat, mw_p = np.nan, np.nan
            cohens_d = np.nan
            mean_el_nino, mean_la_nina, mean_neutral = np.nan, np.nan, np.nan
        
        # 4. Lagged correlation
        max_lag_corr = 0
        best_lag = 0
        
        for lag in range(0, 3):
            if lag < len(df):
                mei_lagged = df['MEI'].values[:-lag] if lag > 0 else df['MEI'].values
                emis_lagged = df['Emissions'].values[lag:] if lag > 0 else df['Emissions'].values
                
                if len(mei_lagged) > 3:
                    corr, _ = stats.spearmanr(mei_lagged, emis_lagged)
                    if abs(corr) > abs(max_lag_corr):
                        max_lag_corr = corr
                        best_lag = lag
        
        return {
            'source': source_name,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'mann_whitney_stat': mw_stat,
            'mann_whitney_p': mw_p,
            'cohens_d': cohens_d,
            'mean_el_nino': mean_el_nino,
            'mean_la_nina': mean_la_nina,
            'mean_neutral': mean_neutral,
            'std_el_nino': float(np.std(el_nino_emissions, ddof=1)) if len(el_nino_emissions) > 1 else np.nan,
            'std_la_nina': float(np.std(la_nina_emissions, ddof=1)) if len(la_nina_emissions) > 1 else np.nan,
            'n_el_nino_years': len(el_nino_emissions),
            'n_la_nina_years': len(la_nina_emissions),
            'emission_diff_elnino_lanina': mean_el_nino - mean_la_nina,
            'max_lagged_corr': max_lag_corr,
            'best_lag': best_lag,
            'n_neutral_years': len(enso_years['neutral'])
        }
    
    def compute_isotope_enso_shift(self, years, isotope_values, source_name, isotope_type):
        """Test if isotope values differ between ENSO phases."""
        df = pd.DataFrame({'Year': years, 'Isotope': isotope_values})
        df = df.merge(self.mei_df, on='Year', how='inner')
        
        if len(df) < 10:
            return self._empty_isotope_metrics()
        
        enso_years = self.classify_enso_years(threshold=0.5)
        
        el_nino_mask = df['Year'].isin(enso_years['el_nino'])
        la_nina_mask = df['Year'].isin(enso_years['la_nina'])
        neutral_mask = df['Year'].isin(enso_years['neutral'])
        
        el_nino_iso = df.loc[el_nino_mask, 'Isotope'].values
        la_nina_iso = df.loc[la_nina_mask, 'Isotope'].values
        neutral_iso = df.loc[neutral_mask, 'Isotope'].values
        
        if len(el_nino_iso) > 0 and len(la_nina_iso) > 0:
            mw_stat, mw_p = stats.mannwhitneyu(
                el_nino_iso, la_nina_iso, alternative='two-sided'
            )
            
            mean_diff = np.mean(el_nino_iso) - np.mean(la_nina_iso)
            pooled_std = np.sqrt(
                (np.var(el_nino_iso) + np.var(la_nina_iso)) / 2
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            mean_el_nino = np.mean(el_nino_iso)
            mean_la_nina = np.mean(la_nina_iso)
            mean_neutral = np.mean(neutral_iso) if len(neutral_iso) > 0 else np.nan
            
            spearman_r, spearman_p = stats.spearmanr(df['MEI'], df['Isotope'])
            
        else:
            mw_stat, mw_p = np.nan, np.nan
            cohens_d = np.nan
            mean_el_nino, mean_la_nina, mean_neutral = np.nan, np.nan, np.nan
            spearman_r, spearman_p = np.nan, np.nan
        
        return {
            'source': source_name,
            'isotope_type': isotope_type,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mann_whitney_p': mw_p,
            'cohens_d': cohens_d,
            'mean_el_nino': mean_el_nino,
            'mean_la_nina': mean_la_nina,
            'mean_neutral': mean_neutral,
            'std_el_nino': float(np.std(el_nino_iso, ddof=1)) if len(el_nino_iso) > 1 else np.nan,
            'std_la_nina': float(np.std(la_nina_iso, ddof=1)) if len(la_nina_iso) > 1 else np.nan,
            'n_el_nino_years': len(el_nino_iso),
            'n_la_nina_years': len(la_nina_iso),
            'isotope_shift_elnino_lanina': mean_el_nino - mean_la_nina,
            'n_neutral_years': len(neutral_iso)
        }
    
    def _empty_isotope_metrics(self):
        return {
            'source': '', 'isotope_type': '',
            'spearman_r': np.nan, 'spearman_p': np.nan,
            'mann_whitney_p': np.nan, 'cohens_d': np.nan,
            'mean_el_nino': np.nan, 'mean_la_nina': np.nan, 'mean_neutral': np.nan,
            'std_el_nino': np.nan, 'std_la_nina': np.nan,
            'isotope_shift_elnino_lanina': np.nan,
            'n_el_nino_years': 0, 'n_la_nina_years': 0, 'n_neutral_years': 0
        }
    
    def _empty_metrics(self):
        return {
            'source': '',
            'spearman_r': np.nan, 'spearman_p': np.nan,
            'pearson_r': np.nan, 'pearson_p': np.nan,
            'mann_whitney_stat': np.nan, 'mann_whitney_p': np.nan,
            'cohens_d': np.nan,
            'mean_el_nino': np.nan, 'mean_la_nina': np.nan, 'mean_neutral': np.nan,
            'std_el_nino': np.nan, 'std_la_nina': np.nan,
            'emission_diff_elnino_lanina': np.nan,
            'max_lagged_corr': np.nan, 'best_lag': np.nan,
            'n_el_nino_years': 0, 'n_la_nina_years': 0, 'n_neutral_years': 0
        }

# ============================================================================
# PARAMETER CONFIGURATIONS
# ============================================================================

class ParameterSets:
    """Define different parameter exploration strategies."""
    
    @staticmethod
    def full_grid_small():
        # Focus on making δD respond to ENSO

        prior_d13c = [1.5, 2, 2.5]          # Keep moderate (working well)
        prior_dd = [8, 10, 12]            # Matches InversionConfig default (1.0 ‰); do NOT inflate to 10 —
                                  # a 10× looser δD prior removes δD's ability to constrain the
                                  # fossil/biogenic split and shifts all isotope residuals onto
                                  # emission adjustments instead of source-signature adjustments.

        obs_d13c = [0.04, 0.05, 0.06]          # Keep moderate (working well)
        obs_dd = [0.5]         # TIGHTER (was 2.0, add tighter options)

        prior_emission = [0.3, 0.35]   # Slightly looser (allow more ENSO)

        smooth_em = [10]          # Keep moderate (working)
        smooth_iso_d13c = [0.1]    # Keep for δ13C (working well)
        smooth_iso_dd = [0.01] # MUCH WEAKER for δD (was 1.0, reduce 5-20×!)

        # ADD this line with the other parameter lists
        tau_ihex_vals = [0.65, 0.75, 0.85]            
        variations = []

        # REPLACE the for loop signature:
        for pd13, pdd, od13, odd, pem, sem, siso13c, sisodd, tau in product(
            prior_d13c, prior_dd, obs_d13c, obs_dd, prior_emission, smooth_em, smooth_iso_d13c, smooth_iso_dd, tau_ihex_vals
        ):
            config = {
                'prior_isotope_d13c_error': pd13,
                'prior_isotope_dd_error': pdd,
                'obs_sigma_d13c': od13,
                'obs_sigma_dd': odd,
                'prior_emission_error': pem,
                'smoothness_lambda_emissions': sem,
                'smoothness_lambda_d13c': siso13c,
                'smoothness_lambda_dd': sisodd,
                'tau_ihex': tau,   # ADD this line
                'name': f'prior_δ13C={pd13}|δD={pdd}|obs_δ13C={od13}|δD={odd}|prior_emis_err={pem}|λ_emis={sem}|λ_d13C={siso13c}|λ_dD={sisodd}|tau_ihex={tau}'
            }
            variations.append(config)
        
        print(f"  Grid size: {len(variations)}")
        return variations

# ============================================================================
# RUN MANAGER
# ============================================================================

class InversionRunner:
    """Manage multiple inversion runs."""
    
    def __init__(self, output_dir='multi_inversion_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = []
        self.config_base = InversionConfig()
        
        print("\n=== Loading shared data ===")
        self.prior_emissions = load_prior_emissions(self.config_base)
        self.observations = load_observations(self.config_base)
        self.prior_isotopes = get_prior_isotopes(self.config_base)
        
        self.years = np.arange(self.config_base.year1, self.config_base.year2 + 1)
        self.n_years = len(self.years)
        
        self.init_conditions = {
            'ch4': np.array([self.observations['ch4']['sh'][0], 
                            self.observations['ch4']['nh'][0]]),
            'd13c': np.array([self.observations['d13c']['sh'][0], 
                             self.observations['d13c']['nh'][0]]),
            'dd': np.array([self.observations['dd']['sh'][0], 
                           self.observations['dd']['nh'][0]])
        }
        
        self.mei_df = self.load_mei_data()
        self.enso_analyzer = ENSOCausalityAnalyzer(self.mei_df)
        
        # Initialize enhanced analyzer for MEI sensitivity
        if HAS_MEI_SENSITIVITY:
            self.enhanced_enso_analyzer = EnhancedENSOAnalyzer(self.mei_df)
            print("✓ Enhanced MEI sensitivity analysis enabled")
        else:
            self.enhanced_enso_analyzer = None

    def load_mei_data(self):
        """Load MEI data for ENSO analysis."""
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
            mei_yearly = mei_long.groupby('Year')['Anomaly'].mean().reset_index()
            mei_yearly.rename(columns={'Anomaly': 'MEI'}, inplace=True)
            print(f"✓ Loaded MEI data: {len(mei_yearly)} years")
            return mei_yearly
        except Exception as e:
            print(f"⚠ MEI load failed: {e}")
            return pd.DataFrame({'Year': self.years, 'MEI': np.zeros(len(self.years))})
    
    def compute_enso_causality(self, result):
        """Approach 1: compute_enso_causality (Categorical Analysis)
            Classifies years into discrete ENSO phases: El Niño (MEI > 0.5), La Niña (MEI < -0.5), Neutral
            Compares emissions/isotopes between these categories
            Tests whether there are statistically significant differences between phases

            Key Metrics:
            1. Mean values per phase: Average emissions during El Niño, La Niña, and Neutral years
            2. Emission difference: El Niño mean - La Niña mean (the "excess" you mentioned)
            3. Mann-Whitney U test: Tests if El Niño vs La Niña distributions differ significantly
            4. Cohen's d: Effect size of the difference (how large is the shift?)
            5. Spearman/Pearson correlation: Overall monotonic/linear relationship
            6. Lagged correlation: Tests for time-delayed responses (0-2 year lags)

            Strengths:
            ✅ Intuitive interpretation - "Emissions are 13 Tg/yr higher during El Niño years"
            ✅ Non-parametric - Doesn't assume linear relationships
            ✅ Captures extremes - Focuses on strong El Niño/La Niña events
            ✅ Statistical rigor - Multiple tests (correlation, group comparison, effect size)
            ✅ Temporal dynamics - Tests for lagged responses

            Limitations:
            ❌ Binary categorization - Discards information from neutral years and moderate events
            ❌ Threshold dependent - Results can change with different MEI thresholds (0.5 is arbitrary)
            ❌ No continuous relationship - Can't predict emissions for arbitrary MEI values
            ❌ Sample size - Only ~5-8 events per category limits statistical power
            """
        emissions = result['emissions']
        iso_d13c = result['iso_d13c']
        iso_dd = result['iso_dd']

        year_mask = (self.years >= 1994) & (self.years <= 2022)
        years_subset = self.years[year_mask]

        causality_results = {}

        # ---- 1. All 5 INDIVIDUAL SOURCES (SH, NH, and GLOBAL) ----
        all_sources = ['wetlands', 'agriculture', 'pyrogenic', 'fossil', 'waste']

        for source in all_sources:
            if source in emissions:
                # Extract hemisphere-specific and global emissions
                emis_sh = emissions[source][year_mask, 0]
                emis_nh = emissions[source][year_mask, 1]
                emis_global = emis_sh + emis_nh

                # Compute metrics for each hemisphere separately and global
                for hem, emis_values in [('sh', emis_sh), ('nh', emis_nh), ('global', emis_global)]:
                    key = f'{source}_emissions' if hem == 'global' else f'{source}_emissions_{hem}'
                    metrics = self.enso_analyzer.compute_causality_metrics(
                        years_subset, emis_values, f'{source}_{hem}'
                    )
                    causality_results[key] = metrics
            else:
                # Save placeholder to ensure consistent output
                for hem in ['sh', 'nh', 'global']:
                    key = f'{source}_emissions' if hem == 'global' else f'{source}_emissions_{hem}'
                    causality_results[key] = None

        """# ---- 2. COMBINED WETLANDS + PYROGENIC (SH, NH, and GLOBAL) ----
        if 'wetlands' in emissions and 'pyrogenic' in emissions:
            # SH hemisphere
            emis_wetlands_sh = emissions['wetlands'][year_mask, 0]
            emis_pyrogenic_sh = emissions['pyrogenic'][year_mask, 0]
            emis_combined_sh = emis_wetlands_sh + emis_pyrogenic_sh

            # NH hemisphere
            emis_wetlands_nh = emissions['wetlands'][year_mask, 1]
            emis_pyrogenic_nh = emissions['pyrogenic'][year_mask, 1]
            emis_combined_nh = emis_wetlands_nh + emis_pyrogenic_nh

            # Global
            emis_combined_global = emis_combined_sh + emis_combined_nh

            # Compute metrics for each
            for hem, emis_values in [('sh', emis_combined_sh), ('nh', emis_combined_nh), ('global', emis_combined_global)]:
                combined_key = 'wetlands_pyrogenic_emissions' if hem == 'global' else f'wetlands_pyrogenic_emissions_{hem}'
                metrics = self.enso_analyzer.compute_causality_metrics(
                    years_subset, emis_values, f'wetlands_pyrogenic_{hem}'
                )
                causality_results[combined_key] = metrics
        else:
            for hem in ['sh', 'nh', 'global']:
                combined_key = 'wetlands_pyrogenic_emissions' if hem == 'global' else f'wetlands_pyrogenic_emissions_{hem}'
                causality_results[combined_key] = None"""

        # ---- 3. ISOTOPE SHIFTS (unchanged, but now after emissions block) ----
        for source in ['wetlands', 'agriculture', 'pyrogenic', 'fossil', 'waste']:
            if source in iso_d13c and source in iso_dd:

                for isotope, iso_dict in [('d13c', iso_d13c), ('dd', iso_dd)]:
                    sh = iso_dict[source][year_mask, 0]
                    nh = iso_dict[source][year_mask, 1]

                    # FIX: use emission-weighted global mean rather than simple (sh+nh)/2.
                    # For sources with strongly asymmetric hemispheric magnitudes (e.g.
                    # fossil fuels, which are NH-dominant) a simple average misrepresents
                    # the true global mean isotope signature.
                    if source in emissions:
                        E_sh = emissions[source][year_mask, 0]
                        E_nh = emissions[source][year_mask, 1]
                        E_tot = E_sh + E_nh
                        # Guard against zero-emission years
                        with np.errstate(invalid='ignore', divide='ignore'):
                            glob = np.where(E_tot > 0,
                                            (sh * E_sh + nh * E_nh) / E_tot,
                                            (sh + nh) / 2)
                    else:
                        glob = (sh + nh) / 2  # fallback if emissions not available

                    for hem, values in [('sh', sh), ('nh', nh), ('global', glob)]:

                        metrics = self.enso_analyzer.compute_isotope_enso_shift(
                            years_subset, values, f'{source}_{hem}', isotope
                        )
                        causality_results[f'{source}_{isotope}_{hem}'] = metrics

        return causality_results
    
    def compute_mei_sensitivities(self, result):
        """
        Approach 2: compute_mei_sensitivities (Linear Regression)

        Fits a linear model: Emission = β₀ + β₁ × MEI + ε
        Calculates the sensitivity coefficient (slope): How much emissions change per unit MEI
        Uses all years continuously rather than discrete categories

        Key Metrics:
        1. Slope (β₁): Emission change per unit MEI increase (e.g., +5.2 Tg/yr per MEI unit)
        2. R²: How much variance in emissions is explained by MEI
        3. Standard error: Uncertainty in the slope estimate
        4. p-value: Statistical significance of the relationship
        5. El Niño/La Niña sensitivity: Predicted emissions at MEI = +1 and -1

        Strengths:
        ✅ Continuous relationship - Uses full MEI spectrum (not just extremes)
        ✅ Predictive - Can estimate emissions for any MEI value
        ✅ Quantitative equation - Gives a general formula as you mentioned
        ✅ Efficient use of data - Uses all 29 years (1994-2022), not just ~10-16 extreme years
        ✅ Comparable across studies - Standard metric (Tg/MEI or ‰/MEI)

        Limitations:
        ❌ Assumes linearity - May miss non-linear ENSO-emission relationships
        ❌ Sensitive to outliers - Extreme years can disproportionately affect slope
        ❌ May underestimate extremes - Linear fit averages over weak and strong events
        ❌ Ignores phase asymmetry - Assumes El Niño and La Niña responses are symmetric
        """
        if self.enhanced_enso_analyzer is None:
            return None
        
        # Prepare data (1994-2022 only)
        year_mask = (self.years >= 1994) & (self.years <= 2022)
        years_subset = self.years[year_mask]
        
        emissions = result['emissions']
        iso_d13c = result['iso_d13c']
        iso_dd = result['iso_dd']
        
        sources = ['wetlands', 'pyrogenic', 'agriculture', 'fossil', 'waste']
        
        # Initialize nested dictionaries for all regions
        emissions_dict = {'nh': {}, 'sh': {}, 'global': {}}
        isotopes_d13c_dict = {'nh': {}, 'sh': {}, 'global': {}}
        isotopes_dd_dict = {'nh': {}, 'sh': {}, 'global': {}}
        
        # Populate emissions for all regions
        for source in sources:
            if source in emissions:
                nh_emis = emissions[source][:, 1][year_mask]
                sh_emis = emissions[source][:, 0][year_mask]
                global_emis = nh_emis + sh_emis
                
                emissions_dict['nh'][source] = nh_emis
                emissions_dict['sh'][source] = sh_emis
                emissions_dict['global'][source] = global_emis
        
        # Populate isotopes for all regions
        for source in sources:
            if source in iso_d13c:
                # δ13C
                nh_d13c = iso_d13c[source][:, 1][year_mask]
                sh_d13c = iso_d13c[source][:, 0][year_mask]

                # FIX: emission-weighted global isotope (same as compute_enso_causality)
                if source in emissions:
                    E_sh = emissions[source][:, 0][year_mask]
                    E_nh = emissions[source][:, 1][year_mask]
                    E_tot = E_sh + E_nh
                    with np.errstate(invalid='ignore', divide='ignore'):
                        global_d13c = np.where(E_tot > 0,
                                               (sh_d13c * E_sh + nh_d13c * E_nh) / E_tot,
                                               (sh_d13c + nh_d13c) / 2)
                else:
                    global_d13c = (sh_d13c + nh_d13c) / 2

                isotopes_d13c_dict['nh'][source] = nh_d13c
                isotopes_d13c_dict['sh'][source] = sh_d13c
                isotopes_d13c_dict['global'][source] = global_d13c

                # δD
                nh_dd = iso_dd[source][:, 1][year_mask]
                sh_dd = iso_dd[source][:, 0][year_mask]

                if source in emissions:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        global_dd = np.where(E_tot > 0,
                                             (sh_dd * E_sh + nh_dd * E_nh) / E_tot,
                                             (sh_dd + nh_dd) / 2)
                else:
                    global_dd = (sh_dd + nh_dd) / 2

                isotopes_dd_dict['nh'][source] = nh_dd
                isotopes_dd_dict['sh'][source] = sh_dd
                isotopes_dd_dict['global'][source] = global_dd
        
        # Compute sensitivities for each region
        mei_results = {
            'emissions': {},
            'isotopes_d13c': {},
            'isotopes_dd': {}
        }
        
        for region in ['nh', 'sh', 'global']:
            region_results = self.enhanced_enso_analyzer.analyze_all_sources(
                years_subset, 
                emissions_dict[region], 
                isotopes_d13c_dict[region], 
                isotopes_dd_dict[region]
            )
            
            mei_results['emissions'][region] = region_results['emissions']
            mei_results['isotopes_d13c'][region] = region_results['isotopes_d13c']
            mei_results['isotopes_dd'][region] = region_results['isotopes_dd']
        # -------------------------------------------------------------
        # Lifetime MEI sensitivity (NH, SH, Global)
        # Stores results under mei_results['lifetime'][region]
        # -------------------------------------------------------------
        try:
            # Build yearly lifetime arrays for 1994–2022
            year_mask = (self.years >= 1994) & (self.years <= 2022)
            years_subset = self.years[year_mask]

            # Prefer saved lifetime if present, otherwise compute from sinks+config
            if 'lifetime' in result and result['lifetime'] is not None and 'by_hem' in result['lifetime']:
                tau_by_hem = np.asarray(result['lifetime']['by_hem'])
                tau_sh = tau_by_hem[:, 0][year_mask]
                tau_nh = tau_by_hem[:, 1][year_mask]
            else:
                cfg = result['config']
                sinks = result['sinks']

                k_oh0 = 1.0 / np.asarray(cfg.lifetime_oh).reshape(1, 2)
                k_st0 = 1.0 / np.asarray(cfg.lifetime_strat).reshape(1, 2)
                k_so0 = 1.0 / np.asarray(cfg.lifetime_soil).reshape(1, 2)

                k_oh = k_oh0 * (1.0 + sinks['oh'] / 100.0)
                k_st = k_st0 * (1.0 + sinks['strat'] / 100.0)
                k_so = k_so0 * (1.0 + sinks['soil'] / 100.0)

                tau_by_hem = 1.0 / (k_oh + k_st + k_so)  # (nyears,2) [SH,NH]
                tau_sh = tau_by_hem[:, 0][year_mask]
                tau_nh = tau_by_hem[:, 1][year_mask]

            tau_global = 0.5 * (tau_sh + tau_nh)

            # MEI series aligned to years
            mei_df = self.mei_df if hasattr(self, 'mei_df') else None
            if mei_df is None:
                raise RuntimeError("self.mei_df is missing; cannot compute lifetime MEI sensitivity.")

            df_tau = pd.DataFrame({
                'Year': years_subset,
                'sh': tau_sh,
                'nh': tau_nh,
                'global': tau_global
            }).merge(mei_df[['Year', 'MEI']], on='Year', how='inner').dropna()

            def _linreg(mei, y):
                mei = np.asarray(mei, float)
                y = np.asarray(y, float)
                m = np.isfinite(mei) & np.isfinite(y)
                mei, y = mei[m], y[m]
                if len(mei) < 3:
                    return None
                lr = stats.linregress(mei, y)
                # keep fields consistent/simple for plotting
                return {
                    'slope': float(lr.slope),
                    'intercept': float(lr.intercept),
                    'r': float(lr.rvalue),
                    'p': float(lr.pvalue),
                    'std_error': float(lr.stderr),
                    'n': int(len(mei))
                }

            mei_results['lifetime'] = {
                'nh': {'total': _linreg(df_tau['MEI'].values, df_tau['nh'].values)},
                'sh': {'total': _linreg(df_tau['MEI'].values, df_tau['sh'].values)},
                'global': {'total': _linreg(df_tau['MEI'].values, df_tau['global'].values)},
            }

        except Exception as e:
            # Don't crash the whole inversion if lifetime sensitivity fails
            mei_results['lifetime'] = None
            print(f"⚠ Lifetime MEI sensitivity failed: {e}")

        # Add summary table for backward compatibility (use global)
        if 'summary_table' in region_results:
            mei_results['summary_table'] = region_results['summary_table']
        
        return mei_results

    def compute_yearly_lifetime(self, sinks_post, config):
        """
        Compute yearly methane lifetime (years) from posterior sink scalings.

        sinks_post: dict with keys 'oh','strat','soil' as (% anomaly) arrays of shape (nyears, 2)
                columns are [SH, NH] (consistent with your plotting code).
        config: InversionConfig with baseline lifetimes: lifetime_oh, lifetime_strat, lifetime_soil (length-2 arrays)

        Returns:
        lifetime: dict with:
            - 'sh'      : (nyears,) lifetime in SH
            - 'nh'      : (nyears,) lifetime in NH
            - 'global'  : (nyears,) simple mean of NH/SH
            - 'by_hem'  : (nyears,2) [SH,NH]
            - 'sink_total' : (nyears,2) total sink rate (1/yr)
            - 'sink_components' : dict of component sink rates (1/yr)
        """
        # Baseline sink rates (1/yr)
        k_oh0 = 1.0 / np.asarray(config.lifetime_oh).reshape(1, 2)
        k_st0 = 1.0 / np.asarray(config.lifetime_strat).reshape(1, 2)
        k_so0 = 1.0 / np.asarray(config.lifetime_soil).reshape(1, 2)

        # Posterior sink rates apply scaling anomalies (%)
        k_oh = k_oh0 * (1.0 + sinks_post['oh'] / 100.0)
        k_st = k_st0 * (1.0 + sinks_post['strat'] / 100.0)
        k_so = k_so0 * (1.0 + sinks_post['soil'] / 100.0)

        k_total = k_oh + k_st + k_so
        tau = 1.0 / k_total  # (nyears,2) years

        sh = tau[:, 0]
        nh = tau[:, 1]
        glb = 0.5 * (sh + nh)

        return {
            'sh': sh,
            'nh': nh,
            'global': glb,
            'by_hem': tau,
            'sink_total': k_total,
            'sink_components': {
                'oh': k_oh,
                'strat': k_st,
                'soil': k_so
            }
        }

    def run_single_inversion(self, param_overrides, run_id):
        """Run a single inversion."""
        print(f"\n{'='*70}")
        print(f"RUN {run_id}: {param_overrides['name']}")
        print(f"{'='*70}")
        
        config = InversionConfig()
        for key, value in param_overrides.items():
            if key != 'name' and hasattr(config, key):
                setattr(config, key, value)
        
        config.prior_isotopes = self.prior_isotopes 
        
        state_vec = StateVector(config)
        forward_model = ForwardModel(config)
        
        prior_iso_d13c = {}
        prior_iso_dd = {}
        for source in config.sources:
            prior_iso_d13c[source] = np.tile(
                self.prior_isotopes[source]['d13c'], (self.n_years, 1))
            prior_iso_dd[source] = np.tile(
                self.prior_isotopes[source]['dd'], (self.n_years, 1))
        
        prior_sinks = {sink: np.zeros((self.n_years, 2)) 
                      for sink in ['oh', 'strat', 'soil']}
        
        x_prior = state_vec.pack(self.prior_emissions, prior_iso_d13c, 
                                 prior_iso_dd, prior_sinks)
        
        output_prior = forward_model.run(self.prior_emissions, 
                                         prior_iso_d13c, prior_iso_dd, 
                                         prior_sinks, self.init_conditions)
        
        try:
            x_post, diagnostics = two_step_inversion(
                config, state_vec, forward_model, 
                x_prior, self.prior_emissions, self.prior_isotopes,
                self.observations, self.init_conditions
            )
            
            emis_post, iso_d13c_post, iso_dd_post, sinks_post = state_vec.unpack(x_post)
            lifetime_post = self.compute_yearly_lifetime(sinks_post, config)

            
            from scipy import linalg
            y_obs = get_observation_vector(self.observations, ch4_only=False)
            y_model = observation_operator(diagnostics['final_output'], None, ch4_only=False)
            
            year_mask = (self.years >= 1994) & (self.years <= 2022)
            n_obs_per_year = 6
            obs_mask = np.tile(year_mask, n_obs_per_year)
            
            rmse_1994_2022 = np.sqrt(np.mean((y_obs[obs_mask] - y_model[obs_mask]) ** 2))
            rmse_total = np.sqrt(np.mean((y_obs - y_model) ** 2))
            
            Sy = get_observation_covariance(config, self.n_years, ch4_only=False)
            innovation = y_obs - y_model
            cost_obs = innovation.T @ linalg.inv(Sy) @ innovation
            
            dx = x_post - x_prior
            from twobox_simultaneous_isotope_emission_cost_func_time_varying_inversion_v4 import (
                two_step_construct_prior_covariance, compute_smoothness_penalty
            )
            Sa = two_step_construct_prior_covariance(config, state_vec, 
                                                      self.prior_emissions, 
                                                      self.prior_isotopes)
            cost_prior = dx.T @ linalg.inv(Sa) @ dx
            
            cost_smooth, _, _ = compute_smoothness_penalty(
                x_post, x_prior, config, state_vec, D_matrices=None
            )
            
            cost_total = diagnostics['costs'][-1]
            cost_reduction = diagnostics['costs'][0] - diagnostics['costs'][-1]
            
            # Compute chi-square diagnostics if module available
            if HAS_CHI2:
                n_obs = len(y_obs)
                n_state = len(x_post)
                
                # Chi2 for observations
                chi2_obs_val = cost_obs / n_obs
                chi2_obs_total = {
                    'chi2_obs': chi2_obs_val,
                    'n_obs': n_obs,
                    'interpretation': 'GOOD' if 0.5 <= chi2_obs_val <= 1.5 else 
                                     'UNDERFIT' if chi2_obs_val > 1.5 else 'OVERFIT'
                }
                
                # Chi2 for prior
                chi2_prior_val = cost_prior / n_state
                chi2_prior_total = {
                    'chi2_prior': chi2_prior_val,
                    'n_state': n_state,
                    'interpretation': 'GOOD' if 0.5 <= chi2_prior_val <= 1.5 else
                                     'TOO FAR' if chi2_prior_val > 1.5 else 'TOO CLOSE'
                }
                
                # Chi2 by tracer
                chi2_by_tracer = []
                n_obs_per_tracer = n_obs // 3
                for i, tracer in enumerate(['ch4', 'd13c', 'dd']):
                    start_idx = i * n_obs_per_tracer
                    end_idx = start_idx + n_obs_per_tracer
                    innov_tracer = innovation[start_idx:end_idx]
                    Sy_tracer = Sy[start_idx:end_idx, start_idx:end_idx]
                    cost_tracer = innov_tracer.T @ linalg.inv(Sy_tracer) @ innov_tracer
                    chi2_tracer = cost_tracer / n_obs_per_tracer
                    chi2_by_tracer.append({'tracer': tracer, 'chi2': chi2_tracer})
                
                chi2_by_tracer = pd.DataFrame(chi2_by_tracer)
            else:
                chi2_obs_total = None
                chi2_prior_total = None
                chi2_by_tracer = None
            
            result = {
                'run_id': run_id,
                'name': param_overrides['name'],
                'params': param_overrides,
                'emissions': emis_post,
                'iso_d13c': iso_d13c_post,
                'iso_dd': iso_dd_post,
                # FIX: store prior isotope arrays so downstream code can read them
                # directly from the pkl without needing to retile config.prior_isotopes
                'prior_iso_d13c': prior_iso_d13c,
                'prior_iso_dd': prior_iso_dd,
                'sinks': sinks_post,
                'output': diagnostics['final_output'],
                'output_prior': output_prior,
                'prior_emissions': self.prior_emissions,
                'lifetime': lifetime_post,
                'costs': diagnostics['costs'],
                'cost_obs': cost_obs,
                'cost_prior': cost_prior,
                'cost_smooth': cost_smooth,
                'cost_total': cost_total,
                'cost_reduction': cost_reduction,
                'rmse_total': rmse_total,
                'rmse_1994_2022': rmse_1994_2022,
                'config': config
            }
            
            # Add chi2 diagnostics if available
            if chi2_obs_total is not None:
                result['chi2_obs_total'] = chi2_obs_total
                result['chi2_prior_total'] = chi2_prior_total
                result['chi2_by_tracer'] = chi2_by_tracer
            
            causality = self.compute_enso_causality(result)
            result['enso_causality'] = causality
            
            # MEI sensitivity analysis
            if self.enhanced_enso_analyzer is not None:
                mei_sensitivity = self.compute_mei_sensitivities(result)
                result['mei_sensitivity'] = mei_sensitivity
                # Save hemisphere-specific CSV files
                if mei_sensitivity:
                    for region in ['nh', 'sh', 'global']:
                        region_upper = region.upper()
                        
                        # Emissions CSV
                        if 'emissions' in mei_sensitivity and region in mei_sensitivity['emissions']:
                            emis_data = []
                            for source, sens in mei_sensitivity['emissions'][region].items():
                                emis_data.append({
                                    'region': region_upper,
                                    'source': source,
                                    'slope_Tg_per_MEI': sens.slope,
                                    'std_error': sens.std_error,
                                    'r_squared': sens.r_squared,
                                    'p_value': sens.p_value,
                                    'mean_emission_Tg': sens.mean_emission,
                                    'el_nino_sensitivity': sens.el_nino_sensitivity,
                                    'la_nina_sensitivity': sens.la_nina_sensitivity
                                })
                            
                            if len(emis_data) > 0:
                                df_emis = pd.DataFrame(emis_data)
                                emis_csv = os.path.join(self.output_dir, 
                                                        f"mei_emission_sensitivity_{region}_run{run_id}.csv")
                                df_emis.to_csv(emis_csv, index=False)
                        
                        # δ13C CSV
                        if 'isotopes_d13c' in mei_sensitivity and region in mei_sensitivity['isotopes_d13c']:
                            d13c_data = []
                            for source, sens in mei_sensitivity['isotopes_d13c'][region].items():
                                d13c_data.append({
                                    'region': region_upper,
                                    'source': source,
                                    'slope_permil_per_MEI': sens.slope,
                                    'std_error': sens.std_error,
                                    'r_squared': sens.r_squared,
                                    'p_value': sens.p_value,
                                    'mean_d13c_permil': sens.mean_isotope,
                                    'el_nino_shift': sens.el_nino_shift,
                                    'la_nina_shift': sens.la_nina_shift
                                })
                            
                            if len(d13c_data) > 0:
                                df_d13c = pd.DataFrame(d13c_data)
                                d13c_csv = os.path.join(self.output_dir, 
                                                        f"mei_d13c_sensitivity_{region}_run{run_id}.csv")
                                df_d13c.to_csv(d13c_csv, index=False)
                        
                        # δD CSV
                        if 'isotopes_dd' in mei_sensitivity and region in mei_sensitivity['isotopes_dd']:
                            dd_data = []
                            for source, sens in mei_sensitivity['isotopes_dd'][region].items():
                                dd_data.append({
                                    'region': region_upper,
                                    'source': source,
                                    'slope_permil_per_MEI': sens.slope,
                                    'std_error': sens.std_error,
                                    'r_squared': sens.r_squared,
                                    'p_value': sens.p_value,
                                    'mean_dd_permil': sens.mean_isotope,
                                    'el_nino_shift': sens.el_nino_shift,
                                    'la_nina_shift': sens.la_nina_shift
                                })
                            
                            if len(dd_data) > 0:
                                df_dd = pd.DataFrame(dd_data)
                                dd_csv = os.path.join(self.output_dir, 
                                                    f"mei_dd_sensitivity_{region}_run{run_id}.csv")
                                df_dd.to_csv(dd_csv, index=False)
                    
                    print(f"  ✓ Saved MEI sensitivity CSVs for NH, SH, and Global")
            else:
                result['mei_sensitivity'] = None
            
            self.results.append(result)
            
            print(f"\n✓ Run {run_id} complete:")
            print(f"  Cost: {cost_total:.2f} (obs: {cost_obs:.2f}, prior: {cost_prior:.2f}, smooth: {cost_smooth:.2f})")
            print(f"  RMSE (1994-2022): {rmse_1994_2022:.4f}")
            
            # Print chi2 if available
            if chi2_obs_total is not None:
                print(f"  CHI-SQUARE: χ²_obs={chi2_obs_total['chi2_obs']:.3f} ({chi2_obs_total['interpretation']}), "
                      f"χ²_prior={chi2_prior_total['chi2_prior']:.3f} ({chi2_prior_total['interpretation']})")
            
            if 'wetlands_emissions' in causality:
                print(f"  ENSO-Wetlands: r={causality['wetlands_emissions']['spearman_r']:.3f} (p={causality['wetlands_emissions']['spearman_p']:.3f})")
                print(f"  ENSO-Pyrogenic: r={causality['pyrogenic_emissions']['spearman_r']:.3f} (p={causality['pyrogenic_emissions']['spearman_p']:.3f})")
            
            if result['mei_sensitivity'] and 'emissions' in result['mei_sensitivity']:
                        mei_sens = result['mei_sensitivity']
                        print(f"  MEI SENSITIVITY (Global):")
                        for source in ['wetlands', 'pyrogenic']:
                            if 'global' in mei_sens['emissions'] and source in mei_sens['emissions']['global']:
                                sens = mei_sens['emissions']['global'][source]
                                sig = "***" if sens.p_value < 0.001 else "**" if sens.p_value < 0.01 else "*" if sens.p_value < 0.05 else "n.s."
                                print(f"    {source.capitalize():12s}: {sens.slope:+.3f} Tg/yr per MEI (R²={sens.r_squared:.3f}, {sig})")
            
            return result
            
        except Exception as e:
            print(f"\n✗ Run {run_id} failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_batch(self, parameter_sets):
        """Run multiple inversions."""
        print(f"\n{'='*70}")
        print(f"STARTING BATCH RUN: {len(parameter_sets)} inversions")
        print(f"{'='*70}")
        
        for i, param_set in enumerate(parameter_sets, 1):
            result = self.run_single_inversion(param_set, run_id=i)
            if result is not None:
                self.save_intermediate(result)
        
        self.save_all_results()
        
        print(f"\n{'='*70}")
        print(f"BATCH COMPLETE: {len(self.results)}/{len(parameter_sets)} successful")
        print(f"{'='*70}")
    
    def save_intermediate(self, result):
        safe_name = result['name'].replace('=', '').replace('_', '').replace('.', 'p')
        filename = os.path.join(self.output_dir, f"{safe_name}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Saved: {os.path.basename(filename)}")
    
    def save_all_results(self):
        filename = os.path.join(self.output_dir, 'all_results.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"\n✓ Saved all results to: {filename}")
    
    def load_results(self, filename=None):
        if filename is None:
            filename = os.path.join(self.output_dir, 'all_results.pkl')
        with open(filename, 'rb') as f:
            self.results = pickle.load(f)
        print(f"✓ Loaded {len(self.results)} results from {filename}")

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run multiple inversions with ENSO analysis')
    parser.add_argument('--mode', type=str, default='grid_small',
                       choices=['grid_small', 'custom'],
                       help='Parameter exploration mode')
    parser.add_argument('--output_dir', type=str, default='/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run/multi_inversion_results_v4',
                       help='Output directory')
    parser.add_argument('--load_only', action='store_true',
                       help='Only load and plot existing results')
    
    args = parser.parse_args()
    
    runner = InversionRunner(output_dir=args.output_dir)
    
    if not args.load_only:
        if args.mode == 'grid_small':
            param_sets = ParameterSets.full_grid_small()
        else:
            print("Custom mode not implemented yet")
            return
        
        print(f"\n🚀 Running {len(param_sets)} inversions in '{args.mode}' mode")
        runner.run_batch(param_sets)
    else:
        runner.load_results()

if __name__ == "__main__":
    main() 