"""
STANDALONE PLOTTING SCRIPT FOR ENSO INVERSION RESULTS

This script loads pre-computed inversion results from pickle files and 
generates all composite plots without re-running inversions.

Usage:
    python standalone_plotter.py --results_dir multi_inversion_results
    python standalone_plotter.py --results_file path/to/all_results.pkl

Author: Bibhasvata Dasgupta
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
import pickle
import argparse
from scipy import stats
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Import MEI sensitivity plotter
try:
    from mei_sensitivity_plotter import MEISensitivityPlotter
    HAS_MEI_PLOTTER = True
except ImportError:
    HAS_MEI_PLOTTER = False
    print("⚠ MEI sensitivity plotter not found - will skip MEI plots")

# ============================================================================
# SCHEMATIC HELPER CONSTANTS & FUNCTIONS (used by plot_f1_composite)
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

def create_rounded_box(ax, xy, width, height, text, color='#e3f2fd', fontsize=10):
    """Draw a rounded rectangle with centred text on ax."""
    x, y = xy
    fancy = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle='round,pad=0.15',
        facecolor=color, edgecolor='black', linewidth=1.5,
        transform=ax.transData, zorder=2
    )
    ax.add_patch(fancy)
    ax.text(x + width / 2, y + height / 2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', zorder=3)

def plot_twobox_schematic(ax):
    """Panel (a): Two-box atmospheric model schematic."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.7, 'Two-Box Atmospheric Model', ha='center', va='top',
            fontsize=FONTSIZE['title'], fontweight='bold')

    create_rounded_box(ax, (1.2, 5.8), 7.6, 2.5,
                    'NH\n\nCH₄, δ¹³C, δD',
                    color=COLORS['nh'], fontsize=FONTSIZE['title'])
    create_rounded_box(ax, (1.2, 1.5), 7.6, 2.5,
                    'SH\n\nCH₄, δ¹³C, δD',
                    color=COLORS['sh'], fontsize=FONTSIZE['title'])

    ax.annotate('', xy=(5.35, 5.6), xytext=(5.35, 4.3),
                arrowprops=dict(arrowstyle='<->', lw=1, color='k', mutation_scale=20))
    ax.text(5, 5, 'Interhemispheric         Exchange (τ)',
            fontsize=FONTSIZE['tick_label'], ha='center', va='center',
            fontweight='bold', color='#3498db')

    ax.text(0.8, 8,   'Sources:',     ha='right', va='top', fontsize=FONTSIZE['tick_label'], fontweight='bold')
    ax.text(0.8, 7,   'Wetlands→',    ha='right', va='top', fontsize=FONTSIZE['tick_label'], color=COLORS['wetlands'])
    ax.text(0.8, 6,   'Agriculture→', ha='right', va='top', fontsize=FONTSIZE['tick_label'], color=COLORS['agriculture'])
    ax.text(0.8, 5,   'Pyrogenic→',   ha='right', va='top', fontsize=FONTSIZE['tick_label'], color=COLORS['pyrogenic'])
    ax.text(0.8, 4,   'Fossil→',      ha='right', va='top', fontsize=FONTSIZE['tick_label'], color=COLORS['fossil'])
    ax.text(0.8, 3,   'Waste→',       ha='right', va='top', fontsize=FONTSIZE['tick_label'], color=COLORS['waste'])

    ax.text(9.0, 7, 'Sinks:',  ha='left', va='top', fontsize=FONTSIZE['tick_label'], fontweight='bold')
    ax.text(9.0, 6, '→Trop',   ha='left', va='top', fontsize=FONTSIZE['tick_label'], color=COLORS['trop_sink'])
    ax.text(9.0, 5, '→Strat',  ha='left', va='top', fontsize=FONTSIZE['tick_label'], color=COLORS['strat_sink'])
    ax.text(9.0, 4, '→Soil',   ha='left', va='top', fontsize=FONTSIZE['tick_label'], color=COLORS['soil_sink'])

    ax.text(5, 0.7,
            r'Lifetime$_{\mathrm{net}}$ = 1/(KIE$_{\mathrm{trop}}$ + KIE$_{\mathrm{strat}}$ + KIE$_{\mathrm{soil}}$)',
            ha='center', va='bottom', fontsize=FONTSIZE['title'],
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.3))

def plot_inversion_logic(ax):
    """Panel (b): Bayesian inversion schematic."""
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.axis('off')

    ax.text(5.5, 10.5, 'Bayesian Inversion', ha='center', va='top',
            fontsize=FONTSIZE['title'], fontweight='bold')

    create_rounded_box(ax, (0.5, 7.0), 2.5, 2.4,
                    'Prior\n(x$_a$)\n\nE, δ, k',
                    color='#e3f2fd', fontsize=FONTSIZE['tick_label'])
    create_rounded_box(ax, (4.3, 7.5), 2.5, 1.5,
                    'Forward\nModel',
                    color='#fff3e0', fontsize=FONTSIZE['tick_label'])
    create_rounded_box(ax, (8.1, 7.0), 2.5, 2.4,
                    'Obs\n(y)\n\nCH₄\nδ¹³C, δD',
                    color='#ffebee', fontsize=FONTSIZE['tick_label'])

    ax.annotate('', xy=(4.2, 8.2), xytext=(3.1, 8.2),
                arrowprops=dict(arrowstyle='->', lw=2.5))
    ax.annotate('', xy=(8.0, 8.2), xytext=(6.9, 8.2),
                arrowprops=dict(arrowstyle='->', lw=2.5))

    cost_text = (
        'Cost Function:\n'
        'J = ||y − F(x)||² / σ²$_{obs}$\n'
        '+ ||x − x$_a$||² / σ²$_a$'
        '+ λ ||∇²x||²'
    )
    create_rounded_box(ax, (2, 4.0), 7, 2.5, cost_text, color='#f3e5f5', fontsize=FONTSIZE['tick_label']+1)

    ax.annotate('', xy=(5.5, 4.0), xytext=(5.5, 3.0),
                arrowprops=dict(arrowstyle='->', lw=3))
    create_rounded_box(ax, (3, 1.2), 5, 1.5, 'Posterior (x̂)',
                    color='#c8e6c9', fontsize=FONTSIZE['tick_label'])

    ax.text(1.7, 6.5, 'σ$_{prior}$', ha='right', va='top',
            fontsize=11, fontweight='bold', color='#3498db')
    for text, y in [('σ$_a$(δ¹³C): 1–2‰', 5.7),
                    ('σ$_a$(δD): 5–10‰', 5.1),
                    ('σ$_{sink}$: 10%\n(Trop, Strat, Soil)', 4.2),
                    ('σ$_{lifetime}$: 10%', 3.5),
                    ('σ$_{emis}$: 30% of\nSource Weighted', 2.8)]:
        ax.text(1.7, y, text, ha='right', va='center',
                fontsize=FONTSIZE['tick_label'], color='#3498db')

    ax.text(9.3, 6.5, 'σ$_{obs}$', ha='left', va='top',
            fontsize=FONTSIZE['title'], fontweight='bold', color='#e74c3c')
    for text, y in [('σ$_{obs}$(δ¹³C): 0.1‰', 5.7),
                    ('σ$_{obs}$(δD): 1–2‰', 5.3),
                    ('σ$_{obs}$(CH₄): 1–2 ppb', 4.9)]:
        ax.text(9.3, y, text, ha='left', va='center',
                fontsize=FONTSIZE['tick_label'], color='#e74c3c')

    ax.text(9.3, 4.2, 'Penalty$_{smooth}$', ha='left', va='top',
            fontsize=FONTSIZE['title'], fontweight='bold', color='#6a1b9a')
    for text, y in [('λ$_{emis}$ = 10', 3.5),
                    ('λ$_{iso,δ^{13}C}$ = 1–2', 3.0),
                    ('λ$_{iso,δD}$ = 0.1–0.2', 2.5),
                    ('λ$_{sinks}$ = 1–2', 2.0)]:
        ax.text(9.3, y, text, ha='left', va='center',
                fontsize=FONTSIZE['tick_label'], color='#6a1b9a')

# ============================================================================
# ENSO CAUSALITY ANALYZER (needed for plotter)
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

# ============================================================================
# STANDALONE COMPOSITE PLOTTER
# ============================================================================

class StandalonePlotter:
    """Standalone plotter that loads results and generates all plots."""

    def __init__(self, results_file, output_dir='plots'):
        """
        Initialize plotter with results file.
        
        Parameters:
        -----------
        results_file : str
            Path to all_results.pkl file
        output_dir : str
            Directory to save plots (default: 'plots')
        """
        self.results_file = results_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"LOADING RESULTS FROM: {results_file}")
        print(f"{'='*70}")
        
        # Load results
        with open(results_file, 'rb') as f:
            self.results = pickle.load(f)
        
        print(f"✓ Loaded {len(self.results)} inversion results")
        
        # Extract metadata from first result
        self.years = np.arange(
            self.results[0]['config'].year1,
            self.results[0]['config'].year2 + 1
        )
        self.observations = self.load_observations_from_result()
        self.prior_isotopes = self.results[0]['config'].prior_isotopes
        
        # Load MEI and setup ENSO analyzer
        self.mei_df = self.load_mei_data()
        self.enso_analyzer = ENSOCausalityAnalyzer(self.mei_df)
        
        # Sort results by cost reduction
        self.results_sorted = sorted(
            self.results, 
            key=lambda x: x['cost_reduction'], 
            reverse=True
        )
        self.top5_ids = [r['run_id'] for r in self.results_sorted[:5]]
        
        # Setup colors
        self.setup_colors()
        
        print(f"✓ Initialization complete")
        print(f"  Years: {self.years[0]} - {self.years[-1]}")
        print(f"  Top 5 runs: {self.top5_ids}")
    
    def load_observations_from_result(self):
        """Extract observations from first result."""
        # The observations should be stored or we reconstruct them
        # For now, we'll use a dummy - you should replace this with actual obs loading
        print("⚠ Loading observations from result metadata...")
        
        # Try to extract from result if available
        if 'observations' in self.results[0]:
            return self.results[0]['observations']
        
        # Otherwise reconstruct (you may need to adjust this)
        try:
            from twobox_simultaneous_isotope_emission_cost_func_time_varying_inversion_v4 import load_observations
            return load_observations(self.results[0]['config'])
        except:
            print("⚠ Could not load observations - plots may be incomplete")
            return None
    
    def setup_colors(self):
        """Setup color scheme for results."""
        n_results = len(self.results)
        n_others = n_results - 5 if n_results > 5 else 0

        # Top 5: distinct red shades
        top5_palette = ['darkred', 'maroon', 'firebrick', 'brown', 'indianred']
        self.top5_colors = {
            self.top5_ids[i]: top5_palette[i] 
            for i in range(min(5, len(self.top5_ids)))
        }

        # Others: YlOrRd_r gradient
        if n_others > 0:
            other_colors = plt.cm.YlOrRd_r(np.linspace(0, 1, n_others))
        else:
            other_colors = []
        
        self.result_colors = {}
        other_idx = 0
        for result in self.results_sorted:
            rid = result['run_id']
            if rid in self.top5_ids:
                self.result_colors[rid] = self.top5_colors[rid]
            else:
                self.result_colors[rid] = other_colors[other_idx]
                other_idx += 1

        # Cost reduction range for colorbar
        if len(self.results_sorted) > 5:
            self.cost_reductions = [r['cost_reduction'] for r in self.results_sorted[5:]]
            self.cost_min = min(self.cost_reductions)
            self.cost_max = max(self.cost_reductions)
        else:
            self.cost_reductions = []
            self.cost_min = 0
            self.cost_max = 1
    
    def load_mei_data(self):
        """Load MEI data for ENSO analysis."""
        try:
            mei_file = '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run/input_data/meiv2.data.txt'
            mei_cols = ['Year', 'DJ', 'JF', 'FM', 'MA', 'AM', 'MJ', 'JJ', 'JA', 'AS', 'SO', 'ON', 'ND']
            mei_data = pd.read_csv(mei_file, delim_whitespace=True, skiprows=1, 
                                   skipfooter=4, header=None, engine='python')
            mei_data.columns = mei_cols
            mei_long = mei_data.melt(id_vars='Year', var_name='Month', value_name='Anomaly')
            month_map = {name: str(i).zfill(2) for i, name in enumerate(mei_cols[1:], 1)}
            mei_long['Month'] = mei_long['Month'].map(month_map)
            mei_long['Date'] = pd.to_datetime(
                mei_long['Year'].astype(str) + '-' + mei_long['Month'], 
                format='%Y-%m'
            )
            mei_long = mei_long.dropna(subset=['Anomaly'])
            mei_long['Year'] = mei_long['Date'].dt.year
            mei_yearly = mei_long.groupby('Year')['Anomaly'].mean().reset_index()
            mei_yearly.rename(columns={'Anomaly': 'MEI'}, inplace=True)
            print(f"✓ Loaded MEI data: {len(mei_yearly)} years")
            return mei_yearly
        except Exception as e:
            print(f"⚠ MEI load failed: {e}")
            return pd.DataFrame({'Year': self.years, 'MEI': np.zeros(len(self.years))})
    
    def shade_enso(self, ax):
        """Add ENSO shading to axis."""
        el_nino_years = self.mei_df[self.mei_df['MEI'] > 0.5]['Year'].values
        la_nina_years = self.mei_df[self.mei_df['MEI'] < -0.5]['Year'].values
        
        for y in el_nino_years:
            if y >= self.years[0] and y <= self.years[-1]:
                ax.axvspan(y - 0.5, y + 0.5, color='red', alpha=0.1, zorder=0)
        for y in la_nina_years:
            if y >= self.years[0] and y <= self.years[-1]:
                ax.axvspan(y - 0.5, y + 0.5, color='blue', alpha=0.1, zorder=0)
    
    def debug_print_enso_causality_keys(self):
        """Print all available keys in enso_causality to find correct format."""
        if len(self.results) == 0 or 'enso_causality' not in self.results[0]:
            print("No enso_causality data!")
            return
        
        ec = self.results[0]['enso_causality']
        
        print("\n" + "="*80)
        print("ENSO CAUSALITY KEYS AVAILABLE:")
        print("="*80)
        
        for key in sorted(ec.keys()):
            print(f"\n'{key}':")
            if isinstance(ec[key], dict):
                sub_keys = list(ec[key].keys())
                print(f"  Sub-keys: {sub_keys}")
                if 'mean_diff' in ec[key]:
                    print(f"  → mean_diff = {ec[key]['mean_diff']:.4f}")
        
        print("="*80 + "\n")
        
    def plot_all(self):
        """Generate all plots."""
        print(f"\n{'='*70}")
        print("GENERATING ALL PLOTS")
        print(f"{'='*70}\n")

        plots = [
            ("F1: Model-Data Fit", "composite_f1_model_data_fit.png", self.plot_f1_composite),
            ("F2: Emissions & Sinks", "composite_f2_emissions_isotopes.png", self.plot_f2_composite),
            ("F5: Isotope Timeseries", "composite_f5_isotopes.png", self.plot_f5_isotopes),
            ("Parameter Sensitivity", "parameter_sensitivity.png", self.plot_parameter_sensitivity),
            ("ENSO Causality vs Params", "enso_causality_vs_parameters.png", self.plot_enso_causality_vs_parameters),
            ("ENSO Effect Sizes", "enso_effect_sizes.png", self.plot_enso_effect_size_comparison),
            ("Isotope Shifts (Diff)", "enso_isotope_diff.png", self.plot_isotope_enso_shifts_diff),
            ("Isotope ENSO Phases", "enso_isotope_phases.png", self.plot_isotope_enso_shifts),
            ("Emission ENSO Phases", "enso_emission_phases.png", self.plot_emission_enso_shifts),
            ("Summary Table", "summary_table_complete.csv", self.create_summary_table),

            *([
                ("Chi2 vs Parameters", "chi2_vs_parameters.png", self.plot_chi2_vs_parameters),
                ("Chi2 by Tracer", "chi2_by_tracer.png", self.plot_chi2_by_tracer),
                ("Chi2 Summary", "chi2_summary.png", self.plot_chi2_summary),
            ] if any("chi2_obs_total" in r for r in self.results) else []),

            ("Robustness Check", "robustness_check.png", self.plot_robustness_check),

            *([
                ("MEI Emission Sensitivity", "mei_emission_sensitivity.png", self.plot_mei_emission_sensitivity),
                ("MEI d13C Sensitivity", "mei_d13c_sensitivity.png", self.plot_mei_d13c_sensitivity),
                ("MEI dD Sensitivity", "mei_dd_sensitivity.png", self.plot_mei_dd_sensitivity),
                ("MEI Sensitivity Summary", "mei_sensitivity_summary.png", self.plot_mei_sensitivity_summary),
                ("MEI Sensitivity Summary NH+SH", "mei_sensitivity_summary_nhsh.png", self.plot_mei_sensitivity_summary_nhsh),
                ("MEI Sensitivity CSVs", "mei_sensitivity_csvs", self.export_mei_sensitivity_csvs),
            ] if HAS_MEI_PLOTTER else []),
        ]

        for i, (desc, filename, func) in enumerate(plots, 1):
            print(f"[{i}/{len(plots)}] Creating {desc}...")
            try:
                func(os.path.join(self.output_dir, filename))
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*70}")
        print("✅ ALL PLOTS COMPLETE!")
        print(f"{'='*70}")
        print(f"Output directory: {self.output_dir}")

    # ========================================================================
    # INDIVIDUAL PLOT FUNCTIONS (copied from your script)
    # ========================================================================
    
    def plot_f1_composite_old(self, output_path):
        """F1 with correct plotting order."""
        if self.observations is None:
            print("    ⚠ Skipping F1 - no observations available")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(10, 12.5), sharex=True)
        
        tracers = ['ch4', 'd13c', 'dd']
        tracer_labels = ['CH$_4$ (ppb)', '$\\delta^{13}$C-CH$_4$ (‰)', '$\\delta$D-CH$_4$ (‰)']
        markers = ['+', '.']
        
        prior_output = self.results[0]['output_prior']
        
        for i, (tracer, label) in enumerate(zip(tracers, tracer_labels)):
            ax = axes[i]
            self.shade_enso(ax)
            
            # Layer 1: Other runs
            for result in self.results_sorted:
                #if result['run_id'] not in self.top5_ids:
                color = self.result_colors[result['run_id']]
                for j, hem in enumerate(['sh', 'nh']):
                    ax.plot(self.years, result['output'][tracer][:, j],
                            color=color, alpha=0.3, linewidth=1.0,
                            marker=markers[j], markersize=1, zorder=10)
            
            # Layer 2: Prior
            for j, hem in enumerate(['sh', 'nh']):
                ax.plot(self.years, prior_output[tracer][:, j],
                       color='grey', linestyle=':', linewidth=2.5, alpha=0.7,
                       marker=markers[j], markersize=3, zorder=50,
                       label='Prior' if i == 0 and j == 0 else '')
            
            # Layer 3: Observations
            for j, hem in enumerate(['sh', 'nh']):
                ax.plot(self.years, self.observations[tracer][hem],
                       color='black', marker=markers[j], linestyle='--',
                       linewidth=2.5, markersize=6, alpha=0.9, zorder=100,
                       label=f'Obs {hem.upper()}' if i == 0 else '')
            
            # Layer 4: Top 5
            for rank, result in enumerate(self.results_sorted[:5]):
                color = self.result_colors[result['run_id']]
                for j, hem in enumerate(['sh', 'nh']):
                    label_text = f"#{rank+1}: {result['name']} (cost={result['cost_total']:.0f})" if i == 0 and j == 0 else ''
                    #ax.plot(self.years, result['output'][tracer][:, j],
                           #color=color, alpha=0.9, linewidth=2.5,
                           #marker=markers[j], markersize=3,
                           #label=label_text, zorder=200)
            
            ax.set_ylabel(label, fontsize=12)
            ax.yaxis.set_ticks_position('both')
            ax.grid(alpha=0.3)
            ax.axvspan(self.years[0], 1994, facecolor='grey', alpha=0.3, zorder=1)
            ax.axvspan(2022, self.years[-1], facecolor='grey', alpha=0.3, zorder=1)
            
            if i < 2:
                ax.set_xticks([])
                ax.spines['bottom'].set_visible(False)
            if i > 0:
                ax.spines['top'].set_visible(False)
        
        axes[0].legend(loc='upper left', frameon=True, fontsize=8, ncol=1)
        
        #if self.cost_reductions:
            #norm = Normalize(vmin=self.cost_min, vmax=self.cost_max)
            #sm = ScalarMappable(cmap=plt.cm.YlOrRd_r, norm=norm)
            #sm.set_array([])
            #cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
            #cb = plt.colorbar(sm, cax=cax, orientation='vertical')
            #cb.set_label('Cost Reduction\n(Other Runs)', fontsize=9)
        
        axes[2].set_xlabel('Year', fontsize=12)
        axes[2].set_xticks(self.years[::4])
        plt.xlim(1980, 2024)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.subplots_adjust(hspace=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")

    def plot_f1_composite(self, output_path):
            """F1: Two-panel composite — (a) two-box schematic, (b) Bayesian inversion, (c) model-data fit."""
            if self.observations is None:
                print("    ⚠ Skipping F1 - no observations available")
                return

            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(12, 16))

            # Top section: schematics (a) and (b)
            gs_top = gridspec.GridSpec(1, 2, figure=fig,
                                    left=0.05, right=0.95,
                                    top=0.97, bottom=0.70,
                                    wspace=0.08)

            ax_a = fig.add_subplot(gs_top[0, 0])
            ax_b = fig.add_subplot(gs_top[0, 1])
            plot_twobox_schematic(ax_a)
            plot_inversion_logic(ax_b)

            # Panel labels (a) and (b) — top-left of each schematic axes
            for ax, lbl in [(ax_a, 'a'), (ax_b, 'b')]:
                ax.text(-0.04, 0.95, f'({lbl})', transform=ax.transAxes,
                        fontsize=13, fontweight='bold', va='bottom', ha='left')

            # Bottom section: model-data fit (c) — 3 tracer rows sharing x-axis
            gs_bot = gridspec.GridSpec(3, 1, figure=fig,
                                    left=0.08, right=0.85,
                                    top=0.7, bottom=0.04,
                                    hspace=0)

            tracers = ['ch4', 'd13c', 'dd']
            tracer_labels = ['CH$_4$ (ppb)', '$\\delta^{13}$C-CH$_4$ (‰)', '$\\delta$D-CH$_4$ (‰)']
            markers = ['+', '.']

            prior_output = self.results[0]['output_prior']
            axes_bot = []

            cost_vals = [r['cost_reduction'] for r in self.results_sorted]
            norm = Normalize(vmin=min(cost_vals), vmax=max(cost_vals)) if cost_vals else None

            for i, (tracer, label) in enumerate(zip(tracers, tracer_labels)):
                sharex = axes_bot[0] if i > 0 else None
                ax = fig.add_subplot(gs_bot[i, 0], sharex=sharex)
                axes_bot.append(ax)

                self.shade_enso(ax)

                # Layer 1: all runs
                for result in self.results_sorted:
                    color = plt.cm.YlOrRd_r(norm(result['cost_reduction'])) if norm is not None else 'grey'
                    for j in range(2):
                        ax.plot(self.years, result['output'][tracer][:, j],
                                color=color, alpha=0.3, linewidth=1.0,
                                marker=markers[j], markersize=1, zorder=10)

                # Layer 2: prior
                for j in range(2):
                    ax.plot(self.years, prior_output[tracer][:, j],
                            color='grey', linestyle=':', linewidth=2.5, alpha=0.7,
                            marker=markers[j], markersize=3, zorder=50,
                            label='Prior' if i == 0 and j == 0 else '')

                # Layer 3: observations
                for j, hem in enumerate(['sh', 'nh']):
                    ax.plot(self.years, self.observations[tracer][hem],
                            color='black', marker=markers[j], linestyle='--',
                            linewidth=2.5, markersize=6, alpha=0.9, zorder=100,
                            label=f'Obs {hem.upper()}' if i == 0 else '')

                # Layer 4: highlight top 5 runs
                for rank, result in enumerate(self.results_sorted[:5]):
                    color = plt.cm.YlOrRd_r(norm(result['cost_reduction'])) if norm is not None else self.result_colors[result['run_id']]
                    for j in range(2):
                        ax.plot(self.years, result['output'][tracer][:, j],
                                color=color, alpha=0.9, linewidth=2.5,
                                marker=markers[j], markersize=3, zorder=200)

                ax.set_ylabel(label, fontsize=FONTSIZE['title'])
                ax.yaxis.set_ticks_position('both')
                ax.grid(alpha=0.3)
                ax.axvspan(self.years[0], 1994, facecolor='grey', alpha=0.3, zorder=1)
                ax.axvspan(2022, self.years[-1], facecolor='grey', alpha=0.3, zorder=1)

                if i < 2:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    ax.spines['bottom'].set_visible(False)
                if i > 0:
                    ax.spines['top'].set_visible(False)

            # Panel label (c) — top-left of the first tracer row
            axes_bot[0].text(-0.07, 1.0, '(c)', transform=axes_bot[0].transAxes,
                            fontsize=13, fontweight='bold', va='bottom', ha='left')

            axes_bot[0].legend(loc='upper left', frameon=True, fontsize=FONTSIZE['title'], ncol=1)
            axes_bot[2].set_xlabel('Year', fontsize=FONTSIZE['title'])
            axes_bot[2].set_xticks(self.years[::4])
            axes_bot[2].set_xlim(1980, 2024)

            if norm is not None:
                sm = ScalarMappable(cmap=plt.cm.YlOrRd_r, norm=norm)
                sm.set_array([])
                # Colorbar aligned with panel (c) only: bottom=0.04, top=0.7
                cbar_ax = fig.add_axes([0.88, 0.04, 0.015, 0.66])
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
                # Format labels as scientific notation
                cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                cbar.set_label('Cost Reduction', fontsize=FONTSIZE['title'])

            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"    ✓ Saved to: {output_path}")

    def plot_f2_composite_old(self, output_path):
        """F2: Emissions with correct order and fixed colorbar."""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        sources = self.results[0]['config'].sources + ['total_sink']
        prior_emissions = self.results[0]['prior_emissions']
        
        for i, source in enumerate(sources):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            self.shade_enso(ax)
            
            if source == 'total_sink':
                # Layer 1: Other runs
                for result in self.results_sorted:
                    #if result['run_id'] not in self.top5_ids:
                    color = self.result_colors[result['run_id']]
                    config = result['config']
                    sinks = result['sinks']
                    post_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + sinks['oh'] / 100.0)
                    post_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + sinks['strat'] / 100.0)
                    post_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + sinks['soil'] / 100.0)
                    post_total_sink_rate = post_sink_oh + post_sink_strat + post_sink_soil
                    post_lifetime = 1.0 / post_total_sink_rate
                    post_lifetime_global = 2.0 / (1.0/post_lifetime[:, 0] + 1.0/post_lifetime[:, 1])
                    ax.plot(self.years, post_lifetime_global, color=color, linewidth=1.2, 
                            marker='.', alpha=0.4, zorder=10)
                
                # Layer 2: Prior
                config = self.results[0]['config']
                prior_sinks = {sink: np.zeros((len(self.years), 2)) for sink in ['oh', 'strat', 'soil']}
                prior_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + prior_sinks['oh'] / 100.0)
                prior_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + prior_sinks['strat'] / 100.0)
                prior_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + prior_sinks['soil'] / 100.0)
                prior_total_sink_rate = prior_sink_oh + prior_sink_strat + prior_sink_soil
                prior_lifetime = 1.0 / prior_total_sink_rate
                prior_lifetime_global = 2.0 / (1.0/prior_lifetime[:, 0] + 1.0/prior_lifetime[:, 1])
                ax.plot(self.years, prior_lifetime_global, color='grey', linestyle=':', 
                       linewidth=2.5, marker='.', alpha=0.7, label='Prior', zorder=50)
                
                # Layer 3: Top 5
                for result in self.results_sorted[:5]:
                    color = self.result_colors[result['run_id']]
                    config = result['config']
                    sinks = result['sinks']
                    post_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + sinks['oh'] / 100.0)
                    post_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + sinks['strat'] / 100.0)
                    post_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + sinks['soil'] / 100.0)
                    post_total_sink_rate = post_sink_oh + post_sink_strat + post_sink_soil
                    post_lifetime = 1.0 / post_total_sink_rate
                    post_lifetime_global = 2.0 / (1.0/post_lifetime[:, 0] + 1.0/post_lifetime[:, 1])
                    #ax.plot(self.years, post_lifetime_global, color=color, linewidth=2.0, 
                           #marker='.', alpha=0.8, zorder=100)
                
                ax.set_ylabel('Lifetime (years)', fontsize=11)
            else:
                # Layer 1: Other runs, SH solid / NH dashed
                for result in self.results_sorted:
                    color = self.result_colors[result['run_id']]
                    emis = result['emissions'][source]
                    ax.plot(self.years, emis[:, 0], color=color, linewidth=1.0,
                            linestyle='-',  marker='.', markersize=2, alpha=0.4, zorder=10)
                    ax.plot(self.years, emis[:, 1], color=color, linewidth=0.8,
                            linestyle='--', alpha=0.3, zorder=10)

                # Layer 2: Prior, SH solid / NH dashed
                ax.plot(self.years, prior_emissions[source][:, 0], color='grey',
                        linestyle=':', linewidth=2.5, marker='.', markersize=3, alpha=0.7,
                        zorder=50, label='Prior SH' if i == 0 else '')
                ax.plot(self.years, prior_emissions[source][:, 1], color='grey',
                        linestyle='-.', linewidth=2.5, alpha=0.7, zorder=50,
                        label='Prior NH' if i == 0 else '')

                # Layer 3: Top-5 highlighted
                for result in self.results_sorted[:5]:
                    color = self.result_colors[result['run_id']]
                    emis = result['emissions'][source]
                    ax.plot(self.years, emis[:, 0], color=color, linewidth=2.0,
                            linestyle='-',  marker='.', markersize=3, alpha=0.85, zorder=100)
                    ax.plot(self.years, emis[:, 1], color=color, linewidth=1.5,
                            linestyle='--', alpha=0.7, zorder=100)

                ax.set_ylabel('Emissions (Tg/yr)', fontsize=11)
            
            ax.set_title(source.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_xlim(1980, 2024)
            ax.axvspan(self.years[0], 1994, facecolor='grey', alpha=0.3, zorder=1)
            ax.axvspan(2022, self.years[-1], facecolor='grey', alpha=0.3, zorder=1)
            
            if row < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Year', fontsize=11)
            
            if i == 0:
                ax.legend(loc='lower right', frameon=True, fontsize=9)
        
        if self.cost_reductions:
            norm = Normalize(vmin=self.cost_min, vmax=self.cost_max)
            sm = ScalarMappable(cmap=plt.cm.YlOrRd_r, norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.9, 0.3, 0.015, 0.4])
            cb = plt.colorbar(sm, cax=cax, orientation='vertical')
            cb.set_label('Cost Reduction', fontsize=9)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")

    def plot_f2_composite(self, output_path):
        """F2: Emissions with correct order, fixed colorbar, and panel labels (a–f)."""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        sources = self.results[0]['config'].sources + ['total_sink']
        prior_emissions = self.results[0]['prior_emissions']
        panel_labels = list('abcdef')

        cost_vals = [r['cost_reduction'] for r in self.results_sorted]
        if cost_vals:
            vmin, vmax = min(cost_vals), max(cost_vals)
            if vmin == vmax:
                vmax = vmin + 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = None

        for i, source in enumerate(sources):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            self.shade_enso(ax)

            # Panel label
            ax.text(0.01, 0.98, f'({panel_labels[i]})', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top', ha='left', zorder=200)

            if source == 'total_sink':
                for result in self.results_sorted:
                    color = plt.cm.YlOrRd_r(norm(result['cost_reduction'])) if norm is not None else self.result_colors[result['run_id']]
                    config = result['config']
                    sinks = result['sinks']
                    post_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + sinks['oh'] / 100.0)
                    post_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + sinks['strat'] / 100.0)
                    post_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + sinks['soil'] / 100.0)
                    post_total_sink_rate = post_sink_oh + post_sink_strat + post_sink_soil
                    post_lifetime = 1.0 / post_total_sink_rate
                    post_lifetime_global = 2.0 / (1.0/post_lifetime[:, 0] + 1.0/post_lifetime[:, 1])
                    ax.plot(self.years, post_lifetime_global, color=color, linewidth=1.2,
                            marker='.', alpha=0.4, zorder=10)

                config = self.results[0]['config']
                prior_sinks = {sink: np.zeros((len(self.years), 2)) for sink in ['oh', 'strat', 'soil']}
                prior_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + prior_sinks['oh'] / 100.0)
                prior_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + prior_sinks['strat'] / 100.0)
                prior_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + prior_sinks['soil'] / 100.0)
                prior_total_sink_rate = prior_sink_oh + prior_sink_strat + prior_sink_soil
                prior_lifetime = 1.0 / prior_total_sink_rate
                prior_lifetime_global = 2.0 / (1.0/prior_lifetime[:, 0] + 1.0/prior_lifetime[:, 1])
                ax.plot(self.years, prior_lifetime_global, color='grey', linestyle=':',
                        linewidth=2.5, marker='.', alpha=0.7, label='Prior', zorder=50)

                for result in self.results_sorted[:5]:
                    color = plt.cm.YlOrRd_r(norm(result['cost_reduction'])) if norm is not None else self.result_colors[result['run_id']]
                    config = result['config']
                    sinks = result['sinks']
                    post_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + sinks['oh'] / 100.0)
                    post_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + sinks['strat'] / 100.0)
                    post_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + sinks['soil'] / 100.0)
                    post_total_sink_rate = post_sink_oh + post_sink_strat + post_sink_soil
                    post_lifetime = 1.0 / post_total_sink_rate
                    post_lifetime_global = 2.0 / (1.0/post_lifetime[:, 0] + 1.0/post_lifetime[:, 1])
                    #ax.plot(self.years, post_lifetime_global, color=color, linewidth=2.0,
                    #        marker='.', alpha=0.8, zorder=100)

                ax.set_ylabel('Lifetime (years)', fontsize=11)
            else:
                # ── Layer 1: all runs, SH solid / NH dashed ──────────────────
                for result in self.results_sorted:
                    color = (plt.cm.YlOrRd_r(norm(result['cost_reduction']))
                             if norm is not None else self.result_colors[result['run_id']])
                    emis = result['emissions'][source]
                    ax.plot(self.years, emis[:, 0], color=color, linewidth=1.0,
                            linestyle='-',  marker='.', markersize=2, alpha=0.4, zorder=10)
                    ax.plot(self.years, emis[:, 1], color=color, linewidth=0.8,
                            linestyle='--', marker='',  alpha=0.3, zorder=10)

                # ── Layer 2: prior, SH solid / NH dashed ─────────────────────
                prior_sh = prior_emissions[source][:, 0]
                prior_nh = prior_emissions[source][:, 1]
                ax.plot(self.years, prior_sh, color='grey', linestyle=':',
                        linewidth=2.5, marker='.', markersize=3, alpha=0.7, zorder=50,
                        label='Prior SH' if i == 0 else '')
                ax.plot(self.years, prior_nh, color='grey', linestyle='-.',
                        linewidth=2.5, alpha=0.7, zorder=50,
                        label='Prior NH' if i == 0 else '')

                # ── Layer 3: top-5 runs highlighted ──────────────────────────
                for result in self.results_sorted[:5]:
                    color = (plt.cm.YlOrRd_r(norm(result['cost_reduction']))
                             if norm is not None else self.result_colors[result['run_id']])
                    emis = result['emissions'][source]
                    ax.plot(self.years, emis[:, 0], color=color, linewidth=2.0,
                            linestyle='-',  marker='.', markersize=3, alpha=0.85, zorder=100)
                    ax.plot(self.years, emis[:, 1], color=color, linewidth=1.5,
                            linestyle='--', alpha=0.7, zorder=100)

                ax.set_ylabel('Emissions (Tg/yr)', fontsize=11)

            ax.set_title(source.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_xlim(1980, 2024)
            ax.axvspan(self.years[0], 1994, facecolor='grey', alpha=0.3, zorder=1)
            ax.axvspan(2022, self.years[-1], facecolor='grey', alpha=0.3, zorder=1)

            if row < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Year', fontsize=11)

            if i == 5:
                from matplotlib.lines import Line2D
                proxy = [
                    Line2D([0], [0], color='grey',  linestyle=':',  linewidth=2, label='Prior SH'),
                    Line2D([0], [0], color='grey',  linestyle='-.', linewidth=2, label='Prior NH'),
                    Line2D([0], [0], color='black', linestyle='-',  linewidth=1.5, label='Posterior SH'),
                    Line2D([0], [0], color='black', linestyle='--', linewidth=1.2, label='Posterior NH'),
                ]
                ax.legend(handles=proxy, loc='lower right', frameon=True,
                          fontsize=FONTSIZE['title'], ncol=2)

        if norm is not None:
            sm = ScalarMappable(cmap=plt.cm.YlOrRd_r, norm=norm)
            sm.set_array([])
            # Colorbar takes full height of the plot
            cax = fig.add_axes([0.92, 0.08, 0.015, 0.84])
            cb = plt.colorbar(sm, cax=cax, orientation='vertical')
            # Format labels as scientific notation
            cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            cb.set_label('Cost Reduction', fontsize=FONTSIZE['title'])

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")

    def plot_f5_isotopes(self, output_path):
        """F5: Isotopes with correct order and fixed colorbar."""
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        sources = self.results[0]['config'].sources

        cost_vals = [r['cost_reduction'] for r in self.results_sorted]
        if cost_vals:
            vmin, vmax = min(cost_vals), max(cost_vals)
            if vmin == vmax:
                vmax = vmin + 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = None

        for i, source in enumerate(sources):
            ax_d13c = axes[i, 0]
            self.shade_enso(ax_d13c)
            ax_dd = axes[i, 1]
            self.shade_enso(ax_dd)

            # Panel labels: a/b, c/d, e/f, g/h, i/j
            left_label  = chr(ord('a') + i * 2)
            right_label = chr(ord('a') + i * 2 + 1)
            ax_d13c.text(0.01, 0.98, f'({left_label})',  transform=ax_d13c.transAxes,
                         fontsize=12, fontweight='bold', va='top', ha='left', zorder=200)
            ax_dd.text(  0.01, 0.98, f'({right_label})', transform=ax_dd.transAxes,
                         fontsize=12, fontweight='bold', va='top', ha='left', zorder=200)
            
            # Layer 1: Other runs
            for result in self.results_sorted:
                #if result['run_id'] not in self.top5_ids:
                color = plt.cm.YlOrRd_r(norm(result['cost_reduction'])) if norm is not None else self.result_colors[result['run_id']]
                d13c_sh = result['iso_d13c'][source][:, 0]
                d13c_nh = result['iso_d13c'][source][:, 1]
                dd_sh = result['iso_dd'][source][:, 0]
                dd_nh = result['iso_dd'][source][:, 1]

                ax_d13c.plot(self.years, d13c_sh, color=color, linewidth=1.0,
                            alpha=0.4, linestyle='-', marker='.', markersize=1, zorder=10)
                ax_dd.plot(self.years, dd_sh, color=color, linewidth=1.0,
                        alpha=0.4, linestyle='-', marker='.', markersize=1, zorder=10)
                ax_d13c.plot(self.years, d13c_nh, color=color, linewidth=0.8,
                            alpha=0.3, linestyle='--', marker='', zorder=11)
                ax_dd.plot(self.years, dd_nh, color=color, linewidth=0.8,
                        alpha=0.3, linestyle='--', marker='', zorder=11)

            # Layer 2: Prior
            prior_iso = self.results[0]['config'].prior_isotopes
            prior_d13c_sh = prior_iso[source]['d13c'][0]
            prior_d13c_nh = prior_iso[source]['d13c'][1]
            prior_dd_sh = prior_iso[source]['dd'][0]
            prior_dd_nh = prior_iso[source]['dd'][1]

            ax_d13c.axhline(prior_d13c_sh, color='darkgrey', linestyle='-',
                            linewidth=2.5, alpha=0.7, label='Prior SH' if i == 0 else '', zorder=50)
            ax_d13c.axhline(prior_d13c_nh, color='darkgrey', linestyle='--',
                            linewidth=2.5, alpha=0.7, label='Prior NH' if i == 0 else '', zorder=50)
            ax_dd.axhline(prior_dd_sh, color='darkgrey', linestyle='-',
                        linewidth=2.5, alpha=0.7, label='Prior SH' if i == 0 else '', zorder=50)
            ax_dd.axhline(prior_dd_nh, color='darkgrey', linestyle='--',
                        linewidth=2.5, alpha=0.7, label='Prior NH' if i == 0 else '', zorder=50)

            # Layer 3: Top 5
            for result in self.results_sorted[:5]:
                color = self.result_colors[result['run_id']]
                d13c_sh = result['iso_d13c'][source][:, 0]
                d13c_nh = result['iso_d13c'][source][:, 1]
                dd_sh = result['iso_dd'][source][:, 0]
                dd_nh = result['iso_dd'][source][:, 1]

                """ax_d13c.plot(self.years, d13c_sh, color=color, linewidth=2.0,
                            alpha=0.8, linestyle='-', marker='.', markersize=2, zorder=100)
                ax_dd.plot(self.years, dd_sh, color=color, linewidth=2.0,
                        alpha=0.8, linestyle='-', marker='.', markersize=2, zorder=100)
                ax_d13c.plot(self.years, d13c_nh, color=color, linewidth=1.5,
                            alpha=0.6, linestyle='--', marker='', zorder=101)
                ax_dd.plot(self.years, dd_nh, color=color, linewidth=1.5,
                        alpha=0.6, linestyle='--', marker='', zorder=101)"""

            if i == 0:
                ax_d13c.plot([], [], color='black', linewidth=1.5, linestyle='-',
                            label='Posterior SH', alpha=0.7)
                ax_d13c.plot([], [], color='black', linewidth=1.2, linestyle='--',
                            label='Posterior NH', alpha=0.5)

            ax_d13c.set_ylabel('δ¹³C (‰)', fontsize=11)
            ax_dd.set_ylabel('\u03B4D (‰)', fontsize=11)
            ax_d13c.set_title(f'{source.capitalize()}', fontsize=11, fontweight='bold')
            ax_dd.set_title(f'{source.capitalize()}', fontsize=11, fontweight='bold')
            ax_d13c.grid(alpha=0.3)
            ax_dd.grid(alpha=0.3)
            ax_d13c.axvspan(self.years[0], 1994, facecolor='grey', alpha=0.3, zorder=1)
            ax_d13c.axvspan(2022, self.years[-1], facecolor='grey', alpha=0.3, zorder=1)
            ax_dd.axvspan(self.years[0], 1994, facecolor='grey', alpha=0.3, zorder=1)
            ax_dd.axvspan(2022, self.years[-1], facecolor='grey', alpha=0.3, zorder=1)
            ax_d13c.set_xlim(1980, 2024)
            ax_dd.set_xlim(1980, 2024)

            if i < 4:
                ax_d13c.set_xticks([])
                ax_dd.set_xticks([])
            else:
                ax_d13c.set_xlabel('Year', fontsize=11)
                ax_dd.set_xlabel('Year', fontsize=11)
                ax_d13c.set_xticks(self.years[::4])
                ax_dd.set_xticks(self.years[::4])

            if i == 0:
                ax_d13c.legend(loc='center right', frameon=True, fontsize=FONTSIZE['title'], ncol=2)

        if norm is not None:
            sm = ScalarMappable(cmap=plt.cm.YlOrRd_r, norm=norm)
            sm.set_array([])
            # Colorbar takes full height of the plot
            cax = fig.add_axes([0.92, 0.08, 0.015, 0.84])
            cb = plt.colorbar(sm, cax=cax, orientation='vertical')
            # Format labels as scientific notation
            cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            cb.set_label('Cost Reduction', fontsize=FONTSIZE['title'])

        plt.suptitle('Posterior Isotopic Signatures (SH solid, NH dashed)\n'
                    'Red = El Niño | Blue = La Niña',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 0.9, 0.99])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")
    
    def plot_parameter_sensitivity(self, output_path):
        """
        Parameter sensitivity plot showing how cost reduction varies with parameters.
        
        Updated to include separate smoothness parameters for \u03B4\u00B9\u00B3C and \u03B4D.
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Updated parameter list with separate isotope smoothness
        params_to_plot = [
            ('prior_isotope_d13c_error', 'Prior \u03B4\u00B9\u00B3C Error (‰)'),
            ('prior_isotope_dd_error', 'Prior \u03B4D Error (‰)'),
            ('obs_sigma_d13c', 'Obs \u03B4\u00B9\u00B3C Uncertainty (‰)'),
            ('obs_sigma_dd', 'Obs \u03B4D Uncertainty (‰)'),
            ('prior_emission_error', 'Prior Emission Error (fraction)'),
            ('smoothness_lambda_emissions', 'Smoothness λ (Emissions)'),
            ('smoothness_lambda_d13c', 'Smoothness λ (\u03B4\u00B9\u00B3C)'),  # NEW: Separate for \u03B4\u00B9\u00B3C
            ('smoothness_lambda_dd', 'Smoothness λ (\u03B4D)'),      # NEW: Separate for \u03B4D
        ]
        
        for idx, (param_name, param_label) in enumerate(params_to_plot):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            param_vals = []
            cost_reds = []
            chi2_obs_vals = []
            
            for result in self.results:
                # Check if parameter exists in this result
                if param_name in result['params']:
                    param_vals.append(result['params'][param_name])
                    cost_reds.append(result['cost_reduction'])
                    
                    # Also get chi2_obs for color coding
                    if 'chi2_obs_total' in result:
                        chi2_obs_vals.append(result['chi2_obs_total'].get('chi2_obs', 1.0))
                    else:
                        chi2_obs_vals.append(1.0)
                
                # Fallback: if separate smoothness params don't exist, try combined
                elif param_name in ['smoothness_lambda_d13c', 'smoothness_lambda_dd']:
                    if 'smoothness_lambda_isotopes' in result['params']:
                        param_vals.append(result['params']['smoothness_lambda_isotopes'])
                        cost_reds.append(result['cost_reduction'])
                        if 'chi2_obs_total' in result:
                            chi2_obs_vals.append(result['chi2_obs_total'].get('chi2_obs', 1.0))
                        else:
                            chi2_obs_vals.append(1.0)
            
            if len(param_vals) == 0:
                # No data for this parameter
                ax.text(0.5, 0.5, f'No data for\n{param_label}', 
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color='gray', style='italic')
                ax.set_title(param_label, fontsize=11, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Convert to arrays
            param_vals = np.array(param_vals)
            cost_reds = np.array(cost_reds)
            chi2_obs_vals = np.array(chi2_obs_vals)
            
            # Color code by chi2_obs quality (green = good, red = bad)
            colors = []
            for chi2 in chi2_obs_vals:
                if 0.5 <= chi2 <= 1.5:
                    colors.append('#2ecc71')  # Green - good fit
                elif 0.3 <= chi2 <= 2.0:
                    colors.append('#f39c12')  # Orange - acceptable
                else:
                    colors.append('#e74c3c')  # Red - poor fit
            
            # Scatter plot
            scatter = ax.scatter(param_vals, cost_reds, s=120, alpha=0.7, 
                                c=colors, edgecolors='black', linewidth=1.5, zorder=10)
            
            # Fit trend line if enough points
            if len(np.unique(param_vals)) > 2:
                # Sort for plotting
                sort_idx = np.argsort(param_vals)
                param_sorted = param_vals[sort_idx]
                cost_sorted = cost_reds[sort_idx]
                
                # Polynomial fit (degree 2)
                try:
                    z = np.polyfit(param_sorted, cost_sorted, 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(min(param_vals), max(param_vals), 100)
                    ax.plot(x_smooth, p(x_smooth), 'k--', alpha=0.5, linewidth=2.5, 
                        label='Quadratic fit')
                except:
                    pass
            
            # Labels and formatting
            ax.set_xlabel(param_label, fontsize=11, fontweight='bold')
            ax.set_ylabel('Cost Reduction', fontsize=11, fontweight='bold')
            ax.set_title(f'{param_label}\n({len(param_vals)} runs)', 
                        fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3, linestyle=':', linewidth=1)
            
            # Mark best value
            best_idx = np.argmax(cost_reds)
            best_param = param_vals[best_idx]
            best_cost = cost_reds[best_idx]
            
            ax.scatter([best_param], [best_cost], s=300, marker='*', 
                    color='gold', edgecolors='darkred', linewidth=2, 
                    zorder=20, label='Best')
            
            ax.annotate(f'Best: {best_param:.3f}',
                    xy=(best_param, best_cost),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=9, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', 
                                alpha=0.95, edgecolor='darkgreen', linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', 
                                    lw=1.5, connectionstyle='arc3,rad=0.3'))
            
            # Add statistics
            param_range = np.ptp(param_vals)
            cost_range = np.ptp(cost_reds)
            
            stats_text = (
                f'Range: {np.min(param_vals):.3f} – {np.max(param_vals):.3f}\n'
                f'ΔCost: {cost_range:.0f}'
            )
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
            
            # Legend
            if len(np.unique(param_vals)) > 2:
                ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        
        # Overall title
        title_text = (
            'Parameter Sensitivity Analysis: Cost Reduction vs Parameter Values\n'
            'Color: Green = Good χ²_obs [0.5-1.5] | Orange = Acceptable [0.3-2.0] | Red = Poor | '
            'Gold Star = Best Run'
        )
        
        plt.suptitle(title_text, fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Saved to: {output_path}")

    def plot_enso_effect_size_comparison(self, output_path):
        """Cohen's d comparison plot."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 12))
        sources = ['wetlands_emissions', 'pyrogenic_emissions', 'wetlands_pyrogenic_emissions']
        source_labels = ['Wetlands', 'Pyrogenic', 'Wetlands + Pyrogenic']
        
        for idx, (source, label) in enumerate(zip(sources, source_labels)):
            ax = axes[idx]
            cohens_ds = []
            run_ids = []
            spearman_ps = []
            
            for result in self.results_sorted[:15]:
                if 'enso_causality' in result and source in result['enso_causality']:
                    cohens_ds.append(result['enso_causality'][source]['cohens_d'])
                    run_ids.append(result['run_id'])
                    spearman_ps.append(result['enso_causality'][source]['spearman_p'])
            
            if cohens_ds:
                colors = [self.result_colors[rid] if rid in self.top5_ids else 'lightgray' for rid in run_ids]
                edge_colors = ['green' if p < 0.05 else 'black' for p in spearman_ps]
                linewidths = [3 if p < 0.05 else 1 for p in spearman_ps]
                
                y_pos = range(len(cohens_ds))
                ax.barh(y_pos, cohens_ds, color=colors, alpha=0.7,
                       edgecolor=edge_colors, linewidth=linewidths)
                
                ax.axvline(0, color='black', linewidth=2)
                ax.axvline(0.2, color='blue', linestyle='--', alpha=0.5, label='Small (±0.2)')
                ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (±0.5)')
                ax.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Large (±0.8)')
                ax.axvline(-0.2, color='blue', linestyle='--', alpha=0.5)
                ax.axvline(-0.5, color='orange', linestyle='--', alpha=0.5)
                ax.axvline(-0.8, color='red', linestyle='--', alpha=0.5)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels([f"#{i+1}" for i in range(len(run_ids))], fontsize=9)
                ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
                ax.set_title(f'{label}\nEl Niño vs La Niña', fontsize=13, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                if idx == 0:
                    ax.legend(loc='lower right', fontsize=10)
                
                for i in range(min(5, len(cohens_ds))):
                    ax.text(cohens_ds[i], i, f' {cohens_ds[i]:.2f}',
                           ha='left' if cohens_ds[i] > 0 else 'right',
                           va='center', fontsize=9, fontweight='bold', color='black')
        
        plt.suptitle("ENSO Effect Sizes: Cohen's d\nTop 15 Runs; Top 5 in color, Green border = p<0.05",
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")
    
    def plot_isotope_enso_shifts_diff(self, output_path):
        """Isotope shift box plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sources = ['wetlands', 'pyrogenic']
        source_labels = ['Wetlands', 'Pyrogenic']
        isotopes = ['d13c', 'dd']
        isotope_labels = ['δ¹³C (‰)', '\u03B4D (‰)']
        
        for row, (isotope, iso_label) in enumerate(zip(isotopes, isotope_labels)):
            for col, (source, src_label) in enumerate(zip(sources, source_labels)):
                ax = axes[row, col]
                shifts_sh, shifts_nh, shifts_global = [], [], []
                p_vals_sh, p_vals_nh, p_vals_global = [], [], []
                
                for result in self.results:
                    if 'enso_causality' in result:
                        for hem, shifts_list, p_list in [('sh', shifts_sh, p_vals_sh), 
                                                          ('nh', shifts_nh, p_vals_nh), 
                                                          ('global', shifts_global, p_vals_global)]:
                            key = f'{source}_{isotope}_{hem}'
                            if key in result['enso_causality']:
                                shifts_list.append(result['enso_causality'][key]['isotope_shift_elnino_lanina'])
                                p_list.append(result['enso_causality'][key]['mann_whitney_p'])
                
                positions = [1, 2, 3]
                data_to_plot = [shifts_sh, shifts_nh, shifts_global]
                
                bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                               patch_artist=True, showmeans=True,
                               meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
                
                colors = ['lightblue', 'lightgreen', 'lightyellow']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                for i, (pos, p_vals) in enumerate(zip(positions, [p_vals_sh, p_vals_nh, p_vals_global])):
                    sig_count = sum(1 for p in p_vals if p < 0.05)
                    sig_pct = 100 * sig_count / len(p_vals) if p_vals else 0
                    y_max = ax.get_ylim()[1]
                    ax.text(pos, y_max * 0.95, f'{sig_pct:.0f}%\nsig',
                           ha='center', va='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='yellow' if sig_pct > 50 else 'white', alpha=0.7))
                
                ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
                ax.set_xticks(positions)
                ax.set_xticklabels(['SH', 'NH', 'Global'], fontsize=11)
                ax.set_ylabel(f'{iso_label} Shift\n(El Niño − La Niña)', fontsize=11)
                ax.set_title(f'{src_label} {iso_label.split()[0]}', fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3, axis='y')
                
                for i, (pos, shifts) in enumerate(zip(positions, data_to_plot)):
                    if shifts:
                        mean_val = np.mean(shifts)
                        ax.text(pos, ax.get_ylim()[0], f'μ={mean_val:.2f}',
                               ha='center', va='top', fontsize=8, color='red', fontweight='bold')
        
        plt.suptitle('Isotopic Shifts During ENSO (El Niño − La Niña)\n' +
                     'Box = IQR, Red diamond = mean, % sig = fraction with p<0.05',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")
    
    def plot_isotope_enso_shifts(self, output_path):
        """Isotope values by ENSO phase (all sources + lifetime).

        Notes:
        - For isotopic source signatures, this uses the pre-computed ENSO-phase means stored in
        result['enso_causality'] (mean_el_nino / mean_neutral / mean_la_nina).
        - For lifetime, values are computed from posterior sink scalings and then binned by ENSO
        phase year-by-year (1994–2022) using the MEI classification.
        - The lifetime row shows SH, NH, and GLOBAL.
        """
        fig, axes = plt.subplots(6, 2, figsize=(16, 24))

        sources = ['wetlands', 'agriculture', 'pyrogenic', 'fossil', 'waste', 'lifetime']
        source_labels = ['Wetlands', 'Agriculture', 'Pyrogenic', 'Fossil', 'Waste', 'Lifetime']
        isotopes = ['d13c', 'dd']
        isotope_labels = ['δ¹³C (‰)', '\u03B4D (‰)']
        phase_colors = ['#ff6b6b', '#95e1d3', '#6bcbf5']
        phase_labels = ['El Niño', 'Neutral', 'La Niña']

        enso_years = self.enso_analyzer.classify_enso_years(threshold=0.5)

        for row, (source, src_label) in enumerate(zip(sources, source_labels)):
            for col, (isotope, iso_label) in enumerate(zip(isotopes, isotope_labels)):

                # Skip δD for lifetime panel; place legend instead
                if source == 'lifetime' and isotope == 'dd':
                    ax = axes[row, col]
                    ax.axis('off')
                    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in phase_colors]
                    ax.legend(handles, phase_labels, title='ENSO Phase', loc='center')
                    continue

                ax = axes[row, col]

                # -----------------------------------------------------------------
                # LIFETIME PANEL (SH / NH / GLOBAL; year-by-year binning)
                # -----------------------------------------------------------------
                if source == 'lifetime':
                    data = {
                        'sh': {p: [] for p in ['el_nino', 'neutral', 'la_nina']},
                        'nh': {p: [] for p in ['el_nino', 'neutral', 'la_nina']},
                        'global': {p: [] for p in ['el_nino', 'neutral', 'la_nina']},
                    }

                    for result in self.results:
                        year_mask = (self.years >= 1994) & (self.years <= 2022)
                        years_subset = self.years[year_mask]

                        config = result['config']
                        sinks = result['sinks']

                        # Posterior sink scalings (% anomalies) -> posterior sink rates (1/yr)
                        post_sink_oh = 1 / config.lifetime_oh.reshape(1, 2) * (1 + sinks['oh'] / 100.0)
                        post_sink_strat = 1 / config.lifetime_strat.reshape(1, 2) * (1 + sinks['strat'] / 100.0)
                        post_sink_soil = 1 / config.lifetime_soil.reshape(1, 2) * (1 + sinks['soil'] / 100.0)
                        total_sink = post_sink_oh + post_sink_strat + post_sink_soil
                        lifetime = 1 / total_sink  # (year, [SH,NH]) in years

                        sh_vals = lifetime[year_mask, 0]
                        nh_vals = lifetime[year_mask, 1]
                        gl_vals = 0.5 * (sh_vals + nh_vals)

                        df = pd.DataFrame({
                            'Year': years_subset,
                            'sh': sh_vals,
                            'nh': nh_vals,
                            'global': gl_vals
                        }).merge(self.mei_df, on='Year', how='inner')

                        for phase in ['el_nino', 'neutral', 'la_nina']:
                            phase_years = enso_years[phase]
                            for region in ['sh', 'nh', 'global']:
                                vals = df[df['Year'].isin(phase_years)][region].values
                                vals = vals[np.isfinite(vals)]
                                if len(vals) > 0:
                                    data[region][phase].extend(vals.tolist())

                    # Box data for each region
                    sh_boxes = [data['sh']['el_nino'], data['sh']['neutral'], data['sh']['la_nina']]
                    nh_boxes = [data['nh']['el_nino'], data['nh']['neutral'], data['nh']['la_nina']]
                    gl_boxes = [data['global']['el_nino'], data['global']['neutral'], data['global']['la_nina']]

                    sh_positions = [1, 2, 3]
                    nh_positions = [5, 6, 7]
                    gl_positions = [9, 10, 11]

                    if any(len(d) > 0 for d in sh_boxes + nh_boxes + gl_boxes):
                        bp_sh = ax.boxplot(
                            sh_boxes, positions=sh_positions, widths=0.6,
                            patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='black', markersize=6)
                        )
                        for patch, colr in zip(bp_sh['boxes'], phase_colors):
                            patch.set_facecolor(colr)
                            patch.set_alpha(0.7)

                        bp_nh = ax.boxplot(
                            nh_boxes, positions=nh_positions, widths=0.6,
                            patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='black', markersize=6)
                        )
                        for patch, colr in zip(bp_nh['boxes'], phase_colors):
                            patch.set_facecolor(colr)
                            patch.set_alpha(0.7)

                        bp_gl = ax.boxplot(
                            gl_boxes, positions=gl_positions, widths=0.6,
                            patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='black', markersize=6)
                        )
                        for patch, colr in zip(bp_gl['boxes'], phase_colors):
                            patch.set_facecolor(colr)
                            patch.set_alpha(0.7)

                        # separators
                        ax.axvline(4, linestyle='--', color='gray', alpha=0.5)
                        ax.axvline(8, linestyle='--', color='gray', alpha=0.5)

                        ax.set_xticks([2, 6, 10])
                        ax.set_xticklabels(['SH', 'NH', 'Global'], fontsize=11, fontweight='bold')
                        ax.set_ylabel('Lifetime (years)')
                        ax.set_title(f'{src_label}')
                        ax.grid(alpha=0.3, axis='y')
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                        ax.set_xticks([])
                        ax.set_yticks([])

                    continue  # done with lifetime panel

                # -----------------------------------------------------------------
                # ISOTOPE PANELS (SH / NH; pre-computed ENSO means)
                # -----------------------------------------------------------------
                data_sh = {p: [] for p in ['el_nino', 'neutral', 'la_nina']}
                data_nh = {p: [] for p in ['el_nino', 'neutral', 'la_nina']}

                for result in self.results:
                    if 'enso_causality' not in result:
                        continue

                    key_sh = f'{source}_{isotope}_sh'
                    key_nh = f'{source}_{isotope}_nh'
                    if key_sh not in result['enso_causality'] or key_nh not in result['enso_causality']:
                        continue

                    shm = result['enso_causality'][key_sh]
                    nhm = result['enso_causality'][key_nh]

                    sh_vals = [shm.get('mean_el_nino', np.nan),
                            shm.get('mean_neutral', np.nan),
                            shm.get('mean_la_nina', np.nan)]
                    nh_vals = [nhm.get('mean_el_nino', np.nan),
                            nhm.get('mean_neutral', np.nan),
                            nhm.get('mean_la_nina', np.nan)]

                    for p, shv, nhv in zip(['el_nino', 'neutral', 'la_nina'], sh_vals, nh_vals):
                        if np.isfinite(shv):
                            data_sh[p].append(shv)
                        if np.isfinite(nhv):
                            data_nh[p].append(nhv)

                sh_boxes = [data_sh['el_nino'], data_sh['neutral'], data_sh['la_nina']]
                nh_boxes = [data_nh['el_nino'], data_nh['neutral'], data_nh['la_nina']]
                sh_positions = [1, 2, 3]
                nh_positions = [5, 6, 7]

                if any(len(d) > 0 for d in sh_boxes + nh_boxes):
                    bp_sh = ax.boxplot(
                        sh_boxes, positions=sh_positions, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='black', markersize=6)
                    )
                    for patch, colr in zip(bp_sh['boxes'], phase_colors):
                        patch.set_facecolor(colr)
                        patch.set_alpha(0.7)

                    ax2 = ax.twinx()
                    bp_nh = ax2.boxplot(
                        nh_boxes, positions=nh_positions, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='black', markersize=6)
                    )
                    for patch, colr in zip(bp_nh['boxes'], phase_colors):
                        patch.set_facecolor(colr)
                        patch.set_alpha(0.7)

                    ax.axvline(4, linestyle='--', color='gray', alpha=0.5)
                    ax.set_xticks([2, 6])
                    ax.set_xticklabels(['SH', 'NH'], fontsize=11, fontweight='bold')

                    if col == 0:
                        ax.set_ylabel(iso_label)
                    else:
                        ax2.set_ylabel(iso_label)

                    ax.set_title(f'{src_label}')
                    ax.grid(alpha=0.3, axis='y')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])

        plt.suptitle('Distribution of Posterior Source Isotopic Composition by ENSO Phase',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"    ✓ Saved to: {output_path}")

    def plot_emission_enso_shifts(self, output_path):
        """Emission values by ENSO phase (all 5 sources) with mean annotations and legend."""
        fig, axes = plt.subplots(5, 1, figsize=(14, 20))
        sources = ['wetlands', 'agriculture', 'pyrogenic', 'fossil', 'waste']
        source_labels = ['Wetlands', 'Agriculture', 'Pyrogenic', 'Fossil', 'Waste']
        phase_colors = ['#ff6b6b', '#95e1d3', '#6bcbf5']
        phase_labels = ['El Niño', 'Neutral', 'La Niña']

        for row, (source, src_label) in enumerate(zip(sources, source_labels)):
            ax = axes[row]

            data_sh = {p: [] for p in ['el_nino','neutral','la_nina']}
            data_nh = {p: [] for p in ['el_nino','neutral','la_nina']}

            for result in self.results:
                if 'enso_causality' not in result or source not in result['emissions']:
                    continue

                year_mask = (self.years >= 1994) & (self.years <= 2022)
                years_subset = self.years[year_mask]

                sh_vals = result['emissions'][source][year_mask, 0]
                nh_vals = result['emissions'][source][year_mask, 1]

                df_sh = pd.DataFrame({'Year': years_subset, 'Value': sh_vals}).merge(self.mei_df, on='Year')
                df_nh = pd.DataFrame({'Year': years_subset, 'Value': nh_vals}).merge(self.mei_df, on='Year')

                enso_years = self.enso_analyzer.classify_enso_years(threshold=0.5)

                for p in ['el_nino','neutral','la_nina']:
                    data_sh[p].extend(df_sh[df_sh['Year'].isin(enso_years[p])]['Value'].values)
                    data_nh[p].extend(df_nh[df_nh['Year'].isin(enso_years[p])]['Value'].values)

            sh_boxes = [data_sh['el_nino'], data_sh['neutral'], data_sh['la_nina']]
            nh_boxes = [data_nh['el_nino'], data_nh['neutral'], data_nh['la_nina']]
            sh_positions = [1, 2, 3]
            nh_positions = [5, 6, 7]

            if any(len(d) > 0 for d in sh_boxes + nh_boxes):
                # SH boxplot
                bp_sh = ax.boxplot(sh_boxes, positions=sh_positions, widths=0.6,
                                patch_artist=True, showmeans=True,
                                meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
                for patch, colr in zip(bp_sh['boxes'], phase_colors):
                    patch.set_facecolor(colr)
                    patch.set_alpha(0.7)

                # Annotate mean values
                for pos, box_data in zip(sh_positions, sh_boxes):
                    mean_val = np.mean(box_data)
                    ax.text(pos, mean_val + 0.5, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

                # NH boxplot
                ax2 = ax.twinx()
                bp_nh = ax2.boxplot(nh_boxes, positions=nh_positions, widths=0.6,
                                    patch_artist=True, showmeans=True,
                                    meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
                for patch, colr in zip(bp_nh['boxes'], phase_colors):
                    patch.set_facecolor(colr)
                    patch.set_alpha(0.7)

                # Annotate NH means
                for pos, box_data in zip(nh_positions, nh_boxes):
                    mean_val = np.mean(box_data)
                    ax2.text(pos, mean_val + 0.5, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

                ax.axvline(4, linestyle='--', color='gray', alpha=0.5)
                ax.set_xticks([2, 6])
                ax.set_xticklabels(['SH', 'NH'], fontsize=11, fontweight='bold')
                ax.set_ylabel('Emissions (Tg/yr)')
                ax.set_title(f'{src_label} Emissions')
                ax.grid(alpha=0.3)

                # Add legend for SH wetlands subplot
                if source == 'wetlands':
                    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in phase_colors]
                    ax.legend(handles, phase_labels, title='ENSO Phase', loc='upper left')

            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle('Distribution of Posterior Emissions by ENSO Phase', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"    ✓ Saved to: {output_path}")

    def plot_enso_causality_vs_parameters(self, output_path):
        """ENSO causality vs parameters."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()
        legend_ax = axes[-1]

        params_to_plot = [
            ('prior_isotope_d13c_error', 'Prior \u03B4\u00B9\u00B3C Error'),
            ('prior_isotope_dd_error', 'Prior \u03B4D Error'),
            ('obs_sigma_d13c', 'Obs \u03B4\u00B9\u00B3C Sigma'),
            ('obs_sigma_dd', 'Obs \u03B4D Sigma'),
            ('smoothness_lambda_emissions', 'Smooth λ (Em)'),
            ('smoothness_lambda_isotopes', 'Smooth λ (Iso)'),
            ('prior_emission_error', 'Prior Em Error')
        ]

        sources = ['wetlands', 'pyrogenic', 'wetlands_pyrogenic']
        source_labels = ['Wetlands', 'Pyrogenic', 'Wet+Pyro']
        source_markers = ['o', 's', '^']

        legend_handles = [
            plt.Line2D([], [], marker=m, linestyle='', color='black', 
                    markerfacecolor='white', markersize=10, label=lab)
            for m, lab in zip(source_markers, source_labels)
        ]

        for idx, (param_name, param_label) in enumerate(params_to_plot):
            ax = axes[idx]

            for source, label, marker in zip(sources, source_labels, source_markers):
                param_vals, spearman_rs, run_ids = [], [], []

                for result in self.results:
                    if param_name in result['params'] and 'enso_causality' in result:
                        key_name = f'{source}_emissions'
                        if key_name in result['enso_causality']:
                            param_vals.append(result['params'][param_name])
                            spearman_rs.append(result['enso_causality'][key_name]['spearman_r'])
                            run_ids.append(result['run_id'])

                if param_vals:
                    for pv, sr, rid in zip(param_vals, spearman_rs, run_ids):
                        if rid in self.top5_ids:
                            color = self.result_colors[rid]
                            alpha = 0.9
                            size = 100
                        else:
                            color = 'lightgray'
                            alpha = 0.3
                            size = 30

                        ax.scatter(pv, sr, s=size, alpha=alpha, marker=marker,
                                  c=[color], edgecolors='black', linewidths=1)

            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel(param_label, fontsize=10)
            ax.set_ylabel('Spearman r', fontsize=10)
            ax.grid(alpha=0.3)

        for idx in range(len(params_to_plot), len(axes) - 1):
            axes[idx].axis('off')

        legend_ax.axis('off')
        legend_ax.legend(handles=legend_handles, loc='center', fontsize=10, frameon=True)

        plt.suptitle('Causality (Spearman r): ENSO vs Posterior emissions\nTop 5 in color',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")

    def create_summary_table(self, output_path):
        """Create comprehensive CSV summary with NH and SH values stored separately."""
        data = []

        for result in self.results_sorted:
            row = {
                'run_id': result['run_id'],
                'name': result['name'],
                'cost_total': result['cost_total'],
                'cost_obs': result['cost_obs'],
                'cost_prior': result['cost_prior'],
                'cost_smooth': result['cost_smooth'],
                'cost_reduction': result['cost_reduction'],
                'rmse_total': result['rmse_total'],
                'rmse_1994_2022': result['rmse_1994_2022']
            }

            # Add parameter values
            for key, val in result['params'].items():
                if key != 'name':
                    row[key] = val

            # NEW: Add ENSO phase parameters
            if 'enso_phase_params' in result:
                for key, val in result['enso_phase_params'].items():
                    row[key] = val

            if 'enso_causality' in result:
                ec = result['enso_causality']

                # ENSO emissions summary
                for source in ['wetlands', 'pyrogenic']:
                    for hem in ['sh', 'nh']:
                        key = f'{source}_emissions' if hem == 'global' else f'{source}_emissions_{hem}'
                        if key in ec:
                            causality = ec[key]
                            prefix = f'{source[:3]}_emis_{hem}' if source != 'wetlands_pyrogenic' else f'wet_pyr_emis_{hem}'

                            #row[f'{prefix}_enso_spearman_r'] = causality.get('spearman_r', np.nan)
                            #row[f'{prefix}_enso_spearman_p'] = causality.get('spearman_p', np.nan)
                            #row[f'{prefix}_enso_pearson_r'] = causality.get('pearson_r', np.nan)
                            #row[f'{prefix}_enso_pearson_p'] = causality.get('pearson_p', np.nan)
                            #row[f'{prefix}_enso_mw_p'] = causality.get('mann_whitney_p', np.nan)
                            #row[f'{prefix}_enso_cohens_d'] = causality.get('cohens_d', np.nan)
                            #row[f'{prefix}_enso_emis_diff'] = causality.get('emission_diff_elnino_lanina', np.nan)
                            #row[f'{prefix}_enso_lag_corr'] = causality.get('max_lagged_corr', np.nan)
                            #row[f'{prefix}_enso_best_lag'] = causality.get('best_lag', np.nan)
                            row[f'{prefix}_mean_elnino'] = causality.get('mean_el_nino', np.nan)
                            row[f'{prefix}_mean_lanina'] = causality.get('mean_la_nina', np.nan)
                            row[f'{prefix}_mean_neutral'] = causality.get('mean_neutral', np.nan)
                            #row[f'{prefix}_n_elnino'] = causality.get('n_el_nino_years', np.nan)
                            #row[f'{prefix}_n_lanina'] = causality.get('n_la_nina_years', np.nan)
                            #row[f'{prefix}_n_neutral'] = causality.get('n_neutral_years', np.nan)

                # ENSO isotope summary
                for source in ['wetlands', 'pyrogenic']:
                    for isotope in ['d13c', 'dd']:
                        for hem in ['sh', 'nh']:
                            key = f'{source}_{isotope}_{hem}'
                            if key in ec:
                                iso_data = ec[key]
                                prefix = f'{source[:3]}_{isotope}_{hem}'

                                #row[f'{prefix}_iso_spearman_r'] = iso_data.get('spearman_r', np.nan)
                                #row[f'{prefix}_iso_spearman_p'] = iso_data.get('spearman_p', np.nan)
                                #row[f'{prefix}_iso_mw_p'] = iso_data.get('mann_whitney_p', np.nan)
                                #row[f'{prefix}_iso_cohens_d'] = iso_data.get('cohens_d', np.nan)
                                row[f'{prefix}_iso_shift'] = iso_data.get('isotope_shift_elnino_lanina', np.nan)
                                row[f'{prefix}_iso_mean_elnino'] = iso_data.get('mean_el_nino', np.nan)
                                row[f'{prefix}_iso_mean_lanina'] = iso_data.get('mean_la_nina', np.nan)
                                row[f'{prefix}_iso_mean_neutral'] = iso_data.get('mean_neutral', np.nan)

            data.append(row)

        df = pd.DataFrame(data)

        # Column ordering
        col_order = ['run_id', 'name', 'cost_reduction', 'cost_total', 'cost_obs', 
                    'cost_prior', 'cost_smooth', 'rmse_1994_2022', 'rmse_total']

        param_cols = ['prior_isotope_d13c_error', 'prior_isotope_dd_error', 
                    'obs_sigma_d13c', 'obs_sigma_dd', 'prior_emission_error',
                    'smoothness_lambda_emissions', 'smoothness_lambda_isotopes']
        for col in param_cols:
            if col in df.columns:
                col_order.append(col)

        # NEW: Add ENSO phase parameter columns (organized by source, parameter, hemisphere, phase)
        enso_phase_cols = [c for c in df.columns if any(
            c.startswith(prefix) for prefix in [
                'wetlands_emission_', 'wetlands_d13c_', 'wetlands_dd_',
                'pyrogenic_emission_', 'pyrogenic_d13c_', 'pyrogenic_dd_'
            ]
        )]
        col_order.extend(sorted(enso_phase_cols))

        # ENSO emission columns
        enso_emis_cols = [c for c in df.columns if 'enso' in c and 'iso' not in c]
        col_order.extend(sorted(enso_emis_cols))

        # ENSO isotope columns
        enso_iso_cols = [c for c in df.columns if 'iso' in c]
        col_order.extend(sorted(enso_iso_cols))

        # Other columns
        other_cols = [c for c in df.columns if c not in col_order]
        col_order.extend(other_cols)

        df = df[[c for c in col_order if c in df.columns]]
        df.to_csv(output_path, index=False)

        print(f"    ✓ Saved to: {output_path}")
        print(f"      Total columns: {len(df.columns)}")
        print(f"\n    Top 5 runs by cost reduction:")
        display_cols = ['run_id', 'name', 'cost_reduction', 'rmse_1994_2022']
        if 'wet_emis_sh_enso_spearman_r' in df.columns:
            display_cols.extend(['wet_emis_sh_enso_spearman_r', 'wet_emis_sh_enso_spearman_p', 'wet_emis_sh_enso_cohens_d'])
        print(df[display_cols].head(5).to_string(index=False))

        print(f"\n    ENSO Causality Summary:")
        for source, prefix in [('Wetlands', 'wet'), ('Pyrogenic', 'pyr'), 
                            ('Wetlands+Pyrogenic', 'wet_pyr')]:
            r_col = f'{prefix}_enso_spearman_r'
            p_col = f'{prefix}_enso_spearman_p'
            d_col = f'{prefix}_enso_cohens_d'

            if r_col in df.columns:
                sig_count = (df[p_col] < 0.05).sum()
                mean_r = df[r_col].mean()
                mean_d = df[d_col].mean()

                print(f"      {source}:")
                print(f"        Mean Spearman r: {mean_r:.3f}")
                print(f"        Mean Cohen's d: {mean_d:.3f}")
                print(f"        Significant runs (p<0.05): {sig_count}/{len(df)}")

    # ========================================================================
    # MEI SENSITIVITY PLOTS
    # ========================================================================

    def plot_mei_emission_sensitivity_old(self, output_path):
        """Create emission-MEI sensitivity plots WITH TOP 5 RUNS."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # ========================================================================
        # Get top 5 runs by cost reduction
        # ========================================================================
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        top_5_runs = sorted_results[:5]
        
        print(f"    Top 5 runs by cost reduction:")
        for i, r in enumerate(top_5_runs, 1):
            cost_red = r.get('cost_reduction', 0)
            chi2_obs = r.get('chi2_obs_total', {}).get('chi2_obs', -999)
            print(f"      #{i}: cost_reduction={cost_red:.1f}, χ²_obs={chi2_obs:.3f}, {r['name']}")
        
        # Create plotter
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        
        # Get MEI results from top 5 runs
        mei_results_list = [r['mei_sensitivity'] for r in top_5_runs]
        run_labels = [f"Run {i+1}" for i in range(len(top_5_runs))]
        
        # Call multi-run plot method
        fig = plotter.plot_emission_mei_sensitivity_multi(
            mei_results_list, 
            sources=['wetlands', 'pyrogenic'], 
            output_path=output_path,
            run_labels=run_labels
        )
        plt.close(fig)
        print(f"    ✓ Saved to: {output_path}")

    def plot_mei_d13c_sensitivity_old(self, output_path):
        """Create \u03B4\u00B9\u00B3C-MEI sensitivity plots WITH TOP 5 RUNS."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # Get top 5 runs
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        top_5_runs = sorted_results[:5]
        
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        mei_results_list = [r['mei_sensitivity'] for r in top_5_runs]
        run_labels = [f"Run {i+1}" for i in range(len(top_5_runs))]
        
        fig = plotter.plot_isotope_mei_sensitivity_multi(
            mei_results_list,
            isotope_type='d13c', 
            sources=['wetlands', 'pyrogenic'], 
            output_path=output_path,
            run_labels=run_labels
        )
        plt.close(fig)
        print(f"    ✓ Saved to: {output_path}")

    def plot_mei_dd_sensitivity_old(self, output_path):
        """Create \u03B4D-MEI sensitivity plots WITH TOP 5 RUNS."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # Get top 5 runs
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        top_5_runs = sorted_results[:5]
        
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        mei_results_list = [r['mei_sensitivity'] for r in top_5_runs]
        run_labels = [f"Run {i+1}" for i in range(len(top_5_runs))]
        
        fig = plotter.plot_isotope_mei_sensitivity_multi(
            mei_results_list,
            isotope_type='dd', 
            sources=['wetlands', 'pyrogenic'], 
            output_path=output_path,
            run_labels=run_labels
        )
        plt.close(fig)
        print(f"    ✓ Saved to: {output_path}")

    def plot_mei_sensitivity_summary_old(self, output_path):
        """Create MEI sensitivity summary bar chart WITH TOP 5 RUNS."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # Get top 5 runs
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        top_5_runs = sorted_results[:5]
        
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        mei_results_list = [r['mei_sensitivity'] for r in top_5_runs]
        run_labels = [f"Run {i+1}" for i in range(len(top_5_runs))]
        
        fig = plotter.plot_sensitivity_summary_multi(
            mei_results_list,
            output_path=output_path,
            run_labels=run_labels
        )
        plt.close(fig)
        print(f"    ✓ Saved to: {output_path}")

    def export_mei_sensitivity_csvs_old(self, output_path):
        """Export MEI sensitivity metrics to CSV files."""
        if 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        mei_results = self.results[0]['mei_sensitivity']
        
        # Emissions CSV
        emis_rows = []
        for source, sens in mei_results['emissions'].items():
            emis_rows.append({
                'Source': source,
                'Slope_Tg_per_MEI': sens.slope,
                'Std_Error': sens.std_error,
                'R_squared': sens.r_squared,
                'p_value': sens.p_value,
                'Mean_Emission_Tg': sens.mean_emission,
                'El_Nino_Effect_Tg': sens.el_nino_sensitivity,
                'La_Nina_Effect_Tg': sens.la_nina_sensitivity,
                'Total_Range_Tg': sens.total_range
            })
        
        emis_df = pd.DataFrame(emis_rows)
        emis_path = os.path.join(self.output_dir, 'mei_sensitivity_emissions.csv')
        emis_df.to_csv(emis_path, index=False, float_format='%.4f')
        print(f"    ✓ Emissions: {emis_path}")
        
        # \u03B4\u00B9\u00B3C CSV
        d13c_rows = []
        for source, sens in mei_results['isotopes_d13c'].items():
            d13c_rows.append({
                'Source': source,
                'Slope_permil_per_MEI': sens.slope,
                'Std_Error': sens.std_error,
                'R_squared': sens.r_squared,
                'p_value': sens.p_value,
                'Mean_d13C_permil': sens.mean_isotope,
                'El_Nino_Shift_permil': sens.el_nino_shift,
                'La_Nina_Shift_permil': sens.la_nina_shift,
                'Total_Range_permil': sens.total_range
            })
        
        d13c_df = pd.DataFrame(d13c_rows)
        d13c_path = os.path.join(self.output_dir, 'mei_sensitivity_d13c.csv')
        d13c_df.to_csv(d13c_path, index=False, float_format='%.4f')
        print(f"    ✓ \u03B4\u00B9\u00B3C: {d13c_path}")
        
        # \u03B4D CSV
        dd_rows = []
        for source, sens in mei_results['isotopes_dd'].items():
            dd_rows.append({
                'Source': source,
                'Slope_permil_per_MEI': sens.slope,
                'Std_Error': sens.std_error,
                'R_squared': sens.r_squared,
                'p_value': sens.p_value,
                'Mean_dD_permil': sens.mean_isotope,
                'El_Nino_Shift_permil': sens.el_nino_shift,
                'La_Nina_Shift_permil': sens.la_nina_shift,
                'Total_Range_permil': sens.total_range
            })
        
        dd_df = pd.DataFrame(dd_rows)
        dd_path = os.path.join(self.output_dir, 'mei_sensitivity_dd.csv')
        dd_df.to_csv(dd_path, index=False, float_format='%.4f')
        print(f"    ✓ \u03B4D: {dd_path}")
        
        # Summary CSV
        summary_path = os.path.join(self.output_dir, 'mei_sensitivity_summary.csv')
        mei_results['summary_table'].to_csv(summary_path, index=False, float_format='%.4f')
        print(f"    ✓ Summary: {summary_path}")
    
    def plot_mei_emission_sensitivity(self, output_path):
        """Create emission-MEI sensitivity plots for BEST RUN - ALL REGIONS."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # Get ONLY the best run by cost reduction
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        best_run = sorted_results[0]
        
        print(f"    Using best run by cost reduction:")
        cost_red = best_run.get('cost_reduction', 0)
        chi2_obs = best_run.get('chi2_obs_total', {}).get('chi2_obs', -999)
        print(f"      cost_reduction={cost_red:.1f}, χ²_obs={chi2_obs:.3f}, {best_run['name']}")
        
        # Create plotter
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        
        # Wrap in list for compatibility with _multi function signature
        mei_results_list = [best_run['mei_sensitivity']]
        
        # Plot for each region
        for region in ['nh', 'sh', 'global']:
            region_label = region.upper()
            output_path_region = output_path.replace('.png', f'_{region}.png')
            
            fig = plotter.plot_emission_mei_sensitivity_multi(
                mei_results_list, 
                sources=['wetlands', 'pyrogenic'], 
                region=region,
                output_path=output_path_region,
                run_labels=[best_run['name']]
            )
            if fig:
                plt.close(fig)
            print(f"    ✓ Saved {region_label}: {output_path_region}")

    def plot_mei_d13c_sensitivity(self, output_path):
        """Create \u03B4\u00B9\u00B3C-MEI sensitivity plots for BEST RUN - ALL REGIONS."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # Get ONLY the best run
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        best_run = sorted_results[0]
        
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        mei_results_list = [best_run['mei_sensitivity']]
        
        # Plot for each region
        for region in ['nh', 'sh', 'global']:
            region_label = region.upper()
            output_path_region = output_path.replace('.png', f'_{region}.png')
            
            fig = plotter.plot_isotope_mei_sensitivity_multi(
                mei_results_list,
                isotope_type='d13c', 
                sources=['wetlands', 'pyrogenic'], 
                region=region,
                output_path=output_path_region,
                run_labels=[best_run['name']]
            )
            if fig:
                plt.close(fig)
            print(f"    ✓ Saved {region_label}: {output_path_region}")

    def plot_mei_dd_sensitivity(self, output_path):
        """Create \u03B4D-MEI sensitivity plots for BEST RUN - ALL REGIONS."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # Get ONLY the best run
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        best_run = sorted_results[0]
        
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        mei_results_list = [best_run['mei_sensitivity']]
        
        # Plot for each region
        for region in ['nh', 'sh', 'global']:
            region_label = region.upper()
            output_path_region = output_path.replace('.png', f'_{region}.png')
            
            fig = plotter.plot_isotope_mei_sensitivity_multi(
                mei_results_list,
                isotope_type='dd', 
                sources=['wetlands', 'pyrogenic'], 
                region=region,
                output_path=output_path_region,
                run_labels=[best_run['name']]
            )
            if fig:
                plt.close(fig)
            print(f"    ✓ Saved {region_label}: {output_path_region}")

    def plot_mei_sensitivity_summary(self, output_path):
        """Create MEI sensitivity summary bar chart for BEST RUN - ALL REGIONS."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # Get ONLY the best run
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        best_run = sorted_results[0]
        
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        mei_results_list = [best_run['mei_sensitivity']]
        
        # Plot for each region
        for region in ['nh', 'sh', 'global']:
            region_label = region.upper()
            output_path_region = output_path.replace('.png', f'_{region}.png')
            
            fig = plotter.plot_sensitivity_summary_multi(
                mei_results_list,
                region=region,
                output_path=output_path_region,
                run_labels=[best_run['name']]
            )
            if fig:
                plt.close(fig)
            print(f"    ✓ Saved {region_label}: {output_path_region}")

    def plot_mei_sensitivity_summary_nhsh_old(self, output_path):
        """Create MEI sensitivity summary bar chart for BEST RUN - NH/SH/Global SPLIT."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # Get ONLY the best run
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        best_run = sorted_results[0]
        
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        mei_results_list = [best_run['mei_sensitivity']]
        
        # Create single plot with NH/SH/Global split bars
        fig = plotter.plot_sensitivity_summary_multi_nhsh_old(
            mei_results_list,
            output_path=output_path,
            run_labels=[best_run['name']]
        )
        
        if fig:
            plt.close(fig)
            print(f"    ✓ Saved NH/SH/Global split plot: {output_path}")

    def plot_mei_sensitivity_summary_nhsh(self, output_path):
        """Create MEI sensitivity summary bar chart for BEST RUN - NH/SH/Global SPLIT with MEI time series."""
        if not HAS_MEI_PLOTTER or 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        # Get ONLY the best run
        sorted_results = sorted(self.results, 
                            key=lambda r: r.get('cost_reduction', -999), 
                            reverse=True)
        best_run = sorted_results[0]
        
        plotter = MEISensitivityPlotter(self.mei_df, self.output_dir)
        mei_results_list = [best_run['mei_sensitivity']]
        
        # Create single plot with NH/SH/Global split bars + MEI time series
        fig = plotter.plot_sensitivity_summary_multi_nhsh(
            mei_results_list,
            output_path=output_path,
            run_labels=[best_run['name']]
        )
        
        if fig:
            plt.close(fig)
            print(f"    ✓ Saved NH/SH/Global split + MEI time series plot: {output_path}")

    def export_mei_sensitivity_csvs(self, output_path):
        """Export MEI sensitivity metrics to CSV files - ALL REGIONS."""
        if 'mei_sensitivity' not in self.results[0] or not self.results[0]['mei_sensitivity']:
            print("    ⚠ Skipping - no MEI sensitivity data")
            return
        
        mei_results = self.results[0]['mei_sensitivity']
        
        # Export for each region
        for region in ['nh', 'sh', 'global']:
            region_upper = region.upper()
            
            if region not in mei_results['emissions']:
                print(f"    ⚠ Region {region_upper} not found in data")
                continue
            
            # Emissions CSV
            emis_rows = []
            for source, sens in mei_results['emissions'][region].items():
                emis_rows.append({
                    'Region': region_upper,
                    'Source': source,
                    'Slope_Tg_per_MEI': sens.slope,
                    'Std_Error': sens.std_error,
                    'R_squared': sens.r_squared,
                    'p_value': sens.p_value,
                    'Mean_Emission_Tg': sens.mean_emission,
                    'El_Nino_Effect_Tg': sens.el_nino_sensitivity,
                    'La_Nina_Effect_Tg': sens.la_nina_sensitivity,
                    'Total_Range_Tg': sens.total_range
                })
            
            if len(emis_rows) > 0:
                emis_df = pd.DataFrame(emis_rows)
                emis_path = os.path.join(self.output_dir, f'mei_sensitivity_emissions_{region}.csv')
                emis_df.to_csv(emis_path, index=False, float_format='%.4f')
                print(f"    ✓ Emissions ({region_upper}): {emis_path}")
            
            # \u03B4\u00B9\u00B3C CSV
            if 'isotopes_d13c' in mei_results and region in mei_results['isotopes_d13c']:
                d13c_rows = []
                for source, sens in mei_results['isotopes_d13c'][region].items():
                    d13c_rows.append({
                        'Region': region_upper,
                        'Source': source,
                        'Slope_permil_per_MEI': sens.slope,
                        'Std_Error': sens.std_error,
                        'R_squared': sens.r_squared,
                        'p_value': sens.p_value,
                        'Mean_d13C_permil': sens.mean_isotope,
                        'El_Nino_Shift_permil': sens.el_nino_shift,
                        'La_Nina_Shift_permil': sens.la_nina_shift,
                        'Total_Range_permil': sens.total_range
                    })
                
                if len(d13c_rows) > 0:
                    d13c_df = pd.DataFrame(d13c_rows)
                    d13c_path = os.path.join(self.output_dir, f'mei_sensitivity_d13c_{region}.csv')
                    d13c_df.to_csv(d13c_path, index=False, float_format='%.4f')
                    print(f"    ✓ \u03B4\u00B9\u00B3C ({region_upper}): {d13c_path}")
            
            # \u03B4D CSV
            if 'isotopes_dd' in mei_results and region in mei_results['isotopes_dd']:
                dd_rows = []
                for source, sens in mei_results['isotopes_dd'][region].items():
                    dd_rows.append({
                        'Region': region_upper,
                        'Source': source,
                        'Slope_permil_per_MEI': sens.slope,
                        'Std_Error': sens.std_error,
                        'R_squared': sens.r_squared,
                        'p_value': sens.p_value,
                        'Mean_dD_permil': sens.mean_isotope,
                        'El_Nino_Shift_permil': sens.el_nino_shift,
                        'La_Nina_Shift_permil': sens.la_nina_shift,
                        'Total_Range_permil': sens.total_range
                    })
                
                if len(dd_rows) > 0:
                    dd_df = pd.DataFrame(dd_rows)
                    dd_path = os.path.join(self.output_dir, f'mei_sensitivity_dd_{region}.csv')
                    dd_df.to_csv(dd_path, index=False, float_format='%.4f')
                    print(f"    ✓ \u03B4D ({region_upper}): {dd_path}")

    # ========================================================================
    # CHI-SQUARE DIAGNOSTICS PLOTS
    # ========================================================================
    
    def plot_chi2_vs_parameters(self, output_path):
        """Plot chi-square metrics vs parameters."""
        if 'chi2_obs_total' not in self.results[0]:
            print("    ⚠ Skipping - no chi2 data")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Parameters to plot
        params = [
            ('prior_isotope_d13c_error', 'Prior \u03B4\u00B9\u00B3C Error (‰)'),
            ('prior_isotope_dd_error', 'Prior \u03B4D Error (‰)'),
            ('obs_sigma_d13c', 'Obs \u03B4\u00B9\u00B3C σ (‰)'),
            ('obs_sigma_dd', 'Obs \u03B4D σ (‰)'),
            ('prior_emission_error', 'Prior Emission Error'),
        ]
        
        for idx, (param_name, param_label) in enumerate(params):
            ax = axes[idx]
            
            param_vals = []
            chi2_obs_vals = []
            chi2_prior_vals = []
            colors = []
            
            for result in self.results:
                if param_name in result['params'] and 'chi2_obs_total' in result:
                    param_vals.append(result['params'][param_name])
                    chi2_obs_vals.append(result['chi2_obs_total']['chi2_obs'])
                    chi2_prior_vals.append(result['chi2_prior_total']['chi2_prior'])
                    
                    # Color by quality (green if both near 1)
                    chi2_obs = result['chi2_obs_total']['chi2_obs']
                    chi2_prior = result['chi2_prior_total']['chi2_prior']
                    if 0.5 <= chi2_obs <= 1.5 and 0.5 <= chi2_prior <= 1.5:
                        colors.append('green')
                    elif 0.5 <= chi2_obs <= 1.5 or 0.5 <= chi2_prior <= 1.5:
                        colors.append('orange')
                    else:
                        colors.append('red')
            
            if param_vals:
                # Plot chi2_obs
                ax.scatter(param_vals, chi2_obs_vals, s=100, alpha=0.7, 
                          c=colors, edgecolors='black', label='χ²_obs', marker='o')
                
                # Plot chi2_prior
                ax.scatter(param_vals, chi2_prior_vals, s=80, alpha=0.5,
                          c=colors, edgecolors='black', label='χ²_prior', marker='s')
                
                # Reference lines
                ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Ideal (χ²=1)')
                ax.axhspan(0.5, 1.5, color='green', alpha=0.1)
                
                ax.set_xlabel(param_label, fontsize=11, fontweight='bold')
                ax.set_ylabel('χ² Value', fontsize=11)
                ax.set_title(f'χ² vs {param_label}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, loc='best')
                ax.grid(alpha=0.3)
        
        # Hide unused subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")
    
    def plot_chi2_by_tracer(self, output_path):
        """Plot chi-square breakdown by tracer."""
        if 'chi2_by_tracer' not in self.results[0]:
            print("    ⚠ Skipping - no chi2 by tracer data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get chi2 by tracer for all results
        tracers = ['ch4', 'd13c', 'dd']
        tracer_labels = ['CH₄', 'δ¹³C', '\u03B4D']
        
        data_by_tracer = {tracer: [] for tracer in tracers}
        
        for result in self.results[:15]:  # Top 15 results
            if 'chi2_by_tracer' in result:
                chi2_df = result['chi2_by_tracer']
                for tracer in tracers:
                    tracer_data = chi2_df[chi2_df['tracer'] == tracer]
                    if len(tracer_data) > 0:
                        data_by_tracer[tracer].append(tracer_data['chi2'].values[0])
        
        # Box plot
        positions = [1, 2, 3]
        box_data = [data_by_tracer[t] for t in tracers]
        
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Reference line
        ax.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (χ²=1)')
        ax.axhspan(0.5, 1.5, color='green', alpha=0.1, label='Good range')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(tracer_labels, fontsize=12, fontweight='bold')
        ax.set_ylabel('χ² Value', fontsize=12, fontweight='bold')
        ax.set_title('χ² by Tracer (Top 15 Results)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")
    
    def plot_chi2_summary(self, output_path):
        """Create comprehensive chi2 summary plot."""
        if 'chi2_obs_total' not in self.results[0]:
            print("    ⚠ Skipping - no chi2 data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: chi2_obs vs chi2_prior scatter
        ax = axes[0, 0]
        chi2_obs = [r['chi2_obs_total']['chi2_obs'] for r in self.results if 'chi2_obs_total' in r]
        chi2_prior = [r['chi2_prior_total']['chi2_prior'] for r in self.results if 'chi2_prior_total' in r]
        
        colors = []
        for obs, prior in zip(chi2_obs, chi2_prior):
            if 0.5 <= obs <= 1.5 and 0.5 <= prior <= 1.5:
                colors.append('green')
            elif 0.5 <= obs <= 1.5 or 0.5 <= prior <= 1.5:
                colors.append('orange')
            else:
                colors.append('red')
        
        ax.scatter(chi2_obs, chi2_prior, s=100, alpha=0.7, c=colors, edgecolors='black')
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='green', linestyle='--', alpha=0.5)
        ax.axhspan(0.5, 1.5, color='green', alpha=0.1)
        ax.axvspan(0.5, 1.5, color='green', alpha=0.1)
        ax.set_xlabel('χ²_obs', fontsize=12, fontweight='bold')
        ax.set_ylabel('χ²_prior', fontsize=12, fontweight='bold')
        ax.set_title('(a) χ²_obs vs χ²_prior', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Panel 2: Histogram of chi2_obs
        ax = axes[0, 1]
        ax.hist(chi2_obs, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Ideal (χ²=1)')
        ax.axvspan(0.5, 1.5, color='green', alpha=0.2, label='Good range')
        ax.set_xlabel('χ²_obs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('(b) Distribution of χ²_obs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        
        # Panel 3: chi2 vs cost reduction
        ax = axes[1, 0]
        cost_red = [r['cost_reduction'] for r in self.results if 'chi2_obs_total' in r]
        ax.scatter(chi2_obs, cost_red, s=100, alpha=0.7, c=colors, edgecolors='black')
        ax.axvline(1.0, color='green', linestyle='--', alpha=0.5)
        ax.axvspan(0.5, 1.5, color='green', alpha=0.1)
        ax.set_xlabel('χ²_obs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cost Reduction', fontsize=12, fontweight='bold')
        ax.set_title('(c) χ²_obs vs Cost Reduction', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Panel 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        good_runs = sum([1 for c in colors if c == 'green'])
        ok_runs = sum([1 for c in colors if c == 'orange'])
        bad_runs = sum([1 for c in colors if c == 'red'])
        
        summary_text = f"""
        CHI-SQUARE SUMMARY
        {'='*40}
        
        Total runs: {len(self.results)}
        
        Quality breakdown:
          ✓ Good (both χ² in [0.5, 1.5]): {good_runs}
          ⚠ OK (one χ² in range):         {ok_runs}
          ✗ Poor (neither in range):      {bad_runs}
        
        Statistics:
          χ²_obs:
            Mean:   {np.mean(chi2_obs):.3f}
            Median: {np.median(chi2_obs):.3f}
            Std:    {np.std(chi2_obs):.3f}
          
          χ²_prior:
            Mean:   {np.mean(chi2_prior):.3f}
            Median: {np.median(chi2_prior):.3f}
            Std:    {np.std(chi2_prior):.3f}
        
        Interpretation:
          χ² ≈ 1: Model fits data within uncertainties
          χ² < 1: Overfitting or overestimated errors
          χ² > 1: Underfitting or underestimated errors
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
               verticalalignment='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved to: {output_path}")
    
    # ========================================================================
    # ROBUSTNESS CHECKS
    # ========================================================================

    def plot_robustness_check(self, output_path):
        """
        Create comprehensive robustness check plot showing ENSO signal consistency.
        
        Plots:
        - Emission-MEI correlations
        - Emission phase differences
        - Isotope shifts (\u03B4\u00B9\u00B3C and \u03B4D)
        """
        # Define metrics with correct extraction functions
        metrics_config = [
            # =====================================================================
            # EMISSION METRICS
            # =====================================================================
            {
                'extract_func': lambda r: r['enso_causality']['wetlands_emissions']['spearman_r'] 
                    if 'wetlands_emissions' in r['enso_causality'] else None,
                'title': 'Wetlands Emission-MEI\nCorrelation (r)',
                'ylabel': 'Spearman r',
                'description': 'Global emissions vs MEI',
                'ylim': (-0.8, -0.2),
            },
            {
                'extract_func': lambda r: r['enso_causality']['pyrogenic_emissions']['spearman_r']
                    if 'pyrogenic_emissions' in r['enso_causality'] else None,
                'title': 'Pyrogenic Emission-MEI\nCorrelation (r)',
                'ylabel': 'Spearman r',
                'description': 'Global emissions vs MEI',
                'ylim': (0.5, 0.9),
            },
            {
                'extract_func': lambda r: r['enso_causality']['wetlands_emissions']['emission_diff_elnino_lanina']
                    if 'wetlands_emissions' in r['enso_causality'] else None,
                'title': 'Wetlands ΔEmission\n(El Niño − La Niña)',
                'ylabel': 'ΔE (Tg/yr)',
                'description': 'Global emission difference',
                'ylim': (-20, -10),
            },
            {
                'extract_func': lambda r: r['enso_causality']['pyrogenic_emissions']['emission_diff_elnino_lanina']
                    if 'pyrogenic_emissions' in r['enso_causality'] else None,
                'title': 'Pyrogenic ΔEmission\n(El Niño − La Niña)',
                'ylabel': 'ΔE (Tg/yr)',
                'description': 'Global emission difference',
                'ylim': (10, 20),
            },
            
            # =====================================================================
            # ISOTOPE METRICS - CORRECTED KEY: 'isotope_shift_elnino_lanina'
            # =====================================================================
            {
                'extract_func': lambda r: r['enso_causality']['wetlands_d13c_sh']['isotope_shift_elnino_lanina']
                    if 'wetlands_d13c_sh' in r['enso_causality'] else None,
                'title': 'Wetlands \u03B4\u00B9\u00B3C Shift (SH)\n(El Niño − La Niña)',
                'ylabel': 'Δ\u03B4\u00B9\u00B3C (‰)',
                'description': 'SH isotope shift',
                'ylim': (-2, 0.5),
            },
            {
                'extract_func': lambda r: r['enso_causality']['pyrogenic_d13c_sh']['isotope_shift_elnino_lanina']
                    if 'pyrogenic_d13c_sh' in r['enso_causality'] else None,
                'title': 'Pyrogenic \u03B4\u00B9\u00B3C Shift (SH)\n(El Niño − La Niña)',
                'ylabel': 'Δ\u03B4\u00B9\u00B3C (‰)',
                'description': 'SH isotope shift',
                'ylim': (-1.5, 0.5),
            },
            {
                'extract_func': lambda r: r['enso_causality']['wetlands_dd_sh']['isotope_shift_elnino_lanina']
                    if 'wetlands_dd_sh' in r['enso_causality'] else None,
                'title': 'Wetlands \u03B4D Shift (SH)\n(El Niño − La Niña)',
                'ylabel': 'Δ\u03B4D (‰)',
                'description': 'SH isotope shift',
                'ylim': (-15, 10),
            },
            {
                'extract_func': lambda r: r['enso_causality']['pyrogenic_dd_sh']['isotope_shift_elnino_lanina']
                    if 'pyrogenic_dd_sh' in r['enso_causality'] else None,
                'title': 'Pyrogenic \u03B4D Shift (SH)\n(El Niño − La Niña)',
                'ylabel': 'Δ\u03B4D (‰)',
                'description': 'SH isotope shift',
                'ylim': (-10, 10),
            },
        ]
        
        # Create figure
        n_metrics = len(metrics_config)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        # Process each metric
        for idx, config in enumerate(metrics_config):
            ax = axes[idx]
            
            # Extract values from all results
            values = []
            for result in self.results:
                if 'enso_causality' not in result:
                    continue
                
                val = config['extract_func'](result)
                if val is not None and not np.isnan(val):
                    values.append(val)
            
            # Check if we have data
            if len(values) == 0:
                ax.text(0.5, 0.5, 'No data available', 
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, color='red', fontweight='bold')
                ax.set_title(config['title'], fontsize=11, fontweight='bold')
                ax.set_xticks([])
                ax.set_facecolor('#f0f0f0')
                continue
            
            # Convert to array
            values = np.array(values)
            
            # Create box plot
            bp = ax.boxplot([values], positions=[1], widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                        markersize=10, markeredgecolor='darkred',
                                        markeredgewidth=1.5, zorder=10))
            
            # Style box plot
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][0].set_edgecolor('darkblue')
            bp['boxes'][0].set_linewidth(2)
            
            for whisker in bp['whiskers']:
                whisker.set(linewidth=2, color='darkblue')
            for cap in bp['caps']:
                cap.set(linewidth=2, color='darkblue')
            bp['medians'][0].set(linewidth=3, color='darkgreen')
            
            # Scatter individual points
            jitter = np.random.normal(0, 0.02, len(values))  # Small jitter for visibility
            ax.scatter([1] * len(values) + jitter, values, alpha=0.5, s=5, 
                    color='navy', edgecolors='black', linewidth=0.8, zorder=10)
            
            # Reference line at 0
            ax.axhline(0, color='gray', linestyle='--', linewidth=2.5, alpha=0.7, zorder=1)
            
            # Calculate statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            cv = (std_val / abs(mean_val) * 100) if abs(mean_val) > 1e-10 else np.inf
            
            # Check if significant (all runs have same sign and |mean| > 2*std)
            all_negative = np.all(values < 0)
            all_positive = np.all(values > 0)
            strong_signal = abs(mean_val) > 2 * std_val  # Signal > 2σ
            is_significant = (all_negative or all_positive) and strong_signal
            
            # Stats text box
            stats_text = (
                f'Mean:   {mean_val:+.3f}\n'
                f'Median: {median_val:+.3f}\n'
                f'Std:    {std_val:.3f}\n'
                f'CV:     {cv:.1f}%\n'
                f'N runs: {len(values)}'
            )
            
            # Add significance indicator
            if is_significant:
                sign = '+' if all_positive else '−'
                stats_text += f'\n✓ {sign} in all runs'
                stats_text += f'\n✓ |μ| > 2σ'
            elif all_negative or all_positive:
                sign = '+' if all_positive else '−'
                stats_text += f'\n✓ {sign} in all runs'
            
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95,
                            edgecolor='black', linewidth=1.5),
                family='monospace')
            
            # Color code background based on CV
            if cv < 20:
                ax.set_facecolor('#e8f5e9')  # Light green - ROBUST
                robustness = 'ROBUST'
            elif cv < 50:
                ax.set_facecolor('#fff9e6')  # Light yellow - MODERATE
                robustness = 'MODERATE'
            else:
                ax.set_facecolor('#ffebee')  # Light red - SENSITIVE
                robustness = 'SENSITIVE'
            
            # Labels
            ax.set_ylabel(config['ylabel'], fontsize=11, fontweight='bold')
            ax.set_title(f"{config['title']}\n({robustness}, CV={cv:.1f}%)", 
                        fontsize=11, fontweight='bold')
            
            # Add description at bottom
            ax.text(0.5, -0.12, config['description'], 
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, style='italic', color='dimgray')
            
            ax.set_xticks([])
            ax.set_xlim(0.6, 1.4)
            ax.grid(alpha=0.3, axis='y', linestyle=':', linewidth=1)
            
            # Set y-limits
            if 'ylim' in config and not np.isinf(cv):
                # Use config limits if data fits reasonably
                data_min, data_max = np.min(values), np.max(values)
                ylim_min, ylim_max = config['ylim']
                
                # Only use config limits if data is within reasonable range
                if data_min >= ylim_min * 0.5 and data_max <= ylim_max * 2:
                    ax.set_ylim(config['ylim'])
                else:
                    # Auto-scale with padding
                    yrange = np.ptp(values)
                    ypad = max(yrange * 0.2, 0.1)
                    ax.set_ylim(data_min - ypad, data_max + ypad)
            else:
                # Auto-scale
                if len(values) > 0:
                    yrange = np.ptp(values)
                    ypad = max(yrange * 0.2, 0.1)
                    ax.set_ylim(np.min(values) - ypad, np.max(values) + ypad)
        
        # Hide unused subplots
        for idx in range(len(metrics_config), len(axes)):
            axes[idx].axis('off')
        
        # Overall title
        title_text = (
            'Robustness Check: ENSO Signal Consistency Across Parameter Variations\n'
            'Background Color: Green = CV<20% (ROBUST) | Yellow = CV<50% (MODERATE) | Red = CV≥50% (SENSITIVE)\n'
            'Box = IQR (25-75%), Red Diamond = Mean, Green Line = Median, Whiskers = Range, Navy Dots = Individual Runs'
        )
        
        plt.suptitle(title_text, fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Saved to: {output_path}")
        
        return fig


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Standalone plotter for ENSO inversion results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standalone_plotter.py
  python standalone_plotter.py --results_file some/other/path.pkl
        """
    )
    
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to specific all_results.pkl file')

    args = parser.parse_args()

    # -------------------------------------------------------------
    # DEFAULT PKL PATH
    # -------------------------------------------------------------
    default_results_file = (
        '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/PostDoc/Conference and Meetings/ENSO/ENSO Manuscript/Enhanced_run/multi_inversion_results_v4/all_results.pkl'
    )

    # Determine file path
    results_file = args.results_file if args.results_file else default_results_file

    if not os.path.exists(results_file):
        print(f"Error: Results file not found:\n{results_file}")
        return

    # -------------------------------------------------------------
    # SAVE PLOTS IN SAME DIRECTORY AS PKL
    # -------------------------------------------------------------
    output_dir = os.path.dirname(results_file)
    print(f"\nSaving plots to same directory as PKL:\n  {output_dir}\n")

    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║           STANDALONE ENSO INVERSION PLOTTER                       ║
╚═══════════════════════════════════════════════════════════════════╝
Using results file:
  {results_file}
""")

    # Initialize plotter
    try:
        plotter = StandalonePlotter(results_file, output_dir=output_dir)
    except Exception as e:
        print(f"✗ Failed to initialize plotter: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate plots
    try:
        # DEBUG: Print available keys
        #plotter.debug_print_enso_causality_keys()
        plotter.plot_all()
    except Exception as e:
        print(f"✗ Failed during plotting: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    PLOTTING COMPLETE!                             ║
║                                                                   ║
║  All plots saved to: {output_dir:<44} ║
╚═══════════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()