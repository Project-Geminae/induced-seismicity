import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# ========================================
# DATA: Individual Well-Level vs Event-Level Aggregated
# ========================================

# Complete radius array (1-20km)
radius = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# Individual Well-Level Data (Step 7 - Enhanced Well-Level Analysis)
individual_total = np.array([6.472e-06, 2.337e-05, 3.031e-05, 3.137e-05, 2.456e-05, 1.417e-05,
                             1.461e-05, 7.607e-06, 7.969e-06, 9.095e-06, 8.176e-06, 6.199e-06,
                             4.725e-06, 4.727e-06, 4.464e-06, 4.115e-06, 4.105e-06, 3.901e-06,
                             4.087e-06, 4.170e-06])

individual_direct = np.array([-3.386e-06, 3.731e-06, 1.136e-05, 1.626e-05, 8.988e-06, 2.929e-06,
                              3.173e-06, 3.877e-07, 3.420e-07, 7.399e-07, 2.520e-07, -4.069e-07,
                              -4.844e-07, -2.739e-07, -3.448e-07, -3.370e-07, -3.573e-07, -4.376e-07,
                              -3.980e-07, -4.159e-07])

individual_indirect = np.array([9.86e-06, 1.96e-05, 1.90e-05, 1.51e-05, 1.56e-05, 1.12e-05,
                                1.14e-05, 7.22e-06, 7.63e-06, 8.36e-06, 7.92e-06, 6.61e-06,
                                5.21e-06, 5.00e-06, 4.81e-06, 4.45e-06, 4.46e-06, 4.34e-06,
                                4.49e-06, 4.59e-06])

individual_mediated = np.array([152.3, 84.0, 62.5, 48.2, 63.4, 79.3, 78.3, 94.9, 95.7, 91.9,
                                96.9, 106.6, 110.3, 105.8, 107.7, 108.2, 108.7, 111.2, 109.7, 110.0])

# Event-Level Aggregated Data (Step 8 - Enhanced Event-Level Analysis)
aggregate_total = np.array([5.720e-06, 1.753e-05, 2.132e-05, 1.724e-05, 1.154e-05, 5.865e-06,
                            4.951e-06, 2.142e-06, 1.848e-06, 1.835e-06, 1.633e-06, 1.276e-06,
                            9.934e-07, 1.070e-06, 1.008e-06, 9.294e-07, 1.001e-06, 9.928e-07,
                            1.048e-06, 1.071e-06])

aggregate_direct = np.array([-9.059e-07, 3.115e-06, 1.126e-05, 9.765e-06, 5.014e-06, 1.620e-06,
                             1.235e-06, 1.657e-07, 1.059e-07, 1.144e-07, 9.195e-08, -5.123e-08,
                             -4.919e-08, 7.183e-09, 1.768e-08, 2.443e-08, 8.158e-08, 1.321e-07,
                             1.707e-07, 2.115e-07])

aggregate_indirect = np.array([6.63e-06, 1.44e-05, 1.01e-05, 7.47e-06, 6.53e-06, 4.25e-06,
                               3.72e-06, 1.98e-06, 1.74e-06, 1.72e-06, 1.54e-06, 1.33e-06,
                               1.04e-06, 1.06e-06, 9.90e-07, 9.05e-07, 9.19e-07, 8.61e-07,
                               8.77e-07, 8.60e-07])

aggregate_mediated = np.array([115.8, 82.2, 47.2, 43.3, 56.6, 72.4, 75.0, 92.3, 94.3, 93.8,
                               94.4, 104.0, 105.0, 99.3, 98.2, 97.4, 91.8, 86.7, 83.7, 80.3])

# Amplification factors (aggregate vs 20km baseline)
amplification_factors = np.array([5.3, 16.4, 19.9, 16.1, 10.8, 5.5, 4.6, 2.0, 1.7, 1.7,
                                  1.5, 1.2, 0.9, 1.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0])


# ========================================
# VISUALIZATION 1: Individual vs Aggregate Comparison
# ========================================

def create_individual_vs_aggregate_comparison():
    """Create side-by-side comparison of individual well vs event-level aggregated effects"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Color scheme
    color_total = '#D32F2F'  # Red
    color_direct = '#1976D2'  # Blue
    color_indirect = '#388E3C'  # Green

    # Left panel: Individual Well-Level Effects
    ax1.plot(radius, individual_total * 1e6, 'o-', color=color_total, linewidth=2.5, markersize=6,
             label='Total Effect', zorder=3)
    ax1.plot(radius, individual_direct * 1e6, 's--', color=color_direct, linewidth=2, markersize=4,
             label='Direct Effect', zorder=2)
    ax1.plot(radius, individual_indirect * 1e6, '^-.', color=color_indirect, linewidth=2, markersize=4,
             label='Indirect Effect', zorder=2)

    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Distance from Injection Well (km)', fontweight='bold')
    ax1.set_ylabel('Effect Size (ΔM per BBL) × 10⁻⁶', fontweight='bold')
    ax1.set_title('A. Individual Well-Level Effects\n(Each well-event pair analyzed separately)',
                  fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, 21])
    ax1.set_xticks(np.arange(0, 21, 2))

    # Highlight peak for individual
    peak_idx_ind = np.argmax(individual_total)
    ax1.scatter(radius[peak_idx_ind], individual_total[peak_idx_ind] * 1e6,
                s=150, color=color_total, edgecolor='black', linewidth=2, zorder=4)
    ax1.annotate(f'Peak: {individual_total[peak_idx_ind] * 1e6:.1f}µ\n@ {radius[peak_idx_ind]}km',
                 xy=(radius[peak_idx_ind], individual_total[peak_idx_ind] * 1e6),
                 xytext=(radius[peak_idx_ind] + 3, individual_total[peak_idx_ind] * 1e6 - 5),
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='-|>', color='black', lw=1.5))

    # Right panel: Event-Level Aggregated Effects
    ax2.plot(radius, aggregate_total * 1e6, 'o-', color=color_total, linewidth=2.5, markersize=6,
             label='Total Effect', zorder=3)
    ax2.plot(radius, aggregate_direct * 1e6, 's--', color=color_direct, linewidth=2, markersize=4,
             label='Direct Effect', zorder=2)
    ax2.plot(radius, aggregate_indirect * 1e6, '^-.', color=color_indirect, linewidth=2, markersize=4,
             label='Indirect Effect', zorder=2)

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('Distance from Injection Well (km)', fontweight='bold')
    ax2.set_ylabel('Effect Size (ΔM per BBL) × 10⁻⁶', fontweight='bold')
    ax2.set_title('B. Event-Level Aggregated Effects\n(Multiple wells per event combined)',
                  fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, 21])
    ax2.set_xticks(np.arange(0, 21, 2))

    # Highlight peak for aggregate
    peak_idx_agg = np.argmax(aggregate_total)
    ax2.scatter(radius[peak_idx_agg], aggregate_total[peak_idx_agg] * 1e6,
                s=150, color=color_total, edgecolor='black', linewidth=2, zorder=4)
    ax2.annotate(f'Peak: {aggregate_total[peak_idx_agg] * 1e6:.1f}µ\n@ {radius[peak_idx_agg]}km',
                 xy=(radius[peak_idx_agg], aggregate_total[peak_idx_agg] * 1e6),
                 xytext=(radius[peak_idx_agg] + 3, aggregate_total[peak_idx_agg] * 1e6 - 3),
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='-|>', color='black', lw=1.5))

    plt.suptitle('Individual Well vs Event-Level Aggregated Causal Effects:\nComparison of Methodological Approaches',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


# ========================================
# VISUALIZATION 2: Total Effect Overlay Comparison
# ========================================

def create_total_effect_overlay():
    """Create overlay plot comparing total effects from both methodologies"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot both total effects on same axis
    line1 = ax.plot(radius, individual_total * 1e6, 'o-', color='#D32F2F', linewidth=3, markersize=7,
                    label='Individual Well-Level', alpha=0.8, zorder=3)
    line2 = ax.plot(radius, aggregate_total * 1e6, 's-', color='#1976D2', linewidth=3, markersize=6,
                    label='Event-Level Aggregated', alpha=0.8, zorder=3)

    # Fill between the curves to show differences
    ax.fill_between(radius, individual_total * 1e6, aggregate_total * 1e6,
                    alpha=0.2, color='purple', label='Methodological Difference')

    # Add background zones
    ax.axvspan(0, 5, alpha=0.1, color='red', label='Near-field Zone (0-5km)')
    ax.axvspan(5, 10, alpha=0.1, color='orange', label='Mid-field Zone (5-10km)')
    ax.axvspan(10, 21, alpha=0.1, color='green', label='Far-field Zone (>10km)')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    # Secondary y-axis for ratio
    ax2 = ax.twinx()
    ratio = individual_total / aggregate_total
    ax2.plot(radius, ratio, 'D--', color='#FF6F00', linewidth=2, markersize=4,
             alpha=0.7, label='Individual/Aggregate Ratio')
    ax2.axhline(y=1, color='#FF6F00', linestyle=':', alpha=0.5, linewidth=2)
    ax2.set_ylabel('Individual/Aggregate Ratio', fontweight='bold', color='#FF6F00')
    ax2.tick_params(axis='y', labelcolor='#FF6F00')

    ax.set_xlabel('Distance from Injection Well (km)', fontweight='bold')
    ax.set_ylabel('Total Effect Size (ΔM per BBL) × 10⁻⁶', fontweight='bold')
    ax.set_title(
        'Total Effect Comparison: Individual Well vs Event-Level Aggregation\nMethodological Impact on Causal Effect Estimation',
        fontweight='bold', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 21])
    ax.set_xticks(np.arange(0, 21, 2))

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    return fig


# ========================================
# VISUALIZATION 3: Mediation Mechanism Comparison
# ========================================

def create_mediation_comparison():
    """Compare pressure mediation patterns between individual and aggregate approaches"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Mediation percentages
    ax1.plot(radius, individual_mediated, 'o-', color='#D32F2F', linewidth=2.5, markersize=6,
             label='Individual Well-Level', zorder=3)
    ax1.plot(radius, aggregate_mediated, 's-', color='#1976D2', linewidth=2.5, markersize=6,
             label='Event-Level Aggregated', zorder=3)

    ax1.axhline(y=100, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Complete Mediation (100%)')
    ax1.fill_between(radius, individual_mediated, aggregate_mediated, alpha=0.2, color='purple')

    ax1.set_ylabel('Pressure Mediation (%)', fontweight='bold')
    ax1.set_title('A. Pressure Mediation Comparison', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    ax1.set_xlim([0, 21])
    ax1.set_xticks(np.arange(0, 21, 2))

    # Bottom panel: Direct vs Indirect effects comparison
    width = 0.35
    x_pos = np.arange(len(radius[::2]))  # Every other radius for readability

    ax2.bar(x_pos - width / 2, individual_direct[::2] * 1e6, width, label='Individual Direct',
            color='#1976D2', alpha=0.7)
    ax2.bar(x_pos + width / 2, aggregate_direct[::2] * 1e6, width, label='Aggregate Direct',
            color='#D32F2F', alpha=0.7)

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax2.set_xlabel('Distance from Injection Well (km)', fontweight='bold')
    ax2.set_ylabel('Direct Effect Size (ΔM per BBL) × 10⁻⁶', fontweight='bold')
    ax2.set_title('B. Direct Effect Comparison (Every 2km)', fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(radius[::2])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'Causal Mechanism Analysis: Individual vs Event-Level Approaches\nPressure Mediation and Direct Effect Patterns',
        fontsize=15, fontweight='bold', y=0.96)
    plt.tight_layout()
    return fig


# ========================================
# VISUALIZATION 4: Enhanced Spatial Impact Map
# ========================================

def create_enhanced_spatial_map():
    """Create spatial map showing both individual and aggregate risk zones"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Common parameters
    well_pos = {'x': 0, 'y': 0}
    key_radii = [3, 5, 7, 10, 15, 20]

    # Left panel: Individual well effects
    colors_individual = plt.cm.Reds(np.linspace(0.3, 1.0, len(key_radii)))
    for i, radius_val in enumerate(sorted(key_radii, reverse=True)):
        idx = np.where(radius == radius_val)[0][0]
        intensity = individual_total[idx] / individual_total.max()

        alpha_val = 0.15 + 0.6 * intensity
        ax1.add_patch(Circle((well_pos['x'], well_pos['y']), radius_val,
                             facecolor=colors_individual[len(key_radii) - 1 - i], alpha=alpha_val,
                             edgecolor='none', zorder=1))
        ax1.add_patch(Circle((well_pos['x'], well_pos['y']), radius_val, fill=False,
                             edgecolor='gray', linestyle='--', linewidth=0.8, alpha=0.6, zorder=2))

        if i < 3:  # Label only inner rings
            ax1.text(well_pos['x'] + radius_val * 0.707, well_pos['y'] + radius_val * 0.707,
                     f'{radius_val}km\n{individual_total[idx] * 1e6:.1f}µ',
                     fontsize=8, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7), zorder=3)

    ax1.scatter(well_pos['x'], well_pos['y'], s=150, c='black', marker='D',
                edgecolor='white', linewidth=2, zorder=4)
    ax1.set_xlim(-22, 22)
    ax1.set_ylim(-22, 22)
    ax1.set_aspect('equal')
    ax1.set_title('Individual Well-Level Risk Zones', fontweight='bold', fontsize=13)
    ax1.set_xlabel('Distance (km)', fontweight='bold')
    ax1.set_ylabel('Distance (km)', fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle=':')

    # Right panel: Event-level aggregated effects
    colors_aggregate = plt.cm.Blues(np.linspace(0.3, 1.0, len(key_radii)))
    for i, radius_val in enumerate(sorted(key_radii, reverse=True)):
        idx = np.where(radius == radius_val)[0][0]
        intensity = aggregate_total[idx] / aggregate_total.max()

        alpha_val = 0.15 + 0.6 * intensity
        ax2.add_patch(Circle((well_pos['x'], well_pos['y']), radius_val,
                             facecolor=colors_aggregate[len(key_radii) - 1 - i], alpha=alpha_val,
                             edgecolor='none', zorder=1))
        ax2.add_patch(Circle((well_pos['x'], well_pos['y']), radius_val, fill=False,
                             edgecolor='gray', linestyle='--', linewidth=0.8, alpha=0.6, zorder=2))

        if i < 3:  # Label only inner rings
            ax2.text(well_pos['x'] + radius_val * 0.707, well_pos['y'] + radius_val * 0.707,
                     f'{radius_val}km\n{aggregate_total[idx] * 1e6:.1f}µ',
                     fontsize=8, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7), zorder=3)

    ax2.scatter(well_pos['x'], well_pos['y'], s=150, c='black', marker='D',
                edgecolor='white', linewidth=2, zorder=4)
    ax2.set_xlim(-22, 22)
    ax2.set_ylim(-22, 22)
    ax2.set_aspect('equal')
    ax2.set_title('Event-Level Aggregated Risk Zones', fontweight='bold', fontsize=13)
    ax2.set_xlabel('Distance (km)', fontweight='bold')
    ax2.set_ylabel('Distance (km)', fontweight='bold')
    ax2.grid(True, alpha=0.2, linestyle=':')

    plt.suptitle(
        'Spatial Risk Zone Comparison: Individual vs Event-Level Analysis\nEffect Size Magnitude (µ = ×10⁻⁶ ΔM per BBL)',
        fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Add more space between suptitle and subplot titles
    return fig


# ========================================
# VISUALIZATION 5: Comprehensive Dashboard
# ========================================

def create_comprehensive_dashboard():
    """Create comprehensive dashboard comparing both methodologies"""

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        'Comprehensive Causal Analysis Dashboard: Individual vs Event-Level Approaches\nInjection-Induced Seismicity Analysis',
        fontsize=18, fontweight='bold', y=0.96)

    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1],
                          hspace=0.35, wspace=0.25)

    # Top row: Total effects comparison
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(radius, individual_total * 1e6, 'o-', color='#D32F2F', linewidth=2.5, markersize=6,
             label='Individual Well-Level', zorder=3)
    ax1.plot(radius, aggregate_total * 1e6, 's-', color='#1976D2', linewidth=2.5, markersize=6,
             label='Event-Level Aggregated', zorder=3)
    ax1.fill_between(radius, individual_total * 1e6, aggregate_total * 1e6,
                     alpha=0.2, color='purple', label='Methodological Difference')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Distance from Injection Well (km)', fontweight='bold')
    ax1.set_ylabel('Total Effect Size (ΔM per BBL) × 10⁻⁶', fontweight='bold')
    ax1.set_title('A. Total Effect Size Comparison (1-20km)', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    ax1.set_xlim([0, 21])
    ax1.set_xticks(np.arange(0, 21, 2))

    # Top right: Summary statistics
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    summary_text = """METHODOLOGICAL COMPARISON
INDIVIDUAL vs EVENT-LEVEL

Peak Effects:
• Individual: 31.4µ @ 4km
• Aggregate: 21.3µ @ 3km  
• Difference: ~32% lower aggregate

Key Insights:
• Individual analysis: Higher overall effects
• Aggregate analysis: Earlier peak, smoother profile
• Both show same mechanistic transitions

Near-field (1-5km):
• Individual: More variable effects
• Aggregate: Reduced noise from multiple wells

Far-field (>10km):  
• Both approaches converge
• Pressure-mediated effects dominate
• Similar mechanism patterns

Statistical Quality:
• Aggregate: Higher R² (0.42-0.55)
• Individual: More granular view
• Both: Robust causal identification
    """
    ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', alpha=0.8))
    ax2.set_title('Summary Comparison', fontweight='bold', loc='center', fontsize=13)

    # Middle row: Direct effects comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(radius, individual_direct * 1e6, 'o-', color='#D32F2F', linewidth=2, markersize=5,
             label='Individual Direct')
    ax3.plot(radius, aggregate_direct * 1e6, 's-', color='#1976D2', linewidth=2, markersize=5,
             label='Aggregate Direct')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Distance (km)', fontweight='bold')
    ax3.set_ylabel('Direct Effect × 10⁻⁶', fontweight='bold')
    ax3.set_title('B. Direct Effects', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.set_xlim([0, 21])

    # Middle center: Indirect effects comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(radius, individual_indirect * 1e6, 'o-', color='#D32F2F', linewidth=2, markersize=5,
             label='Individual Indirect')
    ax4.plot(radius, aggregate_indirect * 1e6, 's-', color='#1976D2', linewidth=2, markersize=5,
             label='Aggregate Indirect')
    ax4.set_xlabel('Distance (km)', fontweight='bold')
    ax4.set_ylabel('Indirect Effect × 10⁻⁶', fontweight='bold')
    ax4.set_title('C. Indirect Effects', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    ax4.set_xlim([0, 21])

    # Middle right: Mediation comparison
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(radius, individual_mediated, 'o-', color='#D32F2F', linewidth=2, markersize=5,
             label='Individual % Med.')
    ax5.plot(radius, aggregate_mediated, 's-', color='#1976D2', linewidth=2, markersize=5,
             label='Aggregate % Med.')
    ax5.axhline(y=100, color='black', linestyle='--', alpha=0.5, linewidth=1, label='100% Mediation')
    ax5.set_xlabel('Distance (km)', fontweight='bold')
    ax5.set_ylabel('Pressure Mediation (%)', fontweight='bold')
    ax5.set_title('D. Mediation Patterns', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    ax5.set_xlim([0, 21])

    # Bottom row: Amplification and ratio analysis
    ax6 = fig.add_subplot(gs[2, :])

    # Create twin axis for ratio
    ax6_twin = ax6.twinx()

    # Plot amplification factors (bars)
    bars = ax6.bar(radius, amplification_factors, alpha=0.6, color='gray',
                   label='Amplification Factor (vs 20km)')

    # Highlight significant amplifications
    for i, (r, amp) in enumerate(zip(radius, amplification_factors)):
        if amp > 5:
            bars[i].set_color('#8B0000')
            bars[i].set_alpha(0.8)

    # Plot ratio on twin axis
    ratio = individual_total / aggregate_total
    ax6_twin.plot(radius, ratio, 'D-', color='#FF6F00', linewidth=2.5, markersize=5,
                  label='Individual/Aggregate Ratio')
    ax6_twin.axhline(y=1, color='#FF6F00', linestyle=':', alpha=0.5, linewidth=2)

    ax6.set_xlabel('Distance from Injection Well (km)', fontweight='bold')
    ax6.set_ylabel('Amplification Factor', fontweight='bold')
    ax6_twin.set_ylabel('Effect Size Ratio', fontweight='bold', color='#FF6F00')
    ax6_twin.tick_params(axis='y', labelcolor='#FF6F00')
    ax6.set_title('E. Amplification Factors & Methodological Ratios', fontweight='bold', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xlim([0.5, 20.5])
    ax6.set_xticks(radius[::2])  # Every other tick for readability

    # Combined legend for bottom plot
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


# ========================================
# Generate All Visualizations
# ========================================

if __name__ == "__main__":
    print("Generating Individual vs Aggregate Well Effects Visualizations...")

    print("\n1. Creating Individual vs Aggregate Comparison...")
    fig1 = create_individual_vs_aggregate_comparison()
    plt.savefig('individual_vs_aggregate_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('individual_vs_aggregate_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig1)

    print("\n2. Creating Total Effect Overlay...")
    fig2 = create_total_effect_overlay()
    plt.savefig('total_effect_overlay_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('total_effect_overlay_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig2)

    print("\n3. Creating Mediation Mechanism Comparison...")
    fig3 = create_mediation_comparison()
    plt.savefig('mediation_mechanism_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('mediation_mechanism_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig3)

    print("\n4. Creating Enhanced Spatial Map...")
    fig4 = create_enhanced_spatial_map()
    plt.savefig('enhanced_spatial_risk_map.png', dpi=300, bbox_inches='tight')
    plt.savefig('enhanced_spatial_risk_map.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig4)

    print("\n5. Creating Comprehensive Dashboard...")
    fig5 = create_comprehensive_dashboard()
    plt.savefig('comprehensive_individual_aggregate_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig('comprehensive_individual_aggregate_dashboard.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig5)

    print("\nAll individual vs aggregate well effect visualizations generated successfully!")
    print("\nKey Insights:")
    print("• Individual well analysis shows higher peak effects (31.4µ at 4km)")
    print("• Event-level aggregation shows earlier peak (21.3µ at 3km) with smoother profile")
    print("• Both methods show same mechanistic transitions (direct → pressure-mediated)")
    print("• Aggregation provides better signal-to-noise ratio and higher predictive power")
    print("• Individual analysis provides more granular view of well-specific effects")