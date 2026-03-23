import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_GROUP_STYLES = [
    {'color': '#105A7F', 'marker': 'o', 'alpha': 1.0},
    {'color': '#229FBF', 'marker': 's', 'alpha': 1.0},
    {'color': '#FFC533', 'marker': '^', 'alpha': 1.0},
    {'color': '#f21c1c', 'marker': 'D', 'alpha': 1.0},
]


def compute_theoretical_joint(shared_sample, mu, delta, sigma_rhs, t, alpha_range):
    """
    Compute the theoretical joint constraint satisfaction probability using the shared sample.

    For each alpha, counts the fraction of shared_sample scenarios where ALL cumulative
    demand constraints are simultaneously within the individual CC margin z_alpha.
    This gives the true joint satisfaction probability implied by individual chance
    constraints at level alpha under the estimated Gaussian model.

    The condition for both upper and lower bound constraints to be satisfied is
    |Z_{j,t}| <= z_alpha for all tanks j and time steps t, where Z is the
    standardized cumulative demand residual.

    Args:
        shared_sample : (p, n) array — scenarios drawn from N(mu, Sigma)
        mu            : (p,) array  — nominal values [prices; demands]
        delta         : (m, n_tanks*t) array — cumulative demand mapping matrix
        sigma_rhs     : (m,) array — std of cumulative demand per constraint
        t             : int — horizon length (first t rows of sample are prices)
        alpha_range   : array — confidence levels at which to evaluate joint probability

    Returns:
        (len(alpha_range),) array of joint satisfaction probabilities
    """
    # Zero-mean demand residuals: shape (n_tanks*t, n_samples)
    demand_residuals = shared_sample[t:, :] - mu[t:, None]

    # Cumulative residuals per constraint: shape (m, n_samples)
    cumulative_residuals = delta @ demand_residuals

    # Standardise: Z_{j,t} ~ N(0,1) marginally if the Gaussian model is correct
    Z = cumulative_residuals / sigma_rhs[:, None]  # shape (m, n_samples)

    # Joint feasibility: both upper (Z <= z_alpha) and lower (-Z <= z_alpha) bounds
    # collapse to |Z| <= z_alpha because upper and lower share the same delta
    result = []
    for alpha in alpha_range:
        z_alpha = norm.ppf(alpha)
        joint_ok = np.all(np.abs(Z) <= z_alpha, axis=0)  # shape (n_samples,)
        result.append(joint_ok.mean())

    return np.array(result)


def plot_por_results(df, group_styles=None, sp_theoretical_data=None, omega_to_alpha='gaussian', stat_analyze=True):
    plot_configs = [
        ('avg_cost', 'reliability', 'Average Cost (€)', 'Constraints Satisfaction (%)'),
        ('avg_cost', 'violated_vol', 'Average Cost (€)', 'Violated Volume ($m^3$)'),
        ('por', 'reliability', 'Price of Robustness', 'Constraints Satisfaction (%)'),
        ('objective', 'reliability', 'Solver Objective (€)', 'Constraints Satisfaction (%)'),
    ]

    # Build styles for each group
    groups = df['group'].unique()
    styles = {}
    for i, group in enumerate(groups):
        if group_styles and group in group_styles:
            styles[group] = group_styles[group]
        else:
            default = DEFAULT_GROUP_STYLES[i % len(DEFAULT_GROUP_STYLES)]
            styles[group] = default.copy()

    figures = []
    for x_col, y_col, xlabel, ylabel in plot_configs:
        if x_col not in df.columns or y_col not in df.columns:
            continue

        fig, ax = plt.subplots()

        for group in groups:
            group_df = df[df['group'] == group].sort_values(by=x_col)
            style = styles[group]

            x_vals = group_df[x_col].values
            y_vals = group_df[y_col].values
            if y_col == 'reliability':
                y_vals = y_vals * 100

            ax.plot(x_vals, y_vals, color=style['color'], marker=style['marker'], alpha=style['alpha'],
                    label=group, linewidth=2, markersize=8, mfc="white")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        figures.append(fig)

    if stat_analyze:
        fig_theory = _plot_theoretical_guarantees(df, styles, sp_theoretical_data=sp_theoretical_data)
        if fig_theory is not None:
            figures.append(fig_theory)

        fig_combined = _plot_combined_guarantees(df, styles, sp_theoretical_data=sp_theoretical_data,
                                                 omega_to_alpha=omega_to_alpha)
        if fig_combined is not None:
            figures.append(fig_combined)

    return figures


def _plot_theoretical_guarantees(df, styles, sp_theoretical_data=None):
    """
    Plot empirical reliability vs theoretical guarantees.

    Creates two side-by-side subplots:
    - Left: RO methods with omega on x-axis
    - Right: SP methods with alpha on x-axis
    Y-axis shows reliability (constraint satisfaction rate).

    Theoretical bounds:
    - RO (ellipsoid): P(violation) <= exp(-omega^2/2), so reliability >= 1 - exp(-omega^2/2)
    - SP (chance constraints): reliability >= alpha
    """
    ro_mask = df['omega'].notna() if 'omega' in df.columns else pd.Series(False, index=df.index)
    sp_mask = df['alpha'].notna() if 'alpha' in df.columns else pd.Series(False, index=df.index)

    has_ro = ro_mask.any()
    has_sp = sp_mask.any()

    if not has_ro and not has_sp:
        return None

    n_plots = int(has_ro) + int(has_sp)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    if has_ro:
        ax = axes[ax_idx]
        ax_idx += 1

        for group in df[ro_mask]['group'].unique():
            group_df = df[(df['group'] == group) & ro_mask].sort_values(by='omega')
            style = styles.get(group, {'color': 'blue', 'marker': 'o', 'alpha': 1.0})
            ax.plot(group_df['omega'].values, group_df['reliability'].values * 100,
                    color=style['color'], marker=style['marker'], alpha=style['alpha'],
                    label=group, linewidth=2, markersize=8, mfc="white")

        omega_range = np.linspace(0, df[ro_mask]['omega'].max() * 1.1, 100)
        theoretical_ro = (1 - np.exp(-omega_range ** 2 / 2)) * 100
        ax.plot(omega_range, theoretical_ro, color='grey', linewidth=1.5,
                label='Bound: $1 - e^{-\\Omega^2/2}$')

        ax.set_xlabel('$\\Omega$ (Ellipsoid Radius)')
        ax.set_ylabel('Constraint Satisfaction (%)')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_title('Robust Optimization')

    if has_sp:
        ax = axes[ax_idx]

        for group in df[sp_mask]['group'].unique():
            group_df = df[(df['group'] == group) & sp_mask].copy()
            group_df['alpha_'] = group_df['alpha'].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) else x
            )
            group_df = group_df.sort_values(by='alpha_')
            style = styles.get(group, {'color': 'orange', 'marker': 's', 'alpha': 1.0})
            ax.plot(group_df['alpha_'].values, group_df['reliability'].values * 100,
                    color=style['color'], marker=style['marker'], alpha=style['alpha'],
                    label=group, linewidth=2, markersize=8, mfc="white")

        alpha_min = df[sp_mask]['alpha'].apply(lambda x: x[1] if isinstance(x, (list, tuple)) else x).min()
        alpha_max = df[sp_mask]['alpha'].apply(lambda x: x[1] if isinstance(x, (list, tuple)) else x).max()
        alpha_range = np.linspace(alpha_min * 0.95, min(alpha_max * 1.05, 1.0), 100)

        ax.plot(alpha_range, alpha_range * 100, color='grey', linewidth=1.5, linestyle='--',
                label='Individual bound: $\\alpha$')

        if sp_theoretical_data is not None:
            joint_prob = compute_theoretical_joint(
                shared_sample=sp_theoretical_data['shared_sample'],
                mu=sp_theoretical_data['mu'],
                delta=sp_theoretical_data['delta'],
                sigma_rhs=sp_theoretical_data['sigma_rhs'],
                t=sp_theoretical_data['t'],
                alpha_range=alpha_range,
            )
            ax.plot(alpha_range, joint_prob * 100, color='grey', linewidth=1.5, linestyle=':',
                    label='Joint bound (Gaussian)')

        ax.set_xlabel('$\\alpha$ (CC Confidence Level)')
        ax.set_ylabel('Constraint Satisfaction (%)')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_title('Stochastic Programming')

    fig.tight_layout()
    return fig


def _plot_combined_guarantees(df, styles, sp_theoretical_data=None, omega_to_alpha='gaussian'):
    """
    Combined plot: RO and SP-CC reliability on the same axes.

    X-axis (bottom): Ω (ellipsoid radius for RO; equivalent Ω for SP via inverse mapping)
    X-axis (top):    α (corresponding confidence level via the chosen mapping)
    Y-axis:          empirical constraint satisfaction (%)

    omega_to_alpha choices:
        'gaussian'    — α = Φ(Ω)  (exact under Gaussian uncertainty)
        'subgaussian' — α = 1 − exp(−Ω²/2)  (conservative sub-Gaussian bound)
    """
    ro_mask = df['omega'].notna() if 'omega' in df.columns else pd.Series(False, index=df.index)
    sp_mask = df['alpha'].notna() if 'alpha' in df.columns else pd.Series(False, index=df.index)

    if not ro_mask.any() or not sp_mask.any():
        return None

    if omega_to_alpha == 'gaussian':
        def map_fn(omega):
            return norm.cdf(omega)
        def inv_map_fn(alpha):
            return norm.ppf(np.clip(alpha, 1e-9, 1 - 1e-9))
        map_label = r'Gaussian: $\alpha = \Phi(\Omega)$'
    else:  # 'subgaussian'
        def map_fn(omega):
            return 1.0 - np.exp(-np.asarray(omega) ** 2 / 2)
        def inv_map_fn(alpha):
            return np.sqrt(-2.0 * np.log(np.clip(1.0 - np.asarray(alpha), 1e-10, 1.0)))
        map_label = r'Sub-Gaussian: $\alpha = 1 - e^{-\Omega^2/2}$'

    fig, ax = plt.subplots()
    all_omegas = []

    for group in df[ro_mask]['group'].unique():
        group_df = df[(df['group'] == group) & ro_mask].sort_values('omega')
        style = styles.get(group, {'color': 'blue', 'marker': 'o', 'alpha': 1.0})
        omega_vals = group_df['omega'].values
        all_omegas.extend(omega_vals)
        ax.plot(omega_vals, group_df['reliability'].values * 100,
                color=style['color'], marker=style['marker'], alpha=style['alpha'],
                label=group, linewidth=2, markersize=8, mfc='white')

    for group in df[sp_mask]['group'].unique():
        group_df = df[(df['group'] == group) & sp_mask].copy()
        group_df['alpha_'] = group_df['alpha'].apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) else x
        )
        group_df = group_df.sort_values('alpha_')
        style = styles.get(group, {'color': 'orange', 'marker': 's', 'alpha': 1.0})
        omega_equiv = inv_map_fn(group_df['alpha_'].values)
        all_omegas.extend(omega_equiv)
        ax.plot(omega_equiv, group_df['reliability'].values * 100,
                color=style['color'], marker=style['marker'], alpha=style['alpha'],
                label=group, linewidth=2, markersize=8, mfc='white')

    omega_min = max(0.0, min(all_omegas) * 0.9)
    omega_max = max(all_omegas) * 1.1
    omega_range = np.linspace(omega_min, omega_max, 200)

    ax.plot(omega_range, map_fn(omega_range) * 100, color='grey', linewidth=1.5,
            linestyle='--', label='Theoretical bound')

    # Secondary x-axis: place ticks at actual alpha values present in the data
    actual_alphas = []
    for group in df[ro_mask]['group'].unique():
        actual_alphas.extend(map_fn(df[(df['group'] == group) & ro_mask]['omega'].values).tolist())
    for group in df[sp_mask]['group'].unique():
        actual_alphas.extend(
            df[(df['group'] == group) & sp_mask]['alpha']
            .apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x).tolist()
        )
    seen_labels, deduped = set(), []
    for a in sorted({round(a, 4) for a in actual_alphas}):
        lbl = f'{a:.2f}'
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            deduped.append(a)

    secax = ax.secondary_xaxis('top', functions=(map_fn, inv_map_fn))
    secax.set_ticks(deduped)
    secax.set_xticklabels([f'{a:.2f}' for a in deduped], rotation=90, ha='center', fontsize=9)
    secax.set_xlabel('$\\alpha$ (Equivalent CC Confidence Level)')

    ax.set_xlabel('$\\Omega$ (Ellipsoid Radius / Equivalent RO Radius)')
    ax.set_ylabel('Constraint Satisfaction (%)')
    ax.set_xlim(omega_min, omega_max)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.text(0.01, 0.02, f'Mapping: {map_label}', transform=ax.transAxes,
            fontsize=8, va='bottom', ha='left', color='grey')

    fig.tight_layout()
    return fig
