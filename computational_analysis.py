"""Build the computational-analysis table (problem dimensions + solve performance) for the paper.

For each case study output directory, merges the static structural stats in
problem_stats.csv (one row per method: var/constraint counts, fixed for a given
method/horizon/sample-size config) with per-step solve stats aggregated from
trajectories.csv (solver_time, solver_mip_gap, bb_node_count, opt_status).

opt_status == 'optimal' is CVXPY's raw Gurobi termination status, so it already
distinguishes a proven-optimal step from one that hit a time/gap limit (would be
'user_limit') or fell back to the last feasible schedule (would be 'fallback');
that is what pct_proven_optimal counts, rather than thresholding solver_mip_gap
against an arbitrary tolerance. solver_mip_gap and bb_node_count are NaN for
non-MILP formulations (see BaseOptModel.solve), so gap/node stats are computed
only over rows where a MIP gap was actually reported.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from graphs import (derive_group, GROUP_STYLES, build_label_colors, to_latex_label, LEG_FONT_SIZE, MPCGraphs,
                    transpose_legend, build_legend_order)

PROBLEM_STATS_COLS = [
    'n_vars_continuous', 'n_vars_binary', 'n_constrs_linear', 'n_constrs_soc', 'n_nonzeros',
]

PROBLEM_STATS_KEYS = ['label', 'group', 'opt_method']
TRAJECTORY_KEYS = ['label', 'group', 'model']

# Display name for each canonical group (see graphs.derive_group) in the LaTeX table.
GROUP_DISPLAY_NAMES = {
    'PTO':          'PTO',
    'RO-Ellipsoid': 'RO-Ell',
    'RO-BOU':       'RO-BoU',
    'CCO':          'CCO',
    'CCO-relaxed':  'CC (Relaxed)',
    'CCO-CVaR':     'CCO-CVaR',
}
# Row order for the LaTeX table -- same family order as graphs.GROUP_STYLES.
GROUP_ORDER = list(GROUP_STYLES.keys())

_LATEX_SPECIAL_CHARS = {
    '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
    '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
}

# Shared exponent factored into the MIP Gap column header (see _fmt_scaled) rather than
# repeated in every cell -- raw gaps sit around 1e-6..1e-4, so 1e-6 keeps every cell a
# compact 1-3 digit number instead of a full '$m \times 10^{n}$' string per row.
MIP_GAP_EXP = -6


def _escape_latex(s):
    """Escape LaTeX special chars in free-text cells (case_study labels, group names).

    Needed because to_latex_table calls df.to_latex(escape=False) so that the
    pre-formatted numeric cells can carry raw LaTeX ($\\pm$, \\%, $\\infty$); that
    means any un-escaped free text -- e.g. a case_study label like 'Price & Ostfeld',
    whose '&' is a literal LaTeX column separator -- silently corrupts the table
    layout instead of raising an error.
    """
    return ''.join(_LATEX_SPECIAL_CHARS.get(c, c) for c in str(s))


def _multiline_header(text, sep='|'):
    """Wrap a header into an N-line \\shortstack (built into LaTeX, no extra package
    needed), breaking exactly where the caller places a `sep` character in the header
    text -- e.g. 'Mean MIP Gap|($\\times 10^{-6}$)' renders as 2 lines, 'A|B|C' as 3.
    Headers with no `sep` are left as a single line.

    Headers like 'Cont. Vars' or 'Solver Time (s)' are wider than the short numeric
    columns they sit above, so they -- not the data -- are usually what pushes a
    computational-analysis table past the page width. This puts the line-break choice
    in the header text itself (_COLUMN_SPECS) rather than an automatic first-space split,
    since the best break point isn't always the first space (e.g. 'Mean MIP Gap' should
    stay on one line, with the '($\\times 10^{-6}$)' part wrapping to a second).
    """
    if sep not in text:
        return text
    body = ' \\\\ '.join(text.split(sep))
    return rf'\shortstack{{{body}}}'


def _fmt_int(x):
    return '--' if pd.isna(x) else f"{x:,.0f}"


def _fmt_pct(x, decimals=2):
    return '--' if pd.isna(x) else f"{x:.{decimals}f}"


def _fmt_mean_std(row, base, scale=1.0, decimals=3):
    """Merge a '{base}_mean'/'{base}_std' pair into one 'mean $\\pm$ std' LaTeX cell."""
    mean = row.get(f'{base}_mean')
    if pd.isna(mean):
        return '--'
    mean_str = r'$\infty$' if np.isinf(mean) else f"{mean * scale:.{decimals}f}"

    std = row.get(f'{base}_std')
    if pd.isna(std) or np.isinf(std):
        return mean_str
    return f"{mean_str}$\\pm${std * scale:.{decimals}f}"


def _fmt_mean(row, base, scale=1.0, decimals=1):
    """Format just the '{base}_mean' field (no std) as a plain number."""
    mean = row.get(f'{base}_mean')
    if pd.isna(mean):
        return '--'
    if np.isinf(mean):
        return r'$\infty$'
    return f"{mean * scale:,.{decimals}f}"


def _fmt_scaled(x, exponent, decimals=1):
    """Format x / 10**exponent as a plain number, for use under a header that states
    the shared '$\\times 10^{exponent}$' factor once instead of repeating it per cell.
    """
    if pd.isna(x):
        return '--'
    if np.isinf(x):
        return r'$\infty$'
    return f"{x / 10 ** exponent:.{decimals}f}"


# Each logical column -> (LaTeX header, formatter(row) -> cell string).
_COLUMN_SPECS = {
    'vars':                ('Total|Vars.', lambda r: _fmt_int(r['n_vars_continuous'] + r['n_vars_binary'])),
    'constrs':             ('Total|Constr.', lambda r: _fmt_int(r['n_constrs_linear'] + r['n_constrs_soc'])),
    'density_pct':        ('NZ|Density|(\\%)', lambda r: _fmt_pct(r['density_pct'], decimals=3)),
    'bb_nodes':           ('Mean|B\\&B|Nodes', lambda r: _fmt_mean(r, 'bb_nodes', decimals=0)),
    # 'mip_gap':            (rf'Mean|MIP|Gap|($\times 10^{{{MIP_GAP_EXP}}}$)',
    #                        lambda r: _fmt_scaled(r.get('mip_gap_mean'), MIP_GAP_EXP)),
    'mip_gap':            ('Mean|MIP|Gap|(\\%)', lambda r: _fmt_mean(r, 'mip_gap', scale=100, decimals=4)),
    'pct_proven_optimal': ('Optimal|(\\%)',    lambda r: _fmt_pct(r['pct_proven_optimal'], decimals=1)),
    'step_total_time':    ('Step|Time|(s)', lambda r: _fmt_mean_std(r, 'step_total_time', decimals=1)),
    # 'n_nonzeros':         ('NZ',         lambda r: _fmt_int(r['n_nonzeros'])),
    # 'solver_time':        ('Solver Time (s)',  lambda r: _fmt_mean_std(r, 'solver_time')),

}

DEFAULT_LATEX_COLUMNS = list(_COLUMN_SPECS.keys())


def _density_pct(row):
    n_vars = row['n_vars_continuous'] + row['n_vars_binary']
    n_constrs = row['n_constrs_linear'] + row['n_constrs_soc']
    return 100 * row['n_nonzeros'] / (n_vars * n_constrs)


def _aggregate_solve_stats(df):
    has_gap = df['solver_mip_gap'].notna()
    gap = df.loc[has_gap, 'solver_mip_gap']
    nodes = df.loc[has_gap, 'bb_node_count']
    return pd.Series({
        'solver_time_mean': df['solver_time'].mean(),
        'solver_time_std': df['solver_time'].std(),
        'step_total_time_mean': df['step_total_time'].mean(),
        'step_total_time_std': df['step_total_time'].std(),
        'mip_gap_mean': gap.mean() if has_gap.any() else float('nan'),
        'mip_gap_std': gap.std() if has_gap.any() else float('nan'),
        'bb_nodes_mean': nodes.mean() if has_gap.any() else float('nan'),
        'bb_nodes_std': nodes.std() if has_gap.any() else float('nan'),
        'pct_proven_optimal': 100 * (df['opt_status'] == 'optimal').mean(),
    })


def _filter_infeasible(trajectories, case_study, max_infeasible):
    """Drop rows for any label whose fallback-step count exceeds max_infeasible, within
    a single case study's trajectories. Infeasible steps are approximated the same way
    main.py's _compute_metrics_from_df does -- opt_status == 'fallback' -- since a solver
    fallback is what a step recorded as infeasible actually looks like in trajectories.csv.
    A contaminated sweep point (e.g. RO_omega=3.0 in the Price & Ostfeld case study, which
    had 176 fallback steps) otherwise dominates its group with steps that never solved to
    begin with. Dropped labels are printed so the exclusion is visible rather than silent.
    """
    infeasible_counts = trajectories[trajectories['opt_status'] == 'fallback'].groupby('label').size()
    to_drop = infeasible_counts[infeasible_counts > max_infeasible]
    for label, count in to_drop.items():
        print(f"dropping {case_study} / {label} ({count} infeasible steps > {max_infeasible})")
    return trajectories[~trajectories['label'].isin(to_drop.index)].reset_index(drop=True)


def load_case_study(output_dir, case_study, max_infeasible=None):
    """Load problem_stats.csv + trajectories.csv from one output_dir and merge into one table.

    Args:
        output_dir  : path to an experiment output directory (contains problem_stats.csv
                      and trajectories.csv, as written by main.py's mpc_sweep runner)
        case_study  : label identifying this case study in the resulting table
        max_infeasible: if given, drop any label whose fallback-step count exceeds this
                      threshold (see _filter_infeasible) before aggregating -- the label's
                      row is dropped from the table entirely, not just NaN'd out.

    Returns:
        DataFrame, one row per method (label), with case_study prepended
    """
    output_dir = Path(output_dir)
    problem_stats = pd.read_csv(output_dir / 'problem_stats.csv')
    trajectories = pd.read_csv(output_dir / 'trajectories.csv')

    if max_infeasible is not None:
        trajectories = _filter_infeasible(trajectories, case_study, max_infeasible)
        problem_stats = problem_stats[problem_stats['label'].isin(trajectories['label'])]

    solve_stats = (
        trajectories.groupby(TRAJECTORY_KEYS)
        .apply(_aggregate_solve_stats, include_groups=False)
        .reset_index()
        .rename(columns={'model': 'opt_method'})
    )

    table = problem_stats.merge(solve_stats, on=PROBLEM_STATS_KEYS, how='left')
    table['density_pct'] = table.apply(_density_pct, axis=1)

    # Overwrite the raw config-side 'group' (main.py's method.get('group', label), used only
    # as the problem_stats/trajectories merge key above) with the canonical style-group derived
    # purely from each label's own structure -- same derive_group() graphs.py uses for plot
    # coloring. This is what actually splits CC-SP into CCO vs. CCO-CVaR (lambda_cvar > 0),
    # since computationally a CVaR term changes the problem (extra vars/constraints per sample)
    # regardless of the specific lambda weight, so all lambda_cvar > 0 runs share one group.
    table['group'] = table['label'].apply(derive_group)

    table.insert(0, 'case_study', case_study)
    return table


def build_computational_table(output_dirs, group_by=None, max_infeasible=None):
    """Build the full computational-analysis table across multiple case studies.

    Args:
        output_dirs : dict mapping case_study label -> output_dir path
        max_infeasible: if given, drop labels with too many fallback steps before
                      aggregating (see load_case_study / _filter_infeasible)

    Returns:
        DataFrame with one row per (case_study, method)
    """
    tables = [
        load_case_study(path, case_study, max_infeasible=max_infeasible)
        for case_study, path in output_dirs.items()
    ]
    if group_by is not None:
        tables = [table.groupby(group_by, as_index=False).mean(numeric_only=True) for table in tables]
    return pd.concat(tables, ignore_index=True)


def to_latex_table(table, columns=DEFAULT_LATEX_COLUMNS, output_path=None, resizebox=True):
    """
    Format a computational-analysis table (as built by build_computational_table, one row
    per case_study/group) into an Overleaf-ready LaTeX table.

    mean/std pairs (solver_time, step_total_time) are merged into a single 'mean $\\pm$ std'
    cell; mip_gap and bb_nodes report mean only (std adds little signal at MIP-gap's near-
    zero scale, and repeating it for every column bloats the table); mip_gap is shown as a
    plain number scaled by a shared $\\times 10^{-6}$ factor stated once in the column
    header (MIP_GAP_EXP), rather than repeating '$\\times 10^{-6}$' in every cell -- raw
    gaps sit around 1e-6..1e-4, so fixed-decimal percent formatting would round them all
    to the same value. Continuous/binary var counts and linear/SOC constraint counts are
    each summed into a single 'Vars'/'Constr' total column, since the split is a smaller
    detail than the overall problem size once the table also carries density/time/gap;
    density_pct/pct_proven_optimal as percentages. Rows are
    ordered by case_study, then by the canonical group order in GROUP_ORDER (same family
    order as graphs.GROUP_STYLES). Repeated case_study values are blanked after the first
    row of each case study for readability. Column headers wrap via _multiline_header,
    which breaks at each '|' character placed in the header text in _COLUMN_SPECS (so the
    break point is chosen explicitly per header, not auto-detected) to keep the table
    narrow (headers are usually wider than the short numeric columns they sit above).

    Requires \\usepackage{booktabs} in the LaTeX preamble (pandas.to_latex's default
    \\toprule/\\midrule/\\bottomrule style), plus \\usepackage{graphicx} if resizebox=True.
    resizebox scales the whole tabular down to \\textwidth as a final safety net on top
    of the merged columns/scaled units/multiline headers above, since those reduce the
    table's natural width but don't guarantee it fits any specific page width.

    Args:
        table:   DataFrame from build_computational_table
        columns: which logical columns to include and in what order; defaults to
                 DEFAULT_LATEX_COLUMNS (all columns in _COLUMN_SPECS). Pass a subset
                 to trim the table down for a paper, e.g.
                 ['n_vars_continuous', 'n_constrs_linear', 'density_pct', 'solver_time'].
        output_path: if given, write the .tex snippet to this path.
        resizebox: wrap the tabular in \\resizebox{\\textwidth}{!}{...} so it always
                 fits the page width regardless of how many columns are selected.

    Returns:
        The LaTeX table source as a string (a bare tabular environment, optionally
        wrapped in \\resizebox -- wrap it in your own \\begin{table}...\\end{table}
        with \\caption/\\label in Overleaf).
    """
    unknown = [c for c in columns if c not in _COLUMN_SPECS]
    if unknown:
        raise ValueError(f"Unknown column(s): {unknown}. Available: {list(_COLUMN_SPECS)}")

    df = table.copy()
    df['group'] = pd.Categorical(df['group'], categories=GROUP_ORDER, ordered=True)
    df = df.sort_values(['case_study', 'group'])

    rows = []
    for _, row in df.iterrows():
        group_name = GROUP_DISPLAY_NAMES.get(row['group'], row['group'])
        out_row = {'Case': _escape_latex(row['case_study']), 'Method': _escape_latex(group_name)}
        for col in columns:
            header, fmt = _COLUMN_SPECS[col]
            out_row[header] = fmt(row)
        rows.append(out_row)

    display_df = pd.DataFrame(rows)
    display_df['Case'] = display_df['Case'].mask(display_df['Case'].duplicated(), '')
    display_df = display_df.rename(columns=_multiline_header)

    column_format = 'll' + 'r' * len(columns)
    latex = display_df.to_latex(index=False, escape=False, column_format=column_format)
    if resizebox:
        latex = f"\\resizebox{{\\textwidth}}{{!}}{{%\n{latex.rstrip()}\n}}\n"
    if output_path:
        Path(output_path).write_text(latex)
    return latex


def load_trajectories(output_dirs, max_infeasible=None):
    """Load raw per-step trajectories across case studies, with canonical group and
    case_study attached, for distribution plots (boxplots) that need the individual
    step values rather than the mean/std pairs in build_computational_table.

    Args:
        output_dirs: dict mapping case_study label -> output_dir path (same as
            build_computational_table)
        max_infeasible: if given, drop any (case_study, label) whose infeasible-step
            count exceeds this threshold before returning. Infeasible steps are
            approximated the same way main.py's _compute_metrics_from_df does --
            opt_status == 'fallback' -- since a solver fallback is what a step
            recorded as infeasible actually looks like in trajectories.csv. A
            contaminated sweep point (e.g. RO_omega=3.0 in the Price & Ostfeld case
            study, which had 179 fallback steps) otherwise dominates its group's
            boxplot with steps that never solved to begin with. Dropped labels are
            printed so the exclusion is visible rather than silent.

    Returns:
        DataFrame, all case studies' trajectories.csv concatenated, plus 'case_study'
        and 'group' columns.
    """
    dfs = []
    for case_study, output_dir in output_dirs.items():
        output_dir = Path(output_dir)
        trajectories = pd.read_csv(output_dir / 'trajectories.csv')
        if max_infeasible is not None:
            trajectories = _filter_infeasible(trajectories, case_study, max_infeasible)
        trajectories['group'] = trajectories['label'].apply(derive_group)
        trajectories.insert(0, 'case_study', case_study)
        dfs.append(trajectories)
    return pd.concat(dfs, ignore_index=True)


_N_BINARY_STEPS_RE = re.compile(r'n_binary_steps=(\d+)')
_STRIP_N_BINARY_STEPS_RE = re.compile(r'[_|]n_binary_steps=\d+')


def _n_binary_steps(label):
    """Extract the swept n_binary_steps value out of a '...|n_binary_steps=N' label, or
    None for a label outside the sweep (e.g. a baseline run left un-swept)."""
    m = _N_BINARY_STEPS_RE.search(label)
    return int(m.group(1)) if m else None


def _strip_n_binary_steps(label):
    """Drop the '|n_binary_steps=N' (or 'LP_n_binary_steps=N') suffix, leaving the base
    parameter-configuration label (e.g. 'RO_omega=2.5', 'LP') -- the same string that
    identifies this config's color/legend entry in every other plot (plot_por,
    plot_cost_cvar, ...), so a given omega/alpha/gamma config keeps one consistent
    color across n_binary_steps values instead of getting a new shade per value."""
    return _STRIP_N_BINARY_STEPS_RE.sub('', label)


def load_binarization_sweep(output_dir, max_infeasible=None):
    """Load trajectories.csv from a binarization-sensitivity output_dir (n_binary_steps
    nested under each method's `sweep:` block, per config.expand_methods) and attach the
    canonical `group` and swept `n_binary_steps` columns. Labels with no n_binary_steps in
    the label (e.g. a baseline run left outside the sweep) are dropped -- they have no
    x-axis position in the sensitivity plot.

    Args:
        output_dir: path to a binarization-sweep experiment output_dir
        max_infeasible: if given, drop labels with too many fallback steps first (see
            _filter_infeasible)

    Returns:
        DataFrame, trajectories.csv rows for swept labels only, plus 'group' and
        'n_binary_steps' columns.
    """
    output_dir = Path(output_dir)
    trajectories = pd.read_csv(output_dir / 'trajectories.csv')
    if max_infeasible is not None:
        trajectories = _filter_infeasible(trajectories, output_dir.name, max_infeasible)
    trajectories['group'] = trajectories['label'].apply(derive_group)
    trajectories['n_binary_steps'] = trajectories['label'].apply(_n_binary_steps)
    swept = trajectories.dropna(subset=['n_binary_steps']).copy()
    swept['n_binary_steps'] = swept['n_binary_steps'].astype(int)
    return swept


def plot_binarization_sensitivity(experiment_output_dir, groups=None, max_infeasible=None, output_path=None):
    """
    Two-panel line plot of mean solve time and mean MIP gap vs. n_binary_steps, one line per method group

    Args:
        experiment_output_dir: path to a binarization-sweep output_dir
        groups: which canonical groups and order to plot, defaults to all groups in the sweep, ordered per GROUP_ORDER
        max_infeasible: if given, drop labels with too many fallback steps first (see _filter_infeasible)
        output_path: if given, save the figure to this path

    Returns:
        (fig, (ax_time, ax_gap))
    """
    sweep = load_binarization_sweep(experiment_output_dir, max_infeasible=max_infeasible)

    if groups is None:
        present = set(sweep['group'].unique())
        groups = [g for g in GROUP_ORDER if g in present]

    agg = (
        sweep.groupby(['group', 'n_binary_steps'])
        .agg(step_total_time_mean=('step_total_time', 'mean'), mip_gap_mean=('solver_mip_gap', 'mean'))
        .reset_index()
    )

    fig, (ax_time, ax_gap) = plt.subplots(1, 2, figsize=(7, 4))
    for group in groups:
        sub = agg[agg['group'] == group].sort_values('n_binary_steps')
        style = GROUP_STYLES.get(group, {})
        color, marker = style['color'], style['marker']
        label = GROUP_DISPLAY_NAMES.get(group, group)
        ax_time.plot(sub['n_binary_steps'], sub['step_total_time_mean'], color=color, marker=marker, label=label,
                     markeredgecolor='k', markeredgewidth=0.5)
        ax_gap.plot(sub['n_binary_steps'], sub['mip_gap_mean'], color=color, marker=marker, label=label,
                    markeredgecolor='k', markeredgewidth=0.5)

    x_ticks = sorted(sweep['n_binary_steps'].unique())
    for ax in (ax_time, ax_gap):
        ax.set_xlabel('Binarization steps')
        ax.set_xticks(x_ticks)
        ax.set_yscale('log')

    ax_time.set_ylabel('Mean step time (s)')
    ax_time.grid(visible=True, which='major', axis='x')
    ax_time.grid(visible=True, which='major', axis='y')
    ax_time.grid(visible=True, which='minor', axis='y', linewidth=0.5, alpha=0.8)

    ax_gap.set_ylabel('Mean MIP gap')
    ax_gap.grid(visible=True, which='major', axis='x')
    ax_gap.grid(visible=True, which='major', axis='y')
    ax_gap.grid(visible=True, which='minor', axis='y', linewidth=0.5, alpha=0.8)

    handles, labels_ = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper left', ncol=len(groups),
               bbox_to_anchor=(0.09, 1.0), bbox_transform=fig.transFigure)
    fig.subplots_adjust(top=0.89, wspace=0.3, left=0.1, right=0.95, bottom=0.15)

    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
    return fig, (ax_time, ax_gap)


def plot_binarization_cost_bars(experiment_output_dir, groups=None, max_infeasible=50, ylim_bottom=7000, ylim_top=8800,
                                legend_rows=None, output_path=None):
    """
    Grouped bar chart of realized cost vs. n_binary_steps, one cluster per n_binary_steps value,
    one bar per label - showing that more binaries does not mean a better solution

    Args:
        experiment_output_dir: path to a binarization-sweep output_dir (see load_binarization_sweep)
        groups: which canonical groups to include, in order; defaults to all groups present
            in the sweep, ordered per GROUP_ORDER. Controls which labels are shown, not the
            per-bar granularity -- every label within an included group gets its own bar.
        max_infeasible: if given, drop labels with too many fallback steps first (see_filter_infeasible)
        ylim_bottom: lower y-axis limit (EUR)
        ylim_top: upper y-axis limit (EUR)
        output_path: if given, save the figure to this path

    Returns:
        (fig, ax)
    """
    sweep = load_binarization_sweep(experiment_output_dir, max_infeasible=max_infeasible)

    if groups is None:
        present = set(sweep['group'].unique())
        groups = [g for g in GROUP_ORDER if g in present]
    sweep = sweep[sweep['group'].isin(groups)]

    sweep = sweep.copy()
    sweep['base_label'] = sweep['label'].apply(_strip_n_binary_steps)

    per_label_cost = sweep.groupby(['group', 'base_label', 'n_binary_steps'])['epanet_cost'].sum().reset_index()
    pivot = per_label_cost.pivot(index='base_label', columns='n_binary_steps', values='epanet_cost')

    ordered_labels = []
    for group in groups:
        for label in per_label_cost.loc[per_label_cost['group'] == group, 'base_label']:
            if label not in ordered_labels:
                ordered_labels.append(label)
    ordered_labels = MPCGraphs._sort_methods_by_group(ordered_labels)

    label_to_group = per_label_cost.drop_duplicates('base_label').set_index('base_label')['group'].to_dict()
    colors = build_label_colors({label: label_to_group[label] for label in ordered_labels})

    x_ticks = sorted(sweep['n_binary_steps'].unique())
    x = np.arange(len(x_ticks))
    width = 0.8 / len(ordered_labels)

    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(ordered_labels) + 2), 4.2))
    label_to_handle = {}
    for i, label in enumerate(ordered_labels):
        offset = (i - (len(ordered_labels) - 1) / 2) * width
        bars = ax.bar(x + offset, pivot.loc[label, x_ticks], width=width, color=colors[label], edgecolor='k',
                       label=to_latex_label(label))
        label_to_handle[label] = bars

    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel('Binarization steps')
    ax.set_ylabel('Realized Cost (€)')
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    ax.set_axisbelow(True)
    ax.grid(visible=True, which='major', axis='y', linewidth=0.5, alpha=0.8)

    if legend_rows is not None:
        row_label_lists = [list(row) for row in legend_rows]
        handles_leg, labels_leg, ncol = build_legend_order(row_label_lists, label_to_handle)
    else:
        handles, labels_leg = ax.get_legend_handles_labels()
        handles_leg, labels_leg = transpose_legend(handles, labels_leg, ncol=4)
        ncol = 4

    ax.legend(handles_leg, labels_leg, ncol=ncol, fontsize=LEG_FONT_SIZE, loc='upper left',
              bbox_to_anchor=(0.13, 0.98), bbox_transform=fig.transFigure)
    fig.subplots_adjust(top=0.74, bottom=0.15, left=0.14, right=0.95)

    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
    return fig, ax


def plot_computational_boxplots(trajectories, groups=None, case_colors=None, output_path=None):
    """
    Two-panel boxplot comparing per-step solve behavior across method groups and case
    studies: left panel is step_total_time, right panel is solver_mip_gap (log scale,
    MILP steps only). One box per (group, case_study) pair, colored by case_study, so
    the reader sees the actual distribution/skew/outliers within each group instead of
    just the mean $\\pm$ std collapsed into a single table cell.

    Args:
        trajectories: DataFrame from load_trajectories (raw per-step rows across case
            studies, with 'case_study' and 'group' columns already attached)
        groups: which canonical groups to plot, in order; defaults to all groups
            present in `trajectories`, ordered per GROUP_ORDER
        case_colors: dict {case_study_label: color}; defaults to matplotlib's C0/C1/...
            in first-seen order
        output_path: if given, save the figure to this path

    Returns:
        (fig, (ax_time, ax_gap))
    """
    if groups is None:
        present = set(trajectories['group'].unique())
        groups = [g for g in GROUP_ORDER if g in present]

    case_studies = list(dict.fromkeys(trajectories['case_study']))
    if case_colors is None:
        case_colors = {cs: f'C{i}' for i, cs in enumerate(case_studies)}

    fig, (ax_time, ax_gap) = plt.subplots(1, 2, figsize=(12, 5))

    width = 0.8 / len(case_studies)
    for i, case_study in enumerate(case_studies):
        offset = (i - (len(case_studies) - 1) / 2) * width
        positions = [j + offset for j in range(len(groups))]
        color = case_colors[case_study]

        sub = trajectories[trajectories['case_study'] == case_study]
        has_gap = sub['solver_mip_gap'].apply(np.isfinite)

        time_data = [sub.loc[sub['group'] == g, 'step_total_time'].dropna() for g in groups]
        gap_data = [sub.loc[has_gap & (sub['group'] == g), 'solver_mip_gap'] for g in groups]

        for ax, data in [(ax_time, time_data), (ax_gap, gap_data)]:
            bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True,
                             showfliers=True, medianprops={'color': 'black'})
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.6)

    group_labels = [GROUP_DISPLAY_NAMES.get(g, g) for g in groups]
    for ax in (ax_time, ax_gap):
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(group_labels, rotation=30, ha='right')
        ax.set_xlim(-0.5, len(groups) - 0.5)

    ax_time.set_yscale('log')
    ax_time.set_ylabel('Step Time (s)')
    ax_time.set_title('Per-step solve time')

    ax_gap.set_yscale('log')
    ax_gap.set_ylabel('MIP Gap')
    ax_gap.set_title('Per-step MIP gap (MILP steps only)')

    handles = [Patch(facecolor=case_colors[cs], alpha=0.6, label=cs) for cs in case_studies]
    fig.legend(handles=handles, loc='upper center', ncol=len(case_studies), bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
    return fig, (ax_time, ax_gap)