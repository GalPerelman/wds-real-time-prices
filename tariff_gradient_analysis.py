import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graphs import (
    MPCGraphs, LEG_FONT_SIZE, derive_group, build_label_colors, to_latex_label,
    validate_output_dir, build_legend_order, transpose_legend
)


def tariff_cv(tariff_path):
    df = pd.read_csv(tariff_path)
    return df['tariff'].std() / df['tariff'].mean()


def _regret_table_rows(category, category_val, summary, oracle_cost, methods):
    lp = summary[summary['label'] == 'LP'].iloc[0]
    lp_cost = lp['actual_cost']
    lp_vol = lp['violated_vol']

    rows = []
    for label in methods:
        match = summary[summary['label'] == label]
        if match.empty:
            raise ValueError(f"Label '{label}' not found for {category} '{category_val}'")
        r = match.iloc[0]
        regret = r['actual_cost'] - oracle_cost if oracle_cost is not None else float('nan')
        vol = r['violated_vol']
        vol_avoided = lp_vol - vol
        premium = r['actual_cost'] - lp_cost
        rows.append({
            category: category_val, 'method': label,
            'cost': r['actual_cost'], 'regret': regret, 'violated_vol': vol,
            'premium': premium,
            'vol_reduction_pct': 100 * vol_avoided / lp_vol if lp_vol > 0 else float('nan'),
            'premium_per_m3': premium / vol_avoided if vol_avoided > 0 else float('nan'),
        })
    return rows


def build_gradient_table(tariff_configs, methods):
    rows = []
    for tariff, cfg in tariff_configs.items():
        summary = pd.read_csv(cfg['summary'])
        oracle_cost = pd.read_csv(cfg['oracle'])['actual_cost'].iloc[0]
        cv = tariff_cv(cfg['tariff_path'])
        for row in _regret_table_rows('tariff', tariff, summary, oracle_cost, methods):
            row['cv'] = cv
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['cv', 'tariff']).reset_index(drop=True)


def build_forecast_table(forecast_configs, oracle_summary, methods):
    oracle_cost = pd.read_csv(oracle_summary)['actual_cost'].iloc[0]
    rows = []
    for forecast, cfg in forecast_configs.items():
        summary = pd.read_csv(cfg['summary'])
        price_mae = summary['price_mae'].iloc[0]
        demand_mae = summary['demand_mae'].iloc[0]
        for row in _regret_table_rows('forecast', forecast, summary, oracle_cost, methods):
            row['price_mae'] = price_mae
            row['demand_mae'] = demand_mae
            rows.append(row)
    return pd.DataFrame(rows)


def build_case_study_table(case_configs, methods):
    rows = []
    for case, tariff_configs in case_configs.items():
        for tariff, cfg in tariff_configs.items():
            summary = pd.read_csv(cfg['summary'])
            oracle_cost = pd.read_csv(cfg['oracle'])['actual_cost'].iloc[0] if cfg['oracle'] else None
            for row in _regret_table_rows('tariff', tariff, summary, oracle_cost, methods):
                row['case'] = case
                rows.append(row)
    return pd.DataFrame(rows)


def _sp_lambda_value(label):
    """Extract the lambda_cvar value out of a 'SP_...|lambda_cvar=v' label, or None if absent."""
    m = re.search(r'lambda_cvar=([^|]+)', label)
    return m.group(1) if m else None


def _compact_method_label(label, drop_lambda):
    """
    to_latex_label(), but with the '\\lambda=...' term stripped from CCO-CVaR labels when every CVaR method being
    displayed shares the same lambda (true for METHODS) - repeating a constant in every row just adds width,
    so it's dropped here and noted below the table instead. alpha is left untouched.
    """
    latex = to_latex_label(label)
    if drop_lambda and derive_group(label) == 'CCO-CVaR':
        latex = re.sub(r',\s*\\lambda=[^,{}]*', '', latex)
    return latex


def export_case_study_latex(table, tariffs, methods, caption, label, output_path=""):
    """
    LaTeX-ready summary table restricted to `tariffs` (e.g. ['HU'] for the main-paper table,
    ['IT', 'TOU'] for the supplementary table). Rows grouped by case study (and tariff, when more
    than one is included) and method; columns = cost, premium vs. LP, violated volume, and
    premium per m3 of violation avoided vs. LP. Regret is omitted since not every case/tariff has
    an oracle run.

    Post-processes pandas' raw to_latex() output: header cells use \\shortstack (a bare '\\\\'
    inside a plain l/r column breaks the whole row, not just that cell, so wrapping is needed to
    keep the table narrow enough to fit the page width), the column-header row and the index-name
    row (pandas emits them separately whenever the index has names) are merged into one line, the
    case/tariff \\multirow labels drop the [t] vpos flag, the caption is right-aligned via the
    caption package (plain \\begin{flushright} has no effect -- \\caption's own singlelinecheck
    re-centers short captions regardless of surrounding alignment), and the \\cline separator
    pandas draws after every group (including the last one, redundant right above \\bottomrule)
    is kept between groups but dropped at the very end.
    """
    sub = table[table['tariff'].isin(tariffs) & table['method'].isin(methods)].copy()
    sub['method'] = pd.Categorical(sub['method'], categories=methods, ordered=True)

    cvar_lambdas = {_sp_lambda_value(m) for m in methods if derive_group(m) == 'CCO-CVaR'}
    drop_lambda = len(cvar_lambdas) == 1
    sub['Method'] = sub['method'].map(lambda m: _compact_method_label(m, drop_lambda))
    note = rf"CCO-CVaR uses $\lambda={cvar_lambdas.pop()}$." if drop_lambda else None

    multi_tariff = len(tariffs) > 1
    sort_cols = ['case', 'tariff', 'method'] if multi_tariff else ['case', 'method']
    index_cols = ['case', 'tariff', 'Method'] if multi_tariff else ['case', 'Method']
    sub = sub.sort_values(sort_cols).set_index(index_cols)
    sub.index = sub.index.set_names([n.capitalize() for n in sub.index.names])

    col_map = {
        'cost': r'\shortstack{Cost\\(\euro{})}',
        'premium': r'\shortstack{Premium\\(\euro{})}',
        'violated_vol': r'\shortstack{Violated\\Vol.\\($m^3$)}',
        'premium_per_m3': r'\shortstack{Reliability\\Premium\\(\euro{}/$m^3$)}',
    }
    out = sub[list(col_map)].rename(columns=col_map)

    latex = out.to_latex(float_format="%.1f", multirow=True, caption=caption, label=label,
                         na_rep='--', escape=False)

    latex = latex.replace(r'\multirow[t]', r'\multirow')
    latex = re.sub(r'(?:\\cline\{[^}]*\}\s*)+\\bottomrule', r'\\bottomrule', latex)
    latex = _merge_header_rows(latex)
    if note:
        latex = latex.replace(
            r'\end{tabular}',
            r'\end{tabular}' + '\n' + r'\vskip 5pt' + '\n' + r'{\footnotesize Note: ' + note + r'\par}')

    if output_path:
        validate_output_dir(output_path)
        Path(output_path).write_text(latex)
    return latex


def _merge_header_rows(latex):
    """Pandas emits the column headers and the index level names (here: 'Case'/'Tariff'/'Method')
    as two separate rows above \\midrule whenever the index has names. Merge them into a single
    header row, taking whichever of the two rows has content in each column position."""
    lines = latex.splitlines()
    mid = lines.index(r'\midrule')

    def _cells(line):
        return [c.strip() for c in re.sub(r'\s*\\\\\s*$', '', line).split('&')]

    names_cells = _cells(lines[mid - 1])
    header_cells = _cells(lines[mid - 2])
    merged = [a if a else b for a, b in zip(names_cells, header_cells)]
    lines[mid - 2:mid] = [' & '.join(merged) + r' \\']
    return '\n'.join(lines)


def _plot_gradient_bars(table, category_col, category_order, xticklabels, methods, vol_ylim=None,
                        xlabel=None, extra_content=None, output_path=""):
    """
    Shared two-panel grouped bar chart: regret (EUR) | violated volume (m3), x = category_order
    (tariff or forecast model), bars grouped by method label. LP is included in every cluster as
    the reference point, so a method's cost/reliability relative to LP is read directly off the
    chart.

    extra_content: optional callable(ax_regret, ax_vol) -> (handles, labels), invoked after the
    bars are drawn but before the legend/layout are finalized. Lets a caller (e.g. the forecast
    MAE line overlay) add extra artists and have them merged into the shared legend as a new row,
    without _plot_gradient_bars needing to know what those artists are.
    """
    label_to_group = {label: derive_group(label) for label in methods}
    colors = build_label_colors(label_to_group)

    x = np.arange(len(category_order))
    width = 0.8 / len(methods)

    fig, (ax_regret, ax_vol) = plt.subplots(1, 2, figsize=(max(7, 0.9 * len(methods)), 4.2))

    for i, label in enumerate(methods):
        sub = table[table['method'] == label].set_index(category_col).loc[category_order]
        offset = (i - (len(methods) - 1) / 2) * width

        ax_regret.bar(x + offset, sub['regret'], width=width, color=colors[label],
                      edgecolor='k', label=to_latex_label(label))
        ax_vol.bar(x + offset, sub['violated_vol'], width=width, color=colors[label],
                   edgecolor='k', label=to_latex_label(label))

    ax_regret.axhline(0, color='k', linewidth=0.8, alpha=0.6)
    ax_regret.set_xticks(x)
    ax_regret.set_xticklabels(xticklabels)
    ax_regret.set_ylabel('Regret (€)')
    ax_regret.set_axisbelow(True)

    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(xticklabels)
    ax_vol.set_ylabel('Violated Volume ($m^3$)')
    ax_vol.set_axisbelow(True)
    if xlabel is not None:
        ax_regret.set_xlabel(xlabel)
        ax_vol.set_xlabel(xlabel)
    if vol_ylim is not None:
        ax_vol.set_ylim(bottom=vol_ylim[0], top=vol_ylim[1])

    # build_legend_order applies to_latex_label itself, so key it off the raw method names
    # (methods), not ax_vol's already-latex-rendered legend labels.
    method_handles, _ = ax_vol.get_legend_handles_labels()
    extra_handles, extra_labels = extra_content(ax_regret, ax_vol) if extra_content is not None else ([], [])
    label_to_handle = dict(zip(methods, method_handles))
    label_to_handle.update(dict(zip(extra_labels, extra_handles)))

    ncol = min(len(methods), 3)
    method_rows = [list(methods[i:i + ncol]) for i in range(0, len(methods), ncol)]
    row_label_lists = method_rows + ([list(extra_labels)] if extra_labels else [])
    handles_leg, labels_leg, ncol = build_legend_order(row_label_lists, label_to_handle, ncol=ncol)
    fig.legend(handles_leg, labels_leg, fontsize=LEG_FONT_SIZE, ncol=ncol,
               loc='upper left', bbox_to_anchor=(0.11, 1.0))

    fig.subplots_adjust(top=0.83, wspace=0.3, left=0.12, right=0.97, bottom=0.2)
    return fig, (ax_regret, ax_vol)


def plot_tariff_gradient_bars(table, methods=None, output_path=""):
    if methods is None:
        methods = MPCGraphs._sort_methods_by_group(list(dict.fromkeys(table['method'])))

    tariff_order = table.sort_values('cv')['tariff'].drop_duplicates().tolist()
    cv_by_tariff = table.drop_duplicates('tariff').set_index('tariff')['cv']
    xticklabels = [f"{t}\n" + r"$\frac{\sigma}{\mu}$=" + f"{cv_by_tariff[t]:.2f}" for t in tariff_order]
    xticklabels = [f"{t}" for t in tariff_order]

    fig, (ax_regret, ax_vol) = _plot_gradient_bars(table, 'tariff', tariff_order, xticklabels, methods,
                                                   vol_ylim=(-25, None), xlabel='Tariffs Dataset',
                                                   output_path=output_path)

    if output_path:
        validate_output_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, (ax_regret, ax_vol)


def plot_forecast_gradient_bars(table, methods=None, forecast_order=None, output_path=""):
    """Grouped bar chart vs. forecast model (Naive / RF / XGB); the forecast's price/demand MAE
    is overlaid as two lines on a secondary y-axis on both panels, so the accuracy trend can be
    read against both the regret and violated-volume trends. The secondary axis label/ticks are
    only drawn on the violated-volume panel (kept in sync via a shared ylim) to avoid crowding
    the boundary between the two panels. See _plot_gradient_bars."""
    if methods is None:
        methods = MPCGraphs._sort_methods_by_group(list(dict.fromkeys(table['method'])))

    forecast_order = [f for f in forecast_order if f in table['forecast'].unique()]
    mae_by_forecast = table.drop_duplicates('forecast').set_index('forecast')
    price_mae = [mae_by_forecast.loc[f, 'price_mae'] for f in forecast_order]
    demand_mae = [mae_by_forecast.loc[f, 'demand_mae'] for f in forecast_order]
    x = np.arange(len(forecast_order))
    c = "lightgrey"
    cc = "darkgrey"

    def _add_mae_lines(ax_regret, ax_vol):
        ax_mae_vol = ax_vol.twinx()
        line_price, = ax_mae_vol.plot(x, price_mae, marker='o', color=cc, linestyle='-', markeredgecolor=cc, mfc=c,
                                      label='Price MAE (€/MWh)')
        line_demand, = ax_mae_vol.plot(x, demand_mae, marker='s', color=cc, linestyle='--', markeredgecolor=cc, mfc=c,
                                       label='Demand MAE (m³/hr)')
        ax_mae_vol.set_ylabel('Forecast MAE', color="grey")
        ax_mae_vol.tick_params(axis='y', colors="grey")

        ax_mae_regret = ax_regret.twinx()
        ax_mae_regret.plot(x, price_mae, marker='o', color=cc, markeredgecolor=cc, mfc=c, linestyle='-')
        ax_mae_regret.plot(x, demand_mae, marker='s', color=cc, markeredgecolor=cc, mfc=c, linestyle='--')
        ax_mae_regret.set_ylim(ax_mae_vol.get_ylim())
        ax_mae_regret.axis('off')

        return [line_price, line_demand], ['Price MAE (€/MWh)', 'Demand MAE (m³/hr)']

    fig, (ax_regret, ax_vol) = _plot_gradient_bars(table, 'forecast', forecast_order, forecast_order, methods,
                                                   vol_ylim=(None, None), xlabel='Forecast model',
                                                   extra_content=_add_mae_lines, output_path=output_path)

    fig.set_size_inches(max(8, 0.9 * len(methods)), 4.2)
    fig.subplots_adjust(top=0.78, wspace=0.3, left=0.12, right=0.9, bottom=0.15)
    if output_path:
        validate_output_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, (ax_regret, ax_vol)


def graphical_abstract(table, category_col, category_order, xticklabels, methods, vol_ylim=None, xlabel=""):
    graphical_abstract_labels = {'LP': 'PTO', 'RO_omega=2.5': r'$RO_{Ell}$', 'RO_gamma=[8, 14]': r'$RO_{BOU}$',
                                 'SP_alpha=[0.95, 0.95]|lambda_cvar=0': 'CCO',
                                 'SP_alpha=[0.95, 0.95]|lambda_cvar=1': 'CCO-CVaR'}

    label_to_group = {label: derive_group(label) for label in methods}
    colors = build_label_colors(label_to_group)
    x = np.arange(len(category_order))
    width = 0.8 / len(methods)

    fig, (ax_regret, ax_vol) = plt.subplots(1, 2, figsize=(max(7, 0.9 * len(methods)), 4.2))

    for i, label in enumerate(methods):
        sub = table[table['method'] == label].set_index(category_col).loc[category_order]
        offset = (i - (len(methods) - 1) / 2) * width
        ax_regret.bar(x + offset, sub['regret'], width=width, color=colors[label],
                      edgecolor='k', label=graphical_abstract_labels[label])
        ax_vol.bar(x + offset, sub['violated_vol'], width=width, color=colors[label],
                   edgecolor='k', label=graphical_abstract_labels[label])

    ax_regret.axhline(0, color='k', linewidth=0.8, alpha=0.6)
    ax_regret.set_xticks(x)
    ax_regret.set_xticklabels(xticklabels)
    ax_regret.set_ylabel('Regret (€)', fontsize=14)
    ax_regret.set_axisbelow(True)

    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels(xticklabels)
    ax_vol.set_ylabel('Violated Volume ($m^3$)', fontsize=14)
    ax_vol.set_axisbelow(True)

    ax_regret.set_xlabel(xlabel, fontsize=14)
    ax_vol.set_xlabel(xlabel, fontsize=14)
    if vol_ylim is not None:
        ax_vol.set_ylim(bottom=vol_ylim[0], top=vol_ylim[1])

    ax_regret.tick_params(axis='both', labelsize=14)
    ax_vol.tick_params(axis='both', labelsize=14)

    method_handles, _ = ax_vol.get_legend_handles_labels()
    label_to_handle = dict(zip(methods, method_handles))

    ncol = min(len(methods), 5)
    method_rows = [list(methods[i:i + ncol]) for i in range(0, len(methods), ncol)]
    handles_leg, labels_leg, ncol = build_legend_order(method_rows, label_to_handle, ncol=ncol)

    handles, labels_leg = ax_regret.get_legend_handles_labels()
    handles, labels_leg = transpose_legend(handles, labels_leg, ncol)
    ax_regret.legend(handles, labels_leg, fontsize=14, ncol=ncol,
               loc='upper left', bbox_to_anchor=(-0.03, 1.33), columnspacing=0.9, handletextpad=0.2)

    fig.subplots_adjust(top=0.8, wspace=0.35, left=0.14, right=0.98, bottom=0.15)
    return fig, (ax_regret, ax_vol)
