#!/usr/bin/env python3
"""
Plot mean FA (original reference, target before, harmonized after) for multiple b-shells
in one figure: one dot-plot row per b-value, plus a summary line plot of group means vs b.

Requires existing *_InMNI_FA.nii.gz under templatePath (same convention as debug_FA_modified.py).

Example:
  python plot_meanFA_multibshell.py \\
    --templatePath ./template/ \\
    --refSite ME_device_1_soft_1 \\
    --targetSite CG_device_1_soft_1 \\
    --ref-pattern harmonization_input_ref_b{bshell}.csv.modified \\
    --tar-unproc-pattern harmonization_input_target_b{bshell}.csv.modified \\
    --tar-harm-pattern harmonization_input_target_b{bshell}.csv.modified.harmonized \\
    --mniTmp /data/pnl/IITAtlas/IITmean_FA.nii.gz
"""

import os

os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import sys
from os.path import dirname, basename, abspath, join as pjoin, exists

import numpy as np
from nibabel import load

# Default b-values to iterate (override with --bshells)
DEFAULT_BSHELLS = [200, 500, 1000, 2000, 3000]


def read_caselist_csv(caselist_file):
    cases = []
    with open(caselist_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            dwipath = parts[0].strip()
            maskpath = parts[1].strip() if len(parts) > 1 else ''
            cases.append((dwipath, maskpath))
    return cases


def analyze_stat_original_ref(caselist_file, template_path, mni_tmp):
    """Mean FA on IIT skeleton per subject; original <prefix>_InMNI_FA only."""
    skel_path = pjoin(dirname(mni_tmp), 'IITmean_FA_skeleton.nii.gz')
    skel = load(skel_path)
    skel_mask = (skel.get_fdata() > 0) * 1.0

    cases = read_caselist_csv(caselist_file)
    mean_attr = []
    for img_path, _mask in cases:
        in_prefix = img_path.split('.nii')[0]
        prefix = basename(in_prefix)
        fa_img = pjoin(template_path, prefix + '_InMNI_FA.nii.gz')
        if not exists(fa_img):
            print(f'  WARNING: {basename(fa_img)} does not exist, skipping')
            continue
        data = load(fa_img).get_fdata()
        temp = data * skel_mask
        mean_attr.append(float(temp[temp > 0].mean()))
    return mean_attr


def plot_panel_on_axes(ax, ref_vals, tar_vals, harm_vals, ref_site, target_site, bshell, show_xtick_labels=True):
    """Draw one three-group dot plot on given axes."""
    labels = [ref_site, f'{target_site}\nbefore', f'{target_site}\nafter']
    data = [ref_vals, tar_vals, harm_vals]
    colors = ['#1f77b4', '#d62728', '#2ca02c']

    for i, values in enumerate(data):
        values = np.asarray(values, dtype=float).ravel()
        if values.size == 0:
            continue
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        c = colors[i]
        ax.scatter([i] * len(values), values, marker='*', s=60, color=c, zorder=3, alpha=0.85)
        ax.scatter(i, mean_val, marker='D', s=48, color='cyan', edgecolors='black', zorder=5)
        ax.errorbar(i, mean_val, yerr=std_val, fmt='none', ecolor='black', capsize=6, zorder=2)
        ax.annotate(
            f'{mean_val:.4f}',
            xy=(i, mean_val),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center',
            fontsize=7,
            zorder=6,
        )

    ax.set_xticks(range(3))
    if show_xtick_labels:
        ax.set_xticklabels(labels, fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel('mean FA', fontsize=9)
    ax.set_title(f'b = {bshell}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 2.5)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-b-shell meanFA panel plot (original ref, target before/after).',
    )
    parser.add_argument('--templatePath', required=True, help='Template directory with *_InMNI_FA.nii.gz')
    parser.add_argument('--refSite', required=True)
    parser.add_argument('--targetSite', required=True)
    parser.add_argument(
        '--ref-pattern',
        required=True,
        help='Caselist path with {bshell} placeholder, e.g. harmonization_input_ref_b{bshell}.csv.modified',
    )
    parser.add_argument('--tar-unproc-pattern', required=True, help='Target unprocessed caselist pattern')
    parser.add_argument('--tar-harm-pattern', required=True, help='Harmonized caselist pattern')
    parser.add_argument('--mniTmp', required=True, help='IITmean_FA.nii.gz path')
    parser.add_argument(
        '--bshells',
        default=','.join(str(x) for x in DEFAULT_BSHELLS),
        help=f'Comma-separated b-values (default: {DEFAULT_BSHELLS})',
    )
    parser.add_argument(
        '--out',
        default=None,
        help='Output PNG path (default: templatePath/meanFA_multib_original_ref.png)',
    )
    args = parser.parse_args()

    bshells = [int(x.strip()) for x in args.bshells.split(',') if x.strip()]
    template_path = abspath(args.templatePath)
    mni_tmp = abspath(args.mniTmp)

    if args.out:
        out_path = abspath(args.out)
    else:
        out_path = pjoin(template_path, 'meanFA_multib_original_ref.png')

    # Collect stats per b
    rows = []  # list of dicts with keys ref, tar, harm, b
    for b in bshells:
        ref_csv = abspath(args.ref_pattern.format(bshell=b))
        tar_csv = abspath(args.tar_unproc_pattern.format(bshell=b))
        harm_csv = abspath(args.tar_harm_pattern.format(bshell=b))

        missing = [p for p in (ref_csv, tar_csv, harm_csv) if not exists(p)]
        if missing:
            print(f'WARNING b={b}: missing caselist(s), skipping this shell:')
            for m in missing:
                print(f'    {m}')
            continue

        print(f'=== b = {b} ===')
        ref_m = analyze_stat_original_ref(ref_csv, template_path, mni_tmp)
        tar_m = analyze_stat_original_ref(tar_csv, template_path, mni_tmp)
        harm_m = analyze_stat_original_ref(harm_csv, template_path, mni_tmp)

        if not ref_m:
            print(f'  WARNING b={b}: no reference InMNI FA loaded, skipping')
            continue

        print(
            f'  Reference: mean={np.mean(ref_m):.4f} N={len(ref_m)} | '
            f'before: mean={np.mean(tar_m):.4f} N={len(tar_m)} | '
            f'after: mean={np.mean(harm_m):.4f} N={len(harm_m)}'
        )
        rows.append({'b': b, 'ref': ref_m, 'tar': tar_m, 'harm': harm_m})

    if not rows:
        print('ERROR: No b-shell had valid data. Check caselist paths and templatePath InMNI files.')
        sys.exit(1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(rows)
    height_ratios = [1.0] * n + [0.65]
    fig, axes = plt.subplots(
        n + 1,
        1,
        figsize=(12, 2.2 * n + 3.0),
        height_ratios=height_ratios,
    )
    axes = np.atleast_1d(axes).ravel().tolist()

    means_ref = []
    means_tar = []
    means_harm = []
    bs_used = []

    for idx, row in enumerate(rows):
        ax = axes[idx]
        b = row['b']
        bs_used.append(b)
        means_ref.append(np.mean(row['ref']))
        means_tar.append(np.mean(row['tar']) if row['tar'] else np.nan)
        means_harm.append(np.mean(row['harm']) if row['harm'] else np.nan)

        plot_panel_on_axes(
            ax,
            row['ref'],
            row['tar'],
            row['harm'],
            args.refSite,
            args.targetSite,
            b,
            show_xtick_labels=(idx == n - 1),
        )

    ax_sum = axes[-1]
    ax_sum.plot(bs_used, means_ref, 'o-', color='#1f77b4', label=f'Reference ({args.refSite})', linewidth=2)
    ax_sum.plot(bs_used, means_tar, 's-', color='#d62728', label=f'{args.targetSite} before', linewidth=2)
    ax_sum.plot(bs_used, means_harm, '^-', color='#2ca02c', label=f'{args.targetSite} after', linewidth=2)
    ax_sum.set_xlabel('b-value (s/mm²)', fontsize=11)
    ax_sum.set_ylabel('mean FA (group mean)', fontsize=11)
    ax_sum.set_title('Group mean FA vs b-shell (skeleton means)', fontsize=11)
    ax_sum.legend(loc='best', fontsize=9)
    ax_sum.grid(True, alpha=0.3)
    if len(bs_used) > 1:
        ax_sum.set_xticks(bs_used)

    fig.suptitle(
        'Original reference vs target before / harmonized after — all b-shells',
        fontsize=13,
        y=1.002,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
