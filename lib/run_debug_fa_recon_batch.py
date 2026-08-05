#!/usr/bin/env python3
"""
Run debug_FA_modified.py once per b-shell (200, 500, …) with reconstructed reference,
then build an HTML dashboard with **reconstructed reference only** (no original-ref plots/columns).

Typical full run (registration + stats + meanFAplot_b{b}_reconstr.png per shell)::

  python run_debug_fa_recon_batch.py \\
    --templatePath /path/to/template \\
    --refSite ME_device_1_soft_1 \\
    --targetSite CG_device_1_soft_1 \\
    --ref-pattern /path/harmonization_input_ref_b{bshell}.csv.modified \\
    --tar-unproc-pattern /path/harmonization_input_target_b{bshell}.csv.modified \\
    --tar-harm-pattern /path/harmonization_input_target_b{bshell}.csv.modified.harmonized \\
    --mniTmp /data/pnl/IITAtlas/IITmean_FA.nii.gz

Plots/stats only (existing *_InMNI_*; no ANTs/dtifit)::

  python run_debug_fa_recon_batch.py ... --plot-only

Requires ``build_fa_dashboard.py`` in the same directory (or on PYTHONPATH) for stats/CSS helpers.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import numpy as np
from datetime import datetime, timezone
from html import escape
from os.path import abspath, basename, dirname, exists, join as pjoin

# Same env as debug_FA / dashboard (MKL)
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ.setdefault('OMP_NUM_THREADS', '1')


def _import_dashboard_helpers():
    try:
        from build_fa_dashboard import (
            CSS,
            DEFAULT_BSHELLS,
            mean_fa_per_subject,
            png_to_data_uri,
            summarize,
        )
    except ImportError as e:
        print(
            'Could not import build_fa_dashboard (needed for dashboard stats). '
            'Place build_fa_dashboard.py next to this script or set PYTHONPATH.',
            file=sys.stderr,
        )
        raise e
    return CSS, DEFAULT_BSHELLS, mean_fa_per_subject, png_to_data_uri, summarize


def run_debug_fa_one(
    *,
    debug_fa_py: str,
    template_path: str,
    ref_site: str,
    target_site: str,
    ref_csv: str,
    tar_unproc_csv: str,
    tar_harm_csv: str,
    mni_tmp: str,
    bshell: int,
    plot_only: bool,
    skip_ref_register: bool,
    skip_harm_register: bool,
    force: bool,
    extra_args: list[str],
) -> None:
    cmd = [
        sys.executable,
        abspath(debug_fa_py),
        '--templatePath',
        abspath(template_path),
        '--refSite',
        ref_site,
        '--targetSite',
        target_site,
        '--ref',
        abspath(ref_csv),
        '--tar_unproc',
        abspath(tar_unproc_csv),
        '--tar_harm',
        abspath(tar_harm_csv),
        '--bshell_b',
        str(bshell),
        '--mniTmp',
        abspath(mni_tmp),
    ]
    if plot_only:
        cmd.extend(['--plot-only', '--ref-reconstructed'])
    else:
        if not skip_ref_register:
            cmd.append('--register-ref-reconstructed')
        if not skip_harm_register:
            cmd.append('--register')
        # Stats must read reconstructed_*_InMNI_FA whenever we skip re-running ref registration.
        if skip_ref_register:
            cmd.append('--ref-reconstructed')
    if force:
        cmd.append('--force')
    cmd.extend(extra_args)

    print('\n' + '=' * 72)
    print(' '.join(cmd))
    print('=' * 72 + '\n')
    subprocess.check_call(cmd)


def collect_stat_rows(
    template_path: str,
    mni_tmp: str,
    bshells: list[int],
    ref_pattern: str,
    tar_unproc_pattern: str,
    tar_harm_pattern: str,
    mean_fa_per_subject,
    summarize,
) -> list[dict]:
    """Same logic as build_fa_dashboard: one row per b; ref_recon from reconstructed_* InMNI FA."""
    rows = []
    template_path = abspath(template_path)
    mni_tmp = abspath(mni_tmp)

    for b in bshells:
        ref_csv = abspath(ref_pattern.format(bshell=b))
        tar_csv = abspath(tar_unproc_pattern.format(bshell=b))
        harm_csv = abspath(tar_harm_pattern.format(bshell=b))

        row: dict = {'b': b, 'skip': True}
        if not all(exists(p) for p in (ref_csv, tar_csv, harm_csv)):
            print(f'WARNING b={b}: missing caselist, skipping stats row')
            rows.append(row)
            continue

        v_recon = mean_fa_per_subject(ref_csv, template_path, mni_tmp, reconstructed_ref=True)
        v_tar = mean_fa_per_subject(tar_csv, template_path, mni_tmp, reconstructed_ref=False)
        v_harm = mean_fa_per_subject(harm_csv, template_path, mni_tmp, reconstructed_ref=False)

        mr, sr, nr = summarize(v_recon)
        mt, st, nt = summarize(v_tar)
        mh, sh, nh = summarize(v_harm)

        row = {
            'b': b,
            'skip': False,
            'ref_recon_mean': mr,
            'ref_recon_std': sr,
            'ref_recon_n': nr,
            'tar_mean': mt,
            'tar_std': st,
            'tar_n': nt,
            'harm_mean': mh,
            'harm_std': sh,
            'harm_n': nh,
        }
        rows.append(row)
    return rows


def build_recon_only_html(
    title: str,
    ref_site: str,
    target_site: str,
    template_path: str,
    rows: list[dict],
    bshells: list[int],
    css: str,
    png_to_data_uri,
) -> str:
    """Table: ref recon + target before/after only. Figures: meanFAplot_b{b}_reconstr.png only."""
    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        f'<title>{escape(title)}</title><style>{css}</style></head><body>',
        f'<h1>{escape(title)}</h1>',
        f'<p class="meta">Generated {escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))} · '
        f'ref (reconstructed): <strong>{escape(ref_site)}</strong> · target: <strong>{escape(target_site)}</strong> · '
        f'template: <code>{escape(abspath(template_path))}</code></p>',
        '<p class="meta">This dashboard shows only <strong>reconstructed reference</strong> FA and single-shell plots '
        '<code>meanFAplot_b&#123;b&#125;_reconstr.png</code>.</p>',
    ]

    parts.append('<section><h2>Statistics by b-shell (reconstructed reference)</h2>')
    parts.append('<div class="table-wrap">')
    parts.append(
        '<table><thead><tr>'
        '<th>b</th>'
        '<th class="num">Ref (recon) mean</th><th class="num">Ref (recon) std</th><th class="num">N</th>'
        '<th class="num">Target before mean</th><th class="num">std</th><th class="num">N</th>'
        '<th class="num">Target after mean</th><th class="num">std</th><th class="num">N</th>'
        '</tr></thead><tbody>'
    )

    def _fmt_stat(x: float, nd: int = 4) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return '—'
        return f'{float(x):.{nd}f}'

    def _fmt_n(n: int) -> str:
        return str(int(n)) if n else '—'

    for r in rows:
        if r.get('skip'):
            continue
        parts.append(
            '<tr>'
            f'<td>{r["b"]}</td>'
            f'<td class="num">{_fmt_stat(r["ref_recon_mean"])}</td>'
            f'<td class="num">{_fmt_stat(r["ref_recon_std"])}</td>'
            f'<td class="num">{_fmt_n(r["ref_recon_n"])}</td>'
            f'<td class="num">{_fmt_stat(r["tar_mean"])}</td>'
            f'<td class="num">{_fmt_stat(r["tar_std"])}</td>'
            f'<td class="num">{_fmt_n(r["tar_n"])}</td>'
            f'<td class="num">{_fmt_stat(r["harm_mean"])}</td>'
            f'<td class="num">{_fmt_stat(r["harm_std"])}</td>'
            f'<td class="num">{_fmt_n(r["harm_n"])}</td>'
            '</tr>'
        )
    parts.append('</tbody></table></div>')
    parts.append(
        '<p class="caption">Per-subject means over IIT skeleton. '
        'Reference uses <code>reconstructed_&lt;prefix&gt;_InMNI_FA.nii.gz</code>.</p>'
    )
    parts.append('</section>')

    for b in bshells:
        png_path = pjoin(abspath(template_path), f'meanFAplot_b{b}_reconstr.png')
        title_sec = f'Single-shell plot (b={b}, reconstructed reference)'
        parts.append(f'<section><h2>{escape(title_sec)}</h2>')
        uri = png_to_data_uri(png_path)
        if uri:
            parts.append(f'<figure><img src="{uri}" alt="{escape(title_sec)}">')
            parts.append(f'<figcaption class="caption">{escape(basename(png_path))}</figcaption></figure>')
        else:
            parts.append(f'<p class="warn">Image not found (skipped): <code>{escape(png_path)}</code></p>')
        parts.append('</section>')

    parts.append('</body></html>')
    return '\n'.join(parts)


def write_recon_csv(path: str, rows: list[dict]) -> None:
    fieldnames = [
        'b',
        'ref_recon_mean',
        'ref_recon_std',
        'ref_recon_n',
        'tar_mean',
        'tar_std',
        'tar_n',
        'harm_mean',
        'harm_std',
        'harm_n',
    ]

    def _clean(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return ''
        return v

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            if r.get('skip'):
                continue
            w.writerow({k: _clean(r.get(k, '')) for k in fieldnames})


def main():
    here = dirname(abspath(__file__))
    CSS, DEFAULT_BSHELLS_D, mean_fa_per_subject, png_to_data_uri, summarize = _import_dashboard_helpers()

    parser = argparse.ArgumentParser(
        description='Run debug_FA_modified per b-shell with reconstructed reference; build recon-only dashboard.',
    )
    parser.add_argument(
        '--debug-fa',
        default=pjoin(here, 'debug_FA_modified.py'),
        help='Path to debug_FA_modified.py (default: alongside this script)',
    )
    parser.add_argument('--templatePath', required=True)
    parser.add_argument('--refSite', required=True)
    parser.add_argument('--targetSite', required=True)
    parser.add_argument('--ref-pattern', required=True, help='Caselist path with {bshell} placeholder')
    parser.add_argument('--tar-unproc-pattern', required=True)
    parser.add_argument('--tar-harm-pattern', required=True)
    parser.add_argument('--mniTmp', required=True)
    parser.add_argument('--bshells', default=','.join(str(x) for x in DEFAULT_BSHELLS_D))
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Pass --plot-only --ref-reconstructed to debug_FA_modified (no registration)',
    )
    parser.add_argument(
        '--skip-ref-register',
        action='store_true',
        help='Do not pass --register-ref-reconstructed (use existing reconstructed_*_InMNI_*; add --ref-reconstructed)',
    )
    parser.add_argument(
        '--skip-harmonized-register',
        action='store_true',
        help='Do not pass --register (only warp ref recon, or stats-only if combined with --skip-ref-register)',
    )
    parser.add_argument('--force', action='store_true', help='Pass --force to debug_FA_modified')
    parser.add_argument(
        '--skip-debug-fa',
        action='store_true',
        help='Do not run debug_FA_modified; only build dashboard/CSV from existing template outputs',
    )
    parser.add_argument(
        '--out-html',
        default=None,
        help='Output HTML (default: <templatePath>/fa_dashboard_recon_only.html)',
    )
    parser.add_argument(
        'extra',
        nargs='*',
        help='Extra arguments forwarded to debug_FA_modified.py',
    )

    args = parser.parse_args()
    bshells = [int(x.strip()) for x in args.bshells.split(',') if x.strip()]

    if args.plot_only and (args.skip_ref_register or args.skip_harmonized_register):
        print('Note: --plot-only implies no registration; --skip-* flags ignored.', file=sys.stderr)

    if not exists(args.debug_fa):
        print(f'ERROR: debug_FA_modified not found: {args.debug_fa}', file=sys.stderr)
        sys.exit(1)

    template_path = abspath(args.templatePath)
    out_html = abspath(args.out_html) if args.out_html else pjoin(template_path, 'fa_dashboard_recon_only.html')
    out_csv = pjoin(dirname(out_html), 'fa_dashboard_recon_only_stats.csv')

    if not args.skip_debug_fa:
        extra = list(args.extra)
        for b in bshells:
            ref_csv = args.ref_pattern.format(bshell=b)
            tar_u = args.tar_unproc_pattern.format(bshell=b)
            tar_h = args.tar_harm_pattern.format(bshell=b)
            for p, label in ((ref_csv, 'ref'), (tar_u, 'tar_unproc'), (tar_h, 'tar_harm')):
                if not exists(abspath(p)):
                    print(f'ERROR: Missing {label} caselist for b={b}: {p}', file=sys.stderr)
                    sys.exit(1)

            run_debug_fa_one(
                debug_fa_py=args.debug_fa,
                template_path=template_path,
                ref_site=args.refSite,
                target_site=args.targetSite,
                ref_csv=ref_csv,
                tar_unproc_csv=tar_u,
                tar_harm_csv=tar_h,
                mni_tmp=args.mniTmp,
                bshell=b,
                plot_only=args.plot_only,
                skip_ref_register=args.skip_ref_register and not args.plot_only,
                skip_harm_register=args.skip_harmonized_register and not args.plot_only,
                force=args.force,
                extra_args=extra,
            )

    rows = collect_stat_rows(
        template_path,
        args.mniTmp,
        bshells,
        args.ref_pattern,
        args.tar_unproc_pattern,
        args.tar_harm_pattern,
        mean_fa_per_subject,
        summarize,
    )
    write_recon_csv(out_csv, rows)
    print(f'Wrote CSV: {out_csv}')

    title = f'mean FA (reconstructed ref only) — {args.refSite} vs {args.targetSite}'
    html = build_recon_only_html(
        title=title,
        ref_site=args.refSite,
        target_site=args.targetSite,
        template_path=template_path,
        rows=rows,
        bshells=bshells,
        css=CSS,
        png_to_data_uri=png_to_data_uri,
    )
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Wrote HTML: {out_html}')


if __name__ == '__main__':
    main()
