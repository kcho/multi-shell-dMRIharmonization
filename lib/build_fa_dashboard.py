#!/usr/bin/env python3
"""
Build a single HTML dashboard with embedded plots (base64) and FA statistics tables.

Reads the same *_InMNI_FA.nii.gz conventions as debug_FA_modified / plot_meanFA_multibshell*.py.
Optionally embeds PNGs if present in templatePath:

  - meanFA_multib_original_ref.png
  - meanFA_multib_reconstr_ref.png
  - meanFAplot_b{bshell}.png           (original ref single-shell)
  - meanFAplot_b{bshell}_reconstr.png (reconstructed ref single-shell)
  - any other PNGs passed via --extra-plots

Also writes fa_dashboard_stats.csv next to the HTML.

Example:
  python build_fa_dashboard.py \\
    --templatePath ./template/ \\
    --refSite ME_device_1_soft_1 \\
    --targetSite CG_device_1_soft_1 \\
    --ref-pattern harmonization_input_ref_b{bshell}.csv.modified \\
    --tar-unproc-pattern harmonization_input_target_b{bshell}.csv.modified \\
    --tar-harm-pattern harmonization_input_target_b{bshell}.csv.modified.harmonized \\
    --mniTmp /data/pnl/IITAtlas/IITmean_FA.nii.gz \\
    --out ./template/fa_dashboard.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import os
import sys
from datetime import datetime, timezone
from html import escape
from os.path import abspath, basename, dirname, exists, join as pjoin

import numpy as np
from nibabel import load

os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ.setdefault('OMP_NUM_THREADS', '1')

DEFAULT_BSHELLS = [200, 500, 1000, 2000, 3000]

CSS = """
:root { --bg: #f6f8fa; --card: #fff; --border: #d0d7de; --text: #24292f; --muted: #57606a; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 24px;
       background: var(--bg); color: var(--text); line-height: 1.5; }
h1 { font-size: 1.5rem; margin: 0 0 8px; }
h2 { font-size: 1.15rem; margin: 0 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.meta { color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }
section { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 12px 0; }
th, td { border: 1px solid var(--border); padding: 8px 10px; text-align: left; }
th { background: #f3f4f6; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
figure { margin: 0; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 4px; }
.caption { color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
.warn { color: #9a6700; background: #fff8c5; padding: 8px 12px; border-radius: 6px; margin: 8px 0; }
"""


def read_caselist_csv(caselist_file: str):
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


def _skel_mask(mni_tmp: str):
    skel_path = pjoin(dirname(mni_tmp), 'IITmean_FA_skeleton.nii.gz')
    skel = load(skel_path)
    return (skel.get_fdata() > 0) * 1.0


def mean_fa_per_subject(caselist_file: str, template_path: str, mni_tmp: str, reconstructed_ref: bool):
    skel_mask = _skel_mask(mni_tmp)
    prefix_recon = 'reconstructed_' if reconstructed_ref else ''
    cases = read_caselist_csv(caselist_file)
    out = []
    for img_path, _ in cases:
        in_prefix = img_path.split('.nii')[0]
        prefix = basename(in_prefix)
        fa_img = pjoin(template_path, prefix_recon + prefix + '_InMNI_FA.nii.gz')
        if not exists(fa_img):
            continue
        data = load(fa_img).get_fdata()
        temp = data * skel_mask
        out.append(float(temp[temp > 0].mean()))
    return out


def summarize(vals: list[float]) -> tuple[float, float, int]:
    if not vals:
        return float('nan'), float('nan'), 0
    a = np.asarray(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a)), int(a.size)


def png_to_data_uri(path: str) -> str | None:
    if not path or not exists(path):
        return None
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:image/png;base64,{b64}'


def _fmt_stat(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return '—'
    return f'{float(x):.{nd}f}'


def _fmt_n(n: int) -> str:
    return str(int(n)) if n else '—'


def build_html(
    title: str,
    ref_site: str,
    target_site: str,
    template_path: str,
    rows: list[dict],
    images: list[tuple[str, str]],
) -> str:
    """rows: list of stat dicts; images: (section_title, path_or_none)."""
    parts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        f'<title>{escape(title)}</title><style>{CSS}</style></head><body>',
        f'<h1>{escape(title)}</h1>',
        f'<p class="meta">Generated {escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))} · '
        f'ref: <strong>{escape(ref_site)}</strong> · target: <strong>{escape(target_site)}</strong> · '
        f'template: <code>{escape(template_path)}</code></p>',
    ]

    # Stats table
    parts.append('<section><h2>Statistics by b-shell</h2>')
    parts.append(
        '<table><thead><tr>'
        '<th>b</th>'
        '<th class="num">Ref (orig) mean</th><th class="num">Ref (orig) std</th><th class="num">N</th>'
        '<th class="num">Ref (recon) mean</th><th class="num">Ref (recon) std</th><th class="num">N</th>'
        '<th class="num">Target before mean</th><th class="num">std</th><th class="num">N</th>'
        '<th class="num">Target after mean</th><th class="num">std</th><th class="num">N</th>'
        '</tr></thead><tbody>'
    )
    for r in rows:
        if r.get('skip'):
            continue
        parts.append(
            '<tr>'
            f'<td>{r["b"]}</td>'
            f'<td class="num">{_fmt_stat(r["ref_orig_mean"])}</td><td class="num">{_fmt_stat(r["ref_orig_std"])}</td><td class="num">{_fmt_n(r["ref_orig_n"])}</td>'
            f'<td class="num">{_fmt_stat(r["ref_recon_mean"])}</td><td class="num">{_fmt_stat(r["ref_recon_std"])}</td><td class="num">{_fmt_n(r["ref_recon_n"])}</td>'
            f'<td class="num">{_fmt_stat(r["tar_mean"])}</td><td class="num">{_fmt_stat(r["tar_std"])}</td><td class="num">{_fmt_n(r["tar_n"])}</td>'
            f'<td class="num">{_fmt_stat(r["harm_mean"])}</td><td class="num">{_fmt_stat(r["harm_std"])}</td><td class="num">{_fmt_n(r["harm_n"])}</td>'
            '</tr>'
        )
    parts.append('</tbody></table>')
    parts.append(
        '<p class="caption">Per-subject means: mean FA over IIT skeleton for each subject; '
        'group mean/std/N are across subjects. Ref (orig) uses &lt;prefix&gt;_InMNI_FA; '
        'Ref (recon) uses reconstructed_&lt;prefix&gt;_InMNI_FA.</p>'
    )
    parts.append('</section>')

    for section_title, img_path in images:
        parts.append(f'<section><h2>{escape(section_title)}</h2>')
        uri = png_to_data_uri(img_path) if img_path else None
        if uri:
            parts.append(f'<figure><img src="{uri}" alt="{escape(section_title)}">')
            parts.append(f'<figcaption class="caption">{escape(basename(img_path))}</figcaption></figure>')
        else:
            parts.append(f'<p class="warn">Image not found (skipped): <code>{escape(img_path or "")}</code></p>')
        parts.append('</section>')

    parts.append('</body></html>')
    return '\n'.join(parts)


def write_csv_rows(path: str, rows: list[dict]):
    fieldnames = [
        'b', 'ref_orig_mean', 'ref_orig_std', 'ref_orig_n',
        'ref_recon_mean', 'ref_recon_std', 'ref_recon_n',
        'tar_mean', 'tar_std', 'tar_n',
        'harm_mean', 'harm_std', 'harm_n',
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
    parser = argparse.ArgumentParser(description='Build HTML FA dashboard + CSV stats')
    parser.add_argument('--templatePath', required=True)
    parser.add_argument('--refSite', required=True)
    parser.add_argument('--targetSite', required=True)
    parser.add_argument('--ref-pattern', required=True, help='With {bshell} placeholder')
    parser.add_argument('--tar-unproc-pattern', required=True)
    parser.add_argument('--tar-harm-pattern', required=True)
    parser.add_argument('--mniTmp', required=True)
    parser.add_argument('--bshells', default=','.join(str(x) for x in DEFAULT_BSHELLS))
    parser.add_argument('--out', default=None, help='Output HTML path (default: templatePath/fa_dashboard.html)')
    parser.add_argument(
        '--extra-plots',
        default='',
        help='Comma-separated extra PNG paths to append as sections (optional)',
    )
    args = parser.parse_args()

    template_path = abspath(args.templatePath)
    mni_tmp = abspath(args.mniTmp)
    bshells = [int(x.strip()) for x in args.bshells.split(',') if x.strip()]

    out_html = abspath(args.out) if args.out else pjoin(template_path, 'fa_dashboard.html')
    out_csv = pjoin(dirname(out_html), 'fa_dashboard_stats.csv')

    rows = []
    for b in bshells:
        ref_csv = abspath(args.ref_pattern.format(bshell=b))
        tar_csv = abspath(args.tar_unproc_pattern.format(bshell=b))
        harm_csv = abspath(args.tar_harm_pattern.format(bshell=b))

        row = {'b': b, 'skip': True}
        if not all(exists(p) for p in (ref_csv, tar_csv, harm_csv)):
            print(f'WARNING b={b}: missing caselist, skipping row')
            rows.append(row)
            continue

        v_orig = mean_fa_per_subject(ref_csv, template_path, mni_tmp, reconstructed_ref=False)
        v_recon = mean_fa_per_subject(ref_csv, template_path, mni_tmp, reconstructed_ref=True)
        v_tar = mean_fa_per_subject(tar_csv, template_path, mni_tmp, reconstructed_ref=False)
        v_harm = mean_fa_per_subject(harm_csv, template_path, mni_tmp, reconstructed_ref=False)

        mo, so, no = summarize(v_orig)
        mr, sr, nr = summarize(v_recon)
        mt, st, nt = summarize(v_tar)
        mh, sh, nh = summarize(v_harm)

        row = {
            'b': b,
            'skip': False,
            'ref_orig_mean': mo, 'ref_orig_std': so, 'ref_orig_n': no,
            'ref_recon_mean': mr, 'ref_recon_std': sr, 'ref_recon_n': nr,
            'tar_mean': mt, 'tar_std': st, 'tar_n': nt,
            'harm_mean': mh, 'harm_std': sh, 'harm_n': nh,
        }
        rows.append(row)

    write_csv_rows(out_csv, rows)
    print(f'Wrote CSV: {out_csv}')

    title = f'mean FA dashboard — {args.refSite} vs {args.targetSite}'

    images: list[tuple[str, str]] = [
        ('Multi-b: original reference', pjoin(template_path, 'meanFA_multib_original_ref.png')),
        ('Multi-b: reconstructed reference', pjoin(template_path, 'meanFA_multib_reconstr_ref.png')),
    ]
    for b in bshells:
        images.append(
            (f'Single-shell plot (b={b}, original ref)', pjoin(template_path, f'meanFAplot_b{b}.png'))
        )
        images.append(
            (f'Single-shell plot (b={b}, reconstructed ref)', pjoin(template_path, f'meanFAplot_b{b}_reconstr.png'))
        )

    if args.extra_plots.strip():
        for p in args.extra_plots.split(','):
            p = p.strip()
            if p:
                images.append((basename(p), abspath(p)))

    html = build_html(
        title=title,
        ref_site=args.refSite,
        target_site=args.targetSite,
        template_path=template_path,
        rows=rows,
        images=images,
    )

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Wrote HTML: {out_html}')
    print('Open the HTML file in a browser; plots are embedded (no external PNG links needed).')


if __name__ == '__main__':
    main()
