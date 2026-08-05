#!/usr/bin/env python3
"""
Build one HTML page that lists existing per-site ``fa_dashboard_recon_only.html`` dashboards
in a dropdown and shows the selected report in an iframe.

Example (your layout)::

  /data/predict2/home/kcho/tmp_for_suheyla/
    ME_device_1_soft_1_ME_device_2_soft_2/template/fa_dashboard_recon_only.html
    ...

  python combine_fa_recon_dashboards.py \\
    --scan-root /data/predict2/home/kcho/tmp_for_suheyla \\
    --out /data/predict2/home/kcho/tmp_for_suheyla/fa_dashboard_recon_only_combined.html

Open ``fa_dashboard_recon_only_combined.html`` in a browser (from that directory or via a
local web server). Iframes use relative URLs to each site's HTML.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from html import escape
from os.path import abspath, basename, dirname, exists, isdir, join as pjoin, relpath


DEFAULT_SITE_FOLDERS = [
    'ME_device_1_soft_1_ME_device_2_soft_2',
    'ME_device_1_soft_1_SG_device_1_soft_1',
    'ME_device_1_soft_1_OR_device_1_soft_2',
    'ME_device_1_soft_1_SI_device_1_soft_1',
    'ME_device_1_soft_1_CG_device_1_soft_1',
    'ME_device_1_soft_1_PA_device_1_soft_1',
    'ME_device_1_soft_1_TE_device_1_soft_1',
    'ME_device_1_soft_1_MA_device_1_soft_1',
    'ME_device_1_soft_1_SD_device_1_soft_1',
]

DASHBOARD_RELPATH = pjoin('template', 'fa_dashboard_recon_only.html')


def _discover(scan_root: str, pattern: str) -> list[str]:
    scan_root = abspath(scan_root)
    found = []
    for path in sorted(glob.glob(pjoin(scan_root, pattern))):
        if isdir(path):
            found.append(basename(path))
    return found


def build_combined_html(
    *,
    title: str,
    entries: list[tuple[str, str]],
) -> str:
    """entries: (label, relative_url_to_dashboard_html)"""
    opts = ['<option value="">— Select a site pair —</option>']
    for label, url in entries:
        opts.append(f'<option value="{escape(url)}">{escape(label)}</option>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ --border: #d0d7de; --bg: #f6f8fa; --text: #24292f; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
            padding: 16px 20px; background: var(--bg); color: var(--text); }}
    h1 {{ font-size: 1.25rem; margin: 0 0 12px; }}
    .bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 12px; }}
    label {{ font-weight: 600; }}
    select {{ font-size: 0.95rem; padding: 8px 12px; min-width: min(100%, 36rem);
              max-width: 100%; border: 1px solid var(--border); border-radius: 6px; background: #fff; }}
    .hint {{ font-size: 0.85rem; color: #57606a; margin: 0 0 12px; }}
    iframe {{ width: 100%; min-height: calc(100vh - 140px); border: 1px solid var(--border);
              border-radius: 8px; background: #fff; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="hint">Each option loads that folder&rsquo;s <code>template/fa_dashboard_recon_only.html</code> below.</p>
  <div class="bar">
    <label for="site-pick">Site pair</label>
    <select id="site-pick" aria-label="Choose site pair dashboard">
{chr(10).join('      ' + o for o in opts)}
    </select>
  </div>
  <iframe id="dash-frame" title="Selected dashboard" sandbox="allow-same-origin allow-scripts"></iframe>
  <script>
    (function () {{
      var sel = document.getElementById('site-pick');
      var frame = document.getElementById('dash-frame');
      sel.addEventListener('change', function () {{
        frame.src = sel.value || 'about:blank';
      }});
    }})();
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='Combine per-site fa_dashboard_recon_only.html into one page with a dropdown')
    parser.add_argument('--scan-root', required=True, help='Parent directory containing ME_device_1_soft_1_* folders')
    parser.add_argument(
        '--out',
        default=None,
        help='Output combined HTML path (default: <scan-root>/fa_dashboard_recon_only_combined.html)',
    )
    parser.add_argument(
        '--sites',
        default='',
        help='Comma-separated folder names under scan-root (default: built-in list of 9 pairs)',
    )
    parser.add_argument(
        '--discover-glob',
        default='',
        help='If set, discover folder names with glob under scan-root (e.g. ME_device_1_soft_1_*); overrides --sites when used',
    )
    parser.add_argument('--title', default='FA dashboards (reconstructed reference only) — all site pairs')
    args = parser.parse_args()

    scan_root = abspath(args.scan_root)
    if not isdir(scan_root):
        print(f'ERROR: not a directory: {scan_root}', file=sys.stderr)
        sys.exit(1)

    out_path = abspath(args.out) if args.out else pjoin(scan_root, 'fa_dashboard_recon_only_combined.html')
    out_dir = dirname(out_path)

    if args.discover_glob.strip():
        names = _discover(scan_root, args.discover_glob.strip())
        if not names:
            print(f'ERROR: no directories matched {args.discover_glob!r} under {scan_root}', file=sys.stderr)
            sys.exit(1)
    elif args.sites.strip():
        names = [s.strip() for s in args.sites.split(',') if s.strip()]
    else:
        names = list(DEFAULT_SITE_FOLDERS)

    entries: list[tuple[str, str]] = []
    missing = []
    for name in names:
        folder = pjoin(scan_root, name)
        dash = pjoin(folder, DASHBOARD_RELPATH)
        if not exists(dash):
            missing.append(dash)
            continue
        rel = relpath(dash, out_dir).replace(os.sep, '/')
        entries.append((name, rel))

    if not entries:
        print('ERROR: no fa_dashboard_recon_only.html found for any listed site folder.', file=sys.stderr)
        if missing:
            print('Missing files:', file=sys.stderr)
            for m in missing[:20]:
                print(f'  {m}', file=sys.stderr)
            if len(missing) > 20:
                print(f'  ... and {len(missing) - 20} more', file=sys.stderr)
        sys.exit(1)

    if missing:
        print('WARNING: skipped folders without dashboard:', file=sys.stderr)
        for m in missing:
            print(f'  {m}', file=sys.stderr)

    html = build_combined_html(title=args.title, entries=entries)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Wrote {out_path} ({len(entries)} site pair(s))')
    print('Open this file in a browser from the same machine/path so relative iframe URLs resolve.')


if __name__ == '__main__':
    main()
