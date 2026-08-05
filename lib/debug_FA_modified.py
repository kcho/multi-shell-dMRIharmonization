# Work around MKL/OpenMP symbol errors on some Linux+conda setups when matplotlib exits.
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ.setdefault('OMP_NUM_THREADS', '1')

# ===============================================================================
# dMRIharmonization (2018) pipeline is written by-
#
# TASHRIF BILLAH
# Brigham and Women's Hospital/Harvard Medical School
# tbillah@bwh.harvard.edu, tashrifbillah@gmail.com
#
# ===============================================================================
# See details at https://github.com/pnlbwh/dMRIharmonization
# Submit issues at https://github.com/pnlbwh/dMRIharmonization/issues
# View LICENSE at https://github.com/pnlbwh/dMRIharmonization/blob/master/LICENSE
# ===============================================================================
#
# Modified:
# - Optional reconstructed reference: dtifit on reconstructed_{prefix}.nii.gz, then
#   the same two-step warp as harmonized (template construction warps + template0,
#   then TemplateToMNI_{refSite}). No unprocessed-target registration here.
# - Harmonized: uses existing dti/harmonized_*_{DM}.nii.gz (no dtifit here), then
#   native→template0 via warp_bands transforms, then TemplateToMNI_{target}.
# - Original reference _InMNI_* from upstream debug_fa still supported when not using
#   --ref-reconstructed for stats.
# ===============================================================================

import multiprocessing
import numpy as np
import glob
import subprocess
import sys
from subprocess import Popen
from os.path import dirname, basename, abspath, join as pjoin, exists
from os import makedirs
from nibabel import load


def _run_ants_apply_transforms(moving, output, reference, transform_paths):
    """Run ``antsApplyTransforms`` with one ``-t`` per path (no plumbum dependency)."""
    cmd = ['antsApplyTransforms', '-d', '3', '-i', moving, '-o', output, '-r', reference]
    for tp in transform_paths:
        cmd.extend(['-t', tp])
    # Multiprocessing workers can inherit invalid stdio after fork; EBADF on exec otherwise.
    subprocess.check_call(cmd, stdin=subprocess.DEVNULL, start_new_session=True)


def read_caselist_csv(caselist_file):
    """Read a csv/txt caselist with lines of: dwipath,maskpath"""
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


def antsReg(img, mask, mov, outPrefix):
    """Run ANTs registration."""
    if mask:
        p = Popen((' ').join(['antsRegistrationSyNQuick.sh',
                              '-d', '3',
                              '-f', img,
                              '-x', mask,
                              '-m', mov,
                              '-o', outPrefix,
                              '-e', '123456']), shell=True, stdout=sys.stdout, stderr=sys.stdout)
        p.wait()
    else:
        p = Popen((' ').join(['antsRegistrationSyNQuick.sh',
                              '-d', '3',
                              '-f', img,
                              '-m', mov,
                              '-o', outPrefix,
                              '-e', '123456']), shell=True, stdout=sys.stdout, stderr=sys.stdout)
        p.wait()


def dti_fit(imgPath, maskPath, force=False):
    """Run FSL dtifit on a DWI image, saving FA, MD etc. in dti/ subfolder."""

    directory = dirname(imgPath)
    inPrefix = imgPath.split('.nii')[0]
    prefix = basename(inPrefix)

    dtiDir = pjoin(directory, 'dti')
    makedirs(dtiDir, exist_ok=True)

    outPrefix = pjoin(dtiDir, prefix)
    faOut = outPrefix + '_FA.nii.gz'

    if exists(faOut) and not force:
        print(f'  DTI maps already exist for {prefix}, skipping dtifit')
        return

    bvalFile = inPrefix + '.bval'
    bvecFile = inPrefix + '.bvec'

    print(f'  Running FSL dtifit on {basename(imgPath)} ...')
    subprocess.check_call(
        ['dtifit', '-k', imgPath, '-o', outPrefix, '-m', maskPath, '-r', bvecFile, '-b', bvalFile],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _find_template_construction_warps(templatePath, transPrefix):
    """Same glob pattern as buildTemplate.warp_bands (exclude *_FA_ToMNI* warps)."""
    pattern_warp = pjoin(templatePath, transPrefix + '*_FA*[!ToMNI]1Warp.nii.gz')
    pattern_affine = pjoin(templatePath, transPrefix + '*_FA*[!ToMNI]0GenericAffine.mat')
    warps = sorted(glob.glob(pattern_warp))
    affines = sorted(glob.glob(pattern_affine))
    return warps, affines


def _map_cases_parallel(func, cases, N_proc, static_args):
    """Run ``func(imgPath, maskPath, *static_args)`` per caselist row.

    If ``N_proc <= 1``, runs in the main process so tracebacks are immediate.
    Otherwise uses a pool and calls ``AsyncResult.get()`` so worker exceptions propagate.
    """
    if N_proc <= 1:
        for imgPath, maskPath in cases:
            func(imgPath, maskPath, *static_args)
        return
    pool = multiprocessing.Pool(N_proc)
    results = []
    for imgPath, maskPath in cases:
        results.append(pool.apply_async(func, args=(imgPath, maskPath, *static_args)))
    pool.close()
    pool.join()
    for r in results:
        r.get()


def process_reconstructed_reference(imgPath, maskPath, templatePath, mniTmp,
                                    warp2mni, trans2mni, diffusionMeasures, bshell_b, force=False):
    """Reference subject (reconstructed DWI): dtifit, then same two-step warp as harmonized.

    Expects ``{directory}/reconstructed_{prefix}.nii.gz`` (from pipeline ``approx``).
    Uses ``transPrefix`` from the original DWI basename (same subject as template build)
    to find warp_bands transforms; native→template0, then TemplateToMNI_{refSite}.
    """

    directory = dirname(imgPath)
    inPrefix = imgPath.split('.nii')[0]
    prefix = basename(inPrefix)

    recon_dwi = pjoin(directory, 'reconstructed_' + prefix + '.nii.gz')
    if not exists(recon_dwi):
        print(f'  WARNING: {recon_dwi} does not exist, skipping')
        return

    recon_prefix = 'reconstructed_' + prefix
    trans_prefix = prefix.replace(f'_b{bshell_b}', '')

    template0 = pjoin(templatePath, 'template0.nii.gz')
    if not exists(template0):
        print(f'  WARNING: {template0} not found, skipping reconstructed reference')
        return

    warps, affines = _find_template_construction_warps(templatePath, trans_prefix)
    if not warps or not affines:
        print(f'  WARNING: No template construction warps for transPrefix={trans_prefix!r} in {templatePath}')
        return
    if len(warps) > 1 or len(affines) > 1:
        print(f'  WARNING: Multiple warp/affine matches; using first of {len(warps)}/{len(affines)}')
    warp = warps[0]
    trans = affines[0]

    dti_fit(recon_dwi, maskPath, force=force)

    for dm in diffusionMeasures:
        output = pjoin(templatePath, recon_prefix + f'_InMNI_{dm}.nii.gz')

        if exists(output) and not force:
            print(f'  {basename(output)} already exists, skipping')
            continue

        moving_native = pjoin(directory, 'dti', recon_prefix + f'_{dm}.nii.gz')
        if not exists(moving_native):
            print(f'  WARNING: {moving_native} does not exist, skipping')
            continue

        to_template = pjoin(templatePath, recon_prefix + f'_HarmOnTemplate_{dm}.nii.gz')
        print(f'  Warping {recon_prefix} {dm} native→template (same warps as warp_bands)')
        _run_ants_apply_transforms(moving_native, to_template, template0, [warp, trans])

        print(f'  Warping {recon_prefix} {dm} template→MNI')
        _run_ants_apply_transforms(to_template, output, mniTmp, [warp2mni, trans2mni])


def run_reconstructed_reference_registration(templatePath, refSite, caselist, mniTmp,
                                             diffusionMeasures, bshell_b, N_proc, force=False):
    """Ensure TemplateToMNI_{refSite} exists, then register reconstructed reference subjects."""

    moving = pjoin(templatePath, f'Mean_{refSite}_FA_b{bshell_b}.nii.gz')
    out_prefix = pjoin(templatePath, f'TemplateToMNI_{refSite}')
    warp2mni = out_prefix + '1Warp.nii.gz'
    trans2mni = out_prefix + '0GenericAffine.mat'

    if not exists(warp2mni):
        print(f'  Registering {refSite} template mean FA to MNI ...')
        antsReg(mniTmp, None, moving, out_prefix)

    cases = read_caselist_csv(caselist)
    static_args = (templatePath, mniTmp, warp2mni, trans2mni, diffusionMeasures, bshell_b, force)
    _map_cases_parallel(process_reconstructed_reference, cases, N_proc, static_args)


def process_harmonized(imgPath, _maskPath, templatePath, mniTmp,
                       warp2mni, trans2mni, diffusionMeasures, bshell_b, force=False):
    """Harmonized subject: two-step registration only (no dtifit).

    Expects existing ``dti/{prefix}_{DM}.nii.gz`` (e.g. from prior dtifit on harmonized DWI).
    Warps those maps with the same subject-specific transforms as buildTemplate.warp_bands:
    glob(transPrefix + '*_FA*[!ToMNI]1Warp.nii.gz') and affine, ``-r`` template0.nii.gz,
    then TemplateToMNI_{target}.

    Harmonized DWI basename should be ``harmonized_<origPrefix>`` so transPrefix matches
    template warps for that subject.
    """

    directory = dirname(imgPath)
    inPrefix = imgPath.split('.nii')[0]
    prefix = basename(inPrefix)

    orig_prefix = prefix.replace('harmonized_', '', 1) if prefix.startswith('harmonized_') else prefix
    trans_prefix = orig_prefix.replace(f'_b{bshell_b}', '')

    template0 = pjoin(templatePath, 'template0.nii.gz')
    if not exists(template0):
        print(f'  WARNING: {template0} not found, cannot warp harmonized data to template')
        return

    warps, affines = _find_template_construction_warps(templatePath, trans_prefix)
    if not warps or not affines:
        print(f'  WARNING: No template construction warps for transPrefix={trans_prefix!r} in {templatePath}')
        print(f'  Expected files like: {trans_prefix}*_FA*[!ToMNI]1Warp.nii.gz')
        return
    if len(warps) > 1 or len(affines) > 1:
        print(f'  WARNING: Multiple warp/affine matches; using first of {len(warps)}/{len(affines)}')
    warp = warps[0]
    trans = affines[0]

    for dm in diffusionMeasures:
        output = pjoin(templatePath, prefix + f'_InMNI_{dm}.nii.gz')

        if exists(output) and not force:
            print(f'  {basename(output)} already exists, skipping')
            continue

        moving_native = pjoin(directory, 'dti', prefix + f'_{dm}.nii.gz')
        if not exists(moving_native):
            print(f'  WARNING: {moving_native} does not exist, skipping')
            continue

        to_template = pjoin(templatePath, prefix + f'_HarmOnTemplate_{dm}.nii.gz')
        print(f'  Warping {prefix} {dm} native→template (same warps as warp_bands)')
        _run_ants_apply_transforms(moving_native, to_template, template0, [warp, trans])

        print(f'  Warping {prefix} {dm} template→MNI')
        _run_ants_apply_transforms(to_template, output, mniTmp, [warp2mni, trans2mni])


def run_harmonized_registration(templatePath, targetSite, caselist, mniTmp, diffusionMeasures,
                                 bshell_b, N_proc, force=False):
    """Ensure TemplateToMNI_{targetSite} exists, then two-step warp of existing harmonized DTI maps to MNI."""

    moving = pjoin(templatePath, f'Mean_{targetSite}_FA_b{bshell_b}.nii.gz')
    out_prefix = pjoin(templatePath, f'TemplateToMNI_{targetSite}')
    warp2mni = out_prefix + '1Warp.nii.gz'
    trans2mni = out_prefix + '0GenericAffine.mat'

    if not exists(warp2mni):
        print(f'  Registering {targetSite} template mean FA to MNI ...')
        antsReg(mniTmp, None, moving, out_prefix)

    cases = read_caselist_csv(caselist)
    static_args = (templatePath, mniTmp, warp2mni, trans2mni, diffusionMeasures, bshell_b, force)
    _map_cases_parallel(process_harmonized, cases, N_proc, static_args)


def analyzeStat(file, templatePath, mniTmp, reconstructed_ref=False):
    """Compute mean FA over IIT skeleton for each subject.

    If ``reconstructed_ref`` is True (reference caselist only), looks for
    ``reconstructed_{prefix}_InMNI_FA.nii.gz`` in templatePath.
    """

    skel_path = pjoin(dirname(mniTmp), 'IITmean_FA_skeleton.nii.gz')
    skel = load(skel_path)
    skel_mask = (skel.get_fdata() > 0) * 1.

    cases = read_caselist_csv(file)

    recon = 'reconstructed_' if reconstructed_ref else ''

    mean_attr = []
    for imgPath, maskPath in cases:
        in_prefix = imgPath.split('.nii')[0]
        prefix = basename(in_prefix)

        fa_img = pjoin(templatePath, recon + prefix + f'_InMNI_FA.nii.gz')
        if not exists(fa_img):
            print(f'  WARNING: {basename(fa_img)} does not exist, skipping')
            continue
        data = load(fa_img).get_fdata()
        temp = data * skel_mask
        mean_attr.append(temp[temp > 0].mean())

    return mean_attr


def plot_meanFA(ref_meanFA, tar_meanFA, harm_meanFA, refSite, targetSite, outPath=None,
                ref_reconstructed=False, ref_label=None):
    """Plot meanFA comparison before and after harmonization.

    Cyan diamonds = mean of the per-subject skeleton means (same as printed stats).
    Black error bars = std of those subject-level values (population std, ``ddof=0``).
    """

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if ref_label is None:
        ref_label = f'{refSite} (recon)' if ref_reconstructed else str(refSite)

    ref_tick = f'{refSite}\n(recon)' if ref_reconstructed else str(refSite)
    labels = [ref_tick, f'{targetSite}_before', f'{targetSite}_after']
    data = [ref_meanFA, tar_meanFA, harm_meanFA]
    # Distinct colors so reference (recon) is not confused with target groups
    point_colors = ['#1f77b4', '#d62728', '#2ca02c']

    fig, ax = plt.subplots(figsize=(11, 7))

    for i, (label, values) in enumerate(zip(labels, data)):
        values = np.asarray(values, dtype=float).ravel()
        if values.size == 0:
            continue
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        c = point_colors[i]

        ax.scatter([i] * len(values), values, marker='*', s=100, color=c, zorder=3, alpha=0.85)
        ax.scatter(i, mean_val, marker='D', s=72, color='cyan', edgecolors='black', zorder=5)
        ax.errorbar(i, mean_val, yerr=std_val, fmt='none', ecolor='black',
                    capsize=10, capthick=2, elinewidth=2, zorder=2)
        # Same mean as printed to console (avoids eyeball vs diamond mismatch)
        ax.annotate(
            f'{mean_val:.4f}',
            xy=(i, mean_val),
            xytext=(0, 12),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            color='black',
            zorder=6,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('meanFA over IIT_mean_FA_skeleton', fontsize=12)
    title = 'Comparison of meanFA before and after harmonization'
    if ref_reconstructed:
        title += ' (reference: reconstructed DWI)'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, len(labels) - 0.5)

    # Footer: exact same summary lines as printed to stdout (so plot matches log)
    foot_lines = []
    rv = np.asarray(ref_meanFA, dtype=float).ravel()
    if rv.size:
        foot_lines.append(
            f'Reference ({ref_label}): mean={np.mean(rv):.4f}, std={np.std(rv):.4f}, N={rv.size}'
        )
    tv = np.asarray(tar_meanFA, dtype=float).ravel()
    if tv.size:
        foot_lines.append(
            f'Target before ({targetSite}): mean={np.mean(tv):.4f}, std={np.std(tv):.4f}, N={tv.size}'
        )
    hv = np.asarray(harm_meanFA, dtype=float).ravel()
    if hv.size:
        foot_lines.append(
            f'Target after ({targetSite}): mean={np.mean(hv):.4f}, std={np.std(hv):.4f}, N={hv.size}'
        )
    footer = '\n'.join(foot_lines)
    fig.text(0.5, 0.02, footer, ha='center', va='bottom', fontsize=8, family='monospace',
             transform=fig.transFigure)

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    if outPath:
        fig.savefig(outPath, dpi=150, bbox_inches='tight', pad_inches=0.25)
        plt.close(fig)
        print(f'Plot saved to {outPath}')
    else:
        plt.show()
        plt.close(fig)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='Debug FA: optional reconstructed-ref and/or harmonized registration to MNI; '
                    'stats read *_InMNI_* (target-before from upstream).')

    parser.add_argument('--templatePath', required=True, help='Path to template directory')
    parser.add_argument('--refSite', required=True, help='Reference site name')
    parser.add_argument('--targetSite', required=True, help='Target site name')
    parser.add_argument('--ref', required=True, help='Reference caselist CSV (dwipath,maskpath)')
    parser.add_argument('--tar_unproc', required=True, help='Target unprocessed caselist CSV (dwipath,maskpath)')
    parser.add_argument('--tar_harm', required=True, help='Target harmonized caselist CSV (dwipath,maskpath)')
    parser.add_argument('--bshell_b', type=int, required=True, help='b-value of the shell (e.g. 1000, 2000, 3000)')
    parser.add_argument('--diffusionMeasures', default='FA,MD', help='Comma-separated diffusion measures (default: FA,MD)')
    parser.add_argument('--nproc', type=int, default=4, help='Number of parallel processes (default: 4)')
    parser.add_argument('--mniTmp', default=None, help='Path to IITmean_FA.nii.gz')
    parser.add_argument('--register-ref-reconstructed', action='store_true',
                        help='Register SH-reconstructed reference: dtifit on reconstructed_* DWI, '
                             'then native→template0 (warp_bands warps) → TemplateToMNI_ref')
    parser.add_argument('--register', action='store_true',
                        help='Warp existing harmonized dti/* maps to MNI (two-step: template warps + TemplateToMNI_target); does not run dtifit')
    parser.add_argument('--ref-reconstructed', action='store_true',
                        help='Reference stats/plot use reconstructed_<prefix>_InMNI_FA.nii.gz (SH-reconstructed FA). '
                             'If omitted (and no --register-ref-reconstructed for stats), uses original '
                             '<prefix>_InMNI_FA.nii.gz from upstream debug_fa / template warps.')
    parser.add_argument('--ref-original', action='store_true',
                        help='Force reference stats/plot to use original <prefix>_InMNI_FA.nii.gz (not reconstructed). '
                             'Useful with --plot-only. Incompatible with --ref-reconstructed.')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only read existing *_InMNI_* in templatePath and plot (no dtifit, no ANTs). '
                             'Ignores --register and --register-ref-reconstructed.')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files (registration steps only)')

    args = parser.parse_args()

    if args.ref_original and args.ref_reconstructed:
        parser.error('Use only one of --ref-original and --ref-reconstructed.')

    # Decide reference stats *before* --plot-only clears registration flags (otherwise
    # --register-ref-reconstructed would be lost and we'd read wrong *_InMNI_* names).
    ref_reconstructed_for_stats = (False if args.ref_original
                                   else (args.ref_reconstructed or args.register_ref_reconstructed))

    requested_register = args.register
    requested_register_ref_recon = args.register_ref_reconstructed

    if args.plot_only:
        if requested_register or requested_register_ref_recon:
            print('Note: --plot-only set — skipping registration (--register / --register-ref-reconstructed ignored).\n')
        args.register = False
        args.register_ref_reconstructed = False

    print(
        '\n--- debug_FA_modified run plan ---\n'
        f"  --plot-only:                  {bool(args.plot_only)}  "
        '(only stats + figure from existing *_InMNI_*; no processing)\n'
        f"  Registration requested:       ref-recon={requested_register_ref_recon}, harmonized={requested_register}  "
        '(ignored if --plot-only)\n'
        f"  Reference stats (recon files): {ref_reconstructed_for_stats}  "
        '(reads reconstructed_<prefix>_InMNI_FA if True)\n'
        f"  --ref-original:               {bool(args.ref_original)}  "
        '(if True: force <prefix>_InMNI_FA for reference)\n'
        f"  --ref-reconstructed flag:    {bool(args.ref_reconstructed)}\n"
        f"  Reference filename pattern:   {'reconstructed_<prefix>_InMNI_FA' if ref_reconstructed_for_stats else '<prefix>_InMNI_FA'} (under templatePath)\n"
        '  With --plot-only: no dtifit/ants; stats still honor reconstructed reference if requested above.\n'
        '----------------------------------\n'
    )

    if args.mniTmp:
        mni_tmp = args.mniTmp
    else:
        script_dir = dirname(abspath(__file__))
        root_dir = abspath(pjoin(script_dir, '..'))
        mni_tmp = pjoin(root_dir, 'IITAtlas', 'IITmean_FA.nii.gz')

    diffusion_measures = [x.strip() for x in args.diffusionMeasures.split(',')]
    bshell_b = args.bshell_b

    # Reconstructed reference → distinct filename for side-by-side comparison with original ref plot
    if ref_reconstructed_for_stats:
        out_plot = pjoin(args.templatePath, f'meanFAplot_b{bshell_b}_reconstr.png')
    else:
        out_plot = pjoin(args.templatePath, f'meanFAplot_b{bshell_b}.png')
    print(f'\nOutput figure: {out_plot}\n')

    if args.register_ref_reconstructed:
        print('=== Registering reconstructed reference (dtifit + template warps + TemplateToMNI) ===')
        run_reconstructed_reference_registration(
            args.templatePath, args.refSite, args.ref, mni_tmp,
            diffusion_measures, bshell_b, args.nproc, force=args.force,
        )

    if args.register:
        print('=== Registering harmonized target (template warps + TemplateToMNI) ===')
        run_harmonized_registration(
            args.templatePath, args.targetSite, args.tar_harm, mni_tmp,
            diffusion_measures, bshell_b, args.nproc, force=args.force,
        )

    print('=== Computing meanFA statistics ===')
    ref_mean_fa = analyzeStat(args.ref, args.templatePath, mni_tmp, reconstructed_ref=ref_reconstructed_for_stats)
    tar_mean_fa = analyzeStat(args.tar_unproc, args.templatePath, mni_tmp, reconstructed_ref=False)
    harm_mean_fa = analyzeStat(args.tar_harm, args.templatePath, mni_tmp, reconstructed_ref=False)

    if len(ref_mean_fa) == 0:
        if ref_reconstructed_for_stats:
            print('ERROR: No reconstructed_*_InMNI_FA.nii.gz files for reference. '
                  'Run with --register-ref-reconstructed first, or use --ref-reconstructed only after those exist.')
        else:
            print('ERROR: No reference _InMNI_FA.nii.gz files in templatePath. '
                  'Generate with upstream debug_fa (register_reference), or use --register-ref-reconstructed '
                  'and --ref-reconstructed.')
        sys.exit(1)

    ref_label = f'{args.refSite} (recon)' if ref_reconstructed_for_stats else args.refSite
    print(f'\nReference ({ref_label}): mean={np.mean(ref_mean_fa):.4f}, std={np.std(ref_mean_fa):.4f}, N={len(ref_mean_fa)}')
    print(f'Target before ({args.targetSite}): mean={np.mean(tar_mean_fa):.4f}, std={np.std(tar_mean_fa):.4f}, N={len(tar_mean_fa)}')
    print(f'Target after ({args.targetSite}): mean={np.mean(harm_mean_fa):.4f}, std={np.std(harm_mean_fa):.4f}, N={len(harm_mean_fa)}')

    if len(tar_mean_fa) == 0:
        print('\nNote: No target-before InMNI_FA files found. Generate them separately if you need the middle bar.')

    plot_meanFA(
        ref_mean_fa,
        tar_mean_fa,
        harm_mean_fa,
        args.refSite,
        args.targetSite,
        out_plot,
        ref_reconstructed=ref_reconstructed_for_stats,
        ref_label=ref_label,
    )
