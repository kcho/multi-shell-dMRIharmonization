#!/usr/bin/env python

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

from plumbum import cli
import multiprocessing
N_CPU= str(multiprocessing.cpu_count())
from conversion import read_bvals
from util import abspath, dirname, basename, pjoin, SCRIPTDIR, remove, isfile
from subprocess import check_call

from consistencyCheck import consistencyCheck

from separateBshells import separateAllBshells

from joinBshells import joinAllBshells

from fileUtil import check_dir


def separateShellsWrapper(csvFile, ref_bshell_file, N_proc, force=False):
    
    outPrefix= csvFile.with_suffix('')._path
    
    separateAllBshells(csvFile, ref_bshell_file, N_proc, outPrefix, force)

    return outPrefix


class multi_shell_pipeline(cli.Application):

    VERSION = 1.7

    ref_csv = cli.SwitchAttr(
        ['--ref_list'],
        cli.ExistingFile,
        help='reference csv/txt file with first column for dwi and 2nd column for mask: dwi1,mask1\\ndwi2,mask2\\n...',
        mandatory=False)

    target_csv = cli.SwitchAttr(
        ['--tar_list'],
        cli.ExistingFile,
        help='target csv/txt file with first column for dwi and 2nd column for mask: dwi1,mask1\\ndwi2,mask2\\n...',
        mandatory=False)

    templatePath = cli.SwitchAttr(
        ['--template'],
        help='template directory',
        mandatory=True)

    N_shm = cli.SwitchAttr(
        ['--nshm'],
        help='spherical harmonic order, by default maximum possible is used',
        default= '-1')

    N_proc = cli.SwitchAttr(
        '--nproc',
        help= 'number of processes/threads to use (-1 for all available, may slow down your system)',
        default= '4')

    N_zero = cli.SwitchAttr(
        '--nzero',
        help= 'number of zero padding for denoising skull region during signal reconstruction',
        default= '10')

    shells_to_ignore = cli.SwitchAttr(
        '--shells_to_ignore',
        help= 'b-shells to ignore during template creation and harmonization, e.g. --shells_to_ignore 2500,3000',
        default= '')
    
    force = cli.Flag(
        ['--force'],
        help='turn on this flag to overwrite existing data',
        default= False)

    travelHeads = cli.Flag(
        ['--travelHeads'],
        help='travelling heads',
        default= False)

    denoise= None
    bvalMap= None
    resample= None

    create = cli.Flag(
        '--create',
        help= 'turn on this flag to create template',
        default= False)

    process = cli.Flag(
        '--process',
        help= 'turn on this flag to harmonize',
        default= False)

    debug = cli.Flag(
        '--debug',
        help= 'turn on this flag to debug harmonized data (valid only with --process)',
        default= False)

    reference = cli.SwitchAttr(
        '--ref_name',
        help= 'reference site name',
        mandatory= False)

    target = cli.SwitchAttr(
        '--tar_name',
        help= 'target site name',
        mandatory= True)

    verbose= cli.Flag(
        '--verbose',
        help='print everything to STDOUT',
        default= False)


    def run_debug_batch(self, ref_list_prefix, tar_list_prefix, ref_bvals):

        mniTmp = pjoin(dirname(SCRIPTDIR), 'IITAtlas', 'IITmean_FA.nii.gz')
        if not isfile(mniTmp):
            raise FileNotFoundError(f'MNI template not found: {mniTmp}')

        bshells = [str(int(bval)) for bval in sorted(ref_bvals) if int(bval) != 0]
        if not bshells:
            raise ValueError('No non-zero b-shells available for post-harmonization debug batch')

        cmd = [
            pjoin(SCRIPTDIR, 'run_debug_fa_recon_batch.py'),
            '--templatePath', self.templatePath,
            '--refSite', self.reference,
            '--targetSite', self.target,
            '--ref-pattern', f'{ref_list_prefix}_b{{bshell}}.csv.modified',
            '--tar-unproc-pattern', f'{tar_list_prefix}_b{{bshell}}.csv.modified',
            '--tar-harm-pattern', f'{tar_list_prefix}_b{{bshell}}.csv.modified.harmonized',
            '--mniTmp', mniTmp,
            '--bshells', ','.join(bshells),
            '--force'
        ]

        if self.force:
            cmd.append('--force')

        print('## post-harmonization multi-shell debug batch ##')
        check_call(cmd)


    def main(self):

        self.templatePath = abspath(self.templatePath)

        if self.N_proc=='-1':
            self.N_proc= N_CPU

        # check directory existence 
        check_dir(self.templatePath, self.force)

        ## check consistency of b-shells and spatial resolution
        ref_bvals_file= pjoin(self.templatePath, 'ref_bshell_bvalues.txt')
        ref_res_file= pjoin(self.templatePath, 'ref_res_file.npy')
        if self.ref_csv:
            if isfile(ref_bvals_file) and isfile(ref_res_file):
                remove(ref_bvals_file)
                remove(ref_res_file)

            consistencyCheck(self.ref_csv, ref_bvals_file, ref_res_file, shells_to_ignore=self.shells_to_ignore)

        if self.target_csv:
            consistencyCheck(self.target_csv, ref_bvals_file, ref_res_file, shells_to_ignore=self.shells_to_ignore)

        ## ignore shells if specified
        if self.shells_to_ignore:
            shells_to_ignore_list = [int(bval) for bval in self.shells_to_ignore.split(',')]
            ref_bvals = read_bvals(ref_bvals_file)
            ref_bvals_filtered = [bval for bval in ref_bvals if int(bval) not in shells_to_ignore_list]
            with open(ref_bvals_file, 'w') as f:
                f.write(' '.join(map(str, ref_bvals_filtered)))
        
        ## separate b-shells
        if self.ref_csv:
            print(f'\n## separating b-shells: {self.reference} ##')
            refListOutPrefix= separateShellsWrapper(self.ref_csv, ref_bvals_file, self.N_proc, self.force)
        if self.target_csv:
            print(f'\n## separating b-shells: {self.target} ##')
            tarListOutPrefix= separateShellsWrapper(self.target_csv, ref_bvals_file, self.N_proc, self.force)


        ## define variables for template creation and data harmonization

        # variables common to all ref_bvals
        pipeline_vars=[
            '--tar_name', self.target,
            '--nshm', self.N_shm,
            '--nproc', self.N_proc,
            '--template', self.templatePath,
            ]
        
        if self.reference:
            pipeline_vars.append(f'--ref_name {self.reference}')
        if self.N_zero:
            pipeline_vars.append(f'--nzero {self.N_zero}')
        if self.bvalMap:
            pipeline_vars.append(f'--bvalMap {self.bvalMap}')
        if self.resample:
            pipeline_vars.append(f'--resample {self.resample}')
        if self.denoise:
            pipeline_vars.append('--denoise')
        if self.travelHeads:
            pipeline_vars.append('--travelHeads')
        if self.force:
            pipeline_vars.append('--force')
        if self.debug:
            pipeline_vars.append('--debug')
        if self.verbose:
            pipeline_vars.append('--verbose')
        

        # the b-shell bvalues are sorted in descending order because we want to perform registration with highest bval
        # because L0 maps from high b-shells would look too similar to FA, the b-shell bvalues are reordered to perform
        # the template construction with B=1000
        ref_bvals= read_bvals(ref_bvals_file)[::-1]

        target_bval = 1000

        ref_bvals_ordered = sorted(
            ref_bvals,
            key=lambda b: (b == 0, abs(b - target_bval), -b)
        )

        for bval in ref_bvals_ordered:
            if bval == 0:
                continue

            if self.create and not self.process:
                print('## template creation ##')

                check_call((' ').join([pjoin(SCRIPTDIR, 'harmonization.py'),
                '--tar_list', tarListOutPrefix+f'_b{int(bval)}.csv',
                '--bshell_b', str(int(bval)),
                '--ref_list', refListOutPrefix+f'_b{int(bval)}.csv',
                '--create'] + pipeline_vars), shell= True)



            elif not self.create and self.process:
                print(f'\n## data harmonization: b={int(bval)} ##')

                check_call((' ').join([pjoin(SCRIPTDIR, 'harmonization.py'),
                '--tar_list', tarListOutPrefix + f'_b{int(bval)}.csv',
                f'--ref_list {refListOutPrefix}_b{int(bval)}.csv' if self.ref_csv else '',
                '--bshell_b', str(int(bval)),
                '--process'] + pipeline_vars), shell=True)



            elif self.create and self.process:
                check_call((' ').join([pjoin(SCRIPTDIR, 'harmonization.py'),
                '--tar_list', tarListOutPrefix + f'_b{int(bval)}.csv',
                '--bshell_b', str(int(bval)),
                '--ref_list', refListOutPrefix+f'_b{int(bval)}.csv',
                '--create', '--process'] + pipeline_vars), shell=True)

                
            if '--force' in pipeline_vars:
                pipeline_vars.remove('--force')


        ## join harmonized data
        if self.process:
            print(f'\n## joining harmonized data: {self.target} ##')
            joinAllBshells(self.target_csv, ref_bvals_file, 'harmonized_', self.N_proc, self.force)
        
            if self.debug and self.ref_csv:
                print(f'\n## joining reconstructed reference: {self.reference} ##')
                joinAllBshells(self.ref_csv, ref_bvals_file, 'reconstructed_', self.N_proc, self.force)
                self.run_debug_batch(refListOutPrefix, tarListOutPrefix, ref_bvals)

if __name__== '__main__':
    multi_shell_pipeline.run()

