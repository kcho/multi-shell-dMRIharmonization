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

from conversion import read_bvals, read_imgs, read_imgs_masks
import numpy as np
from warnings import warn
from plumbum import local
from util import abspath, load, isfile, getpid
from findBshells import findBShells
from logging_config import log_info
import sys


def check_bshells(ref_imgs, ref_bvals):

    unmatched=[]
    for imgPath in ref_imgs:

        imgPath= local.path(imgPath)
        if not imgPath.exists():
            FileNotFoundError(imgPath)

        inPrefix = abspath(imgPath).split('.nii')[0]
        bvals= findBShells(inPrefix+'.bval')

        if (bvals==ref_bvals).all():
            log_info('b-shells matched for', imgPath.name)

        else:
            log_info(f'\nUnmatched b-shells for {imgPath.name}')
            log_info(bvals)
            log_info(f'ref_bvals {ref_bvals}\n')
            unmatched.append(imgPath._path)

    log_info('')
    if len(unmatched):
        log_info('Unmatched cases:')
        log_info(unmatched)
        raise ValueError('Leave out the unmatched cases or change the reference case for determining b-shell to run multi-shell-dMRIharmonization')

    else:
        log_info('All cases have same b-shells. Data is good for running multi-shell-dMRIharmonization')
    log_info('')



def check_resolution(ref_imgs, ref_res):

    unmatched = []
    for imgPath in ref_imgs:

        imgPath= local.path(imgPath)
        if not imgPath.exists():
            FileNotFoundError(imgPath)

        res= load(imgPath._path).header['pixdim'][1:4]

        if (res-ref_res).sum()<=10e-6:
            log_info('spatial resolution matched for', imgPath.name)

        else:
            log_info(f'\nUnmatched spatial resolution for {imgPath.name}')
            log_info(res)
            log_info(f'ref_res {ref_res}\n')
            unmatched.append(imgPath._path)

    log_info('')
    if len(unmatched):
        log_info('Unmatched cases:')
        log_info(unmatched)
        raise ValueError('Leave out the unmatched cases or change the reference case for determining spatial resolution to run multi-shell-dMRIharmonization')

    else:
        log_info('All cases have same spatial resolution. Data is good for running multi-shell-dMRIharmonization')
    log_info('')


def consistencyCheck(ref_csv, outputBshellFile= None, outPutResolutionFile= None):

    try:
        ref_imgs, _ = read_imgs_masks(ref_csv)
    except:
        ref_imgs = read_imgs(ref_csv)


    if isfile(outputBshellFile) and isfile(outPutResolutionFile):
        ref_bvals= read_bvals(outputBshellFile)
        ref_res = np.load(outPutResolutionFile)
    else:
        ref_bshell_img = ref_imgs[0]
        log_info(f'Using {ref_bshell_img} to determine b-shells')

        inPrefix = abspath(ref_bshell_img).split('.nii')[0]
        ref_bvals = findBShells(inPrefix + '.bval', outputBshellFile)

        ref_res = load(ref_bshell_img).header['pixdim'][1:4]
        np.save(outPutResolutionFile, ref_res)


    log_info('b-shells are', ref_bvals)

    log_info('\nSite', ref_csv, '\n')

    log_info('Checking consistency of b-shells among subjects')
    check_bshells(ref_imgs, ref_bvals)


    log_info('spatial resolution is', ref_res)
    log_info('Checking consistency of spatial resolution among subjects')
    check_resolution(ref_imgs, ref_res)



if __name__ == '__main__':
    if len(sys.argv)==1 or sys.argv[1]=='-h' or sys.argv[1]=='--help':
        log_info('''Check consistency of b-shells and spatial resolution among subjects
Usage:
consistencyCheck list.csv/txt ref_bshell_bvalues.txt ref_res_file.npy

Provide a csv/txt file with first column for dwi and 2nd column for mask: dwi1,mask1\\ndwi2,mask2\\n...
or just one column for dwi1\\ndwi2\\n...
In addition, provide ref_bshell_bvalues and ref_res_file.''')
        exit()

    ref_csv= abspath(sys.argv[1])
    outputBshellFile= abspath(sys.argv[2])
    outPutResolutionFile= abspath(sys.argv[3])
    if isfile(ref_csv):
        consistencyCheck(ref_csv, outputBshellFile, outPutResolutionFile)
    else:
        raise FileNotFoundError(f'{ref_csv} does not exists.')

